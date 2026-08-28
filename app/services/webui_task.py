import threading
from collections import deque

from loguru import logger

from app.config import config
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm
from app.services.loomloom import LoomLoomConfirmedVideoRequest
from app.utils.logging_utils import format_log_record


# Конфигурация WebUI хранится в общепроцессном глобальном словаре. Прежняя
# синхронная реализация держала runtime_config_lock всю генерацию, так что разные
# сессии браузера всё равно выполнялись последовательно. Фиксируем параллелизм
# равным 1: согласованность конфигурации сохраняется, а потоки не толпятся в бессмысленном ожидании за локом конфигурации.
_task_manager = InMemoryTaskManager(
    max_concurrent_tasks=1,
    max_queued_tasks=max(1, int(config.app.get("max_queued_tasks", 100))),
)
_task_logs: dict[str, deque[str]] = {}
_task_logs_lock = threading.RLock()
_MAX_LOG_TASKS = 20
_MAX_LOG_RECORDS_PER_TASK = 1000
# Streamlit не позволяет фоновому потоку напрямую обновлять компоненты — остаётся
# опрос через Fragment. 0.5 секунды достаточно, чтобы логи в WebUI шли почти как в терминале, и при этом браузер не нагружался постоянным частым обновлением.
TASK_LOG_REFRESH_INTERVAL_SECONDS = 0.5


def _append_task_log(task_id: str, message: str) -> None:
    """Хранит ограниченное число строк лога по каждой задаче для безопасного опроса из Fragment Streamlit."""
    with _task_logs_lock:
        records = _task_logs.get(task_id)
        if records is None:
            # Держим логи только последних задач, чтобы долго работающий WebUI не наращивал
            # память. dict сохраняет порядок вставки; логи нужны лишь для диагностики в интерфейсе, и вытеснение самых старых записей на сами задачи не влияет.
            if len(_task_logs) >= _MAX_LOG_TASKS:
                oldest_task_id = next(iter(_task_logs))
                _task_logs.pop(oldest_task_id, None)
            records = deque(maxlen=_MAX_LOG_RECORDS_PER_TASK)
            _task_logs[task_id] = records
        records.append(message.rstrip())


def get_task_logs(task_id: str) -> list[str]:
    """Возвращает снимок лога, чтобы во время отрисовки страницы не держать лок, которым пользуется фоновый поток."""
    with _task_logs_lock:
        return list(_task_logs.get(task_id, ()))


def _run_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
) -> dict:
    """
    Выполняет существующий видеоконвейер в фоновом потоке.

    Sink в Loguru — общепроцессный ресурс, поэтому фильтровать его нужно по
    текущему рабочему потоку. Иначе в лог задачи подмешаются параллельные задачи
    API и записи других страниц. Страница читает только снимок обычного списка и
    не обращается к session_state Streamlit из фонового потока, что в корне
    снимает путаницу delta-путей при обновлении.
    """
    log_handler_id = None
    worker_thread_id = threading.get_ident()
    try:
        if capture_logs:
            log_handler_id = logger.add(
                lambda message: _append_task_log(task_id, str(message)),
                level="DEBUG",
                format=format_log_record,
                colorize=False,
                filter=lambda record: record["thread"].id == worker_thread_id,
            )

        # Полная задача по-прежнему берёт прежний лок конфигурации: иначе другая сессия
        # WebUI сменит провайдера, ключи и прочие общепроцессные настройки прямо посреди генерации, и одно видео окажется собрано с разными настройками.
        with config.runtime_config_lock():
            return tm.start(
                task_id=task_id,
                params=params,
                voice_preview=voice_preview,
                loomloom_video_request=loomloom_video_request,
            )
    except Exception as exc:
        # tm.start уже превращает исключения конвейера в статус ошибки. Здесь дополнительно
        # защищена обёртка WebUI — sink логов, лок конфигурации. Любое исключение фонового
        # потока обязано оставить финальный статус, чтобы менеджер задач не показывал «генерируется» вечно после выхода рабочего потока.
        error = f"{type(exc).__name__}: {exc}"
        failure = {
            "task_id": task_id,
            "state": const.TASK_STATE_FAILED,
            "progress": 0,
            "failed_stage": "webui_worker",
            "error": error,
        }
        sm.state.update_task(
            task_id,
            state=failure["state"],
            progress=failure["progress"],
            failed_stage=failure["failed_stage"],
            error=failure["error"],
        )
        logger.exception(
            f"unexpected WebUI generation worker failure, "
            f"task_id={task_id}, error={exc}"
        )
        return failure
    finally:
        if log_handler_id is not None:
            try:
                logger.remove(log_handler_id)
            except ValueError:
                logger.debug(
                    f"WebUI task log handler already removed: task_id={task_id}"
                )


def submit_generation(
    task_id: str,
    params: VideoParams,
    capture_logs: bool = True,
    voice_preview: dict | None = None,
    loomloom_video_request: LoomLoomConfirmedVideoRequest | None = None,
) -> None:
    """
    Регистрирует и отправляет задачу генерации видео из WebUI, возвращая
    управление сразу после вызова.

    Статус задачи обязан быть записан до старта потока: тогда задачу видно уже по
    окончании текущего прогона скрипта страницы, а обновление браузера или
    переподключение WebSocket не зависят от заглушки в памяти прежней страницы.
    """
    task_params = params.model_copy(deep=True)
    # Полезная нагрузка предпросмотра содержит только неизменяемый путь к аудио,
    # снимок параметров и тайминги субтитров только на чтение. Копируем верхний словарь, чтобы последующий rerun страницы, подменяя поля кэша, не задел задачу, уже отправленную в фоновую очередь.
    voice_preview_snapshot = dict(voice_preview) if voice_preview else None
    # Подтверждённый запрос — замороженный объект данных, передаваемый только внутри
    # текущего процесса. API-ключ не попадает ни в VideoParams, ни в статус задачи, ни в логи, ни в историю на диске, и не зависит от последующих rerun страницы.
    loomloom_request_snapshot = loomloom_video_request
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
        video_subject=task_params.video_subject or task_params.video_script or task_id,
    )
    try:
        _task_manager.add_task(
            _run_generation,
            task_id=task_id,
            params=task_params,
            capture_logs=capture_logs,
            voice_preview=voice_preview_snapshot,
            loomloom_video_request=loomloom_request_snapshot,
        )
    except Exception as exc:
        # Сбой планирования, как и сбой конвейера, обязан стать наблюдаемым статусом,
        # чтобы менеджер задач не показывал «генерируется» вечно. Тип исключения сохраняем, чтобы быстро найти проблему очереди в логах Docker или локальной машины.
        error = f"{type(exc).__name__}: {exc}"
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_FAILED,
            progress=0,
            failed_stage="scheduling",
            error=error,
        )
        logger.exception(
            f"failed to submit WebUI generation task, task_id={task_id}, error={exc}"
        )
        raise
