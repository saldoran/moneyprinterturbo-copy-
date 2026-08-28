import ast
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from app.models import const
from app.models.schema import VideoParams
from app.services import webui_task
from app.utils import logging_utils


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _attribute_name(node):
    """Приводит AST-вызов вида ``module.function`` к устойчивой строке."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return ".".join(reversed(names))


def _log_record(file_path, message="generation finished"):
    """Собирает минимальную запись loguru, нужную для ``format_log_record``."""
    return {
        "file": SimpleNamespace(name=os.path.basename(file_path), path=file_path),
        "message": message,
    }


def test_generation_controls_submit_background_task_instead_of_blocking_page():
    """
    Кнопка генерации в WebUI не должна снова напрямую вызывать синхронный
    конвейер.

    Это основная защита от регрессии белого экрана из Issue #1120: пока скрипт
    полной страницы снова блокируется на ``tm.start``, пользователь, обновивший
    страницу во время генерации, может получить delta, указывающую на старое
    дерево отрисовки.
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_generation_controls"
    )
    calls = {
        _attribute_name(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }

    assert "webui_task.submit_generation" in calls
    assert "tm.start" not in calls


def test_webui_runtime_config_updates_do_not_use_blocking_writes():
    """
    Обычный rerun виджета во время генерации не должен снова ждать лок
    конфигурации, удерживаемый длинной задачей.

    Любая запись конфигурации в WebUI обязана идти через неблокирующий helper.
    Тест подключения к LLM и прослушивание голоса могут быстро вернуться по try
    lock, но код страницы не вправе напрямую вызывать блокирующий лок или
    блокирующую функцию сохранения.
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    calls = {
        _attribute_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "config.runtime_config_lock" not in calls
    assert "config.save_config" not in calls
    assert not calls.intersection(
        {
            "config.app.clear",
            "config.app.pop",
            "config.app.setdefault",
            "config.app.update",
            "config.azure.clear",
            "config.azure.pop",
            "config.azure.setdefault",
            "config.azure.update",
            "config.chatterbox.clear",
            "config.chatterbox.pop",
            "config.chatterbox.setdefault",
            "config.chatterbox.update",
            "config.elevenlabs.clear",
            "config.elevenlabs.pop",
            "config.elevenlabs.setdefault",
            "config.elevenlabs.update",
            "config.siliconflow.clear",
            "config.siliconflow.pop",
            "config.siliconflow.setdefault",
            "config.siliconflow.update",
            "config.ui.clear",
            "config.ui.pop",
            "config.ui.setdefault",
            "config.ui.update",
        }
    )

    synchronized_sections = {
        "app",
        "azure",
        "chatterbox",
        "elevenlabs",
        "siliconflow",
        "ui",
    }
    direct_writes = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]

        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            section = target.value
            if (
                isinstance(section, ast.Attribute)
                and isinstance(section.value, ast.Name)
                and section.value.id == "config"
                and section.attr in synchronized_sections
            ):
                direct_writes.append(node.lineno)

    assert direct_writes == []


@pytest.mark.parametrize(
    ("ui_config", "expected_open_count"),
    [
        ({}, 1),
        ({"open_task_folder_on_completion": True}, 1),
        ({"open_task_folder_on_completion": False}, 0),
    ],
)
def test_completed_task_renders_subject_named_video_download(
    tmp_path, ui_config, expected_open_count
):
    """Завершённая задача даёт скачать готовый ролик, а открытие каталога определяется настройкой WebUI."""
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected_nodes = []
    target_names = {
        "_DOWNLOAD_FILENAME_INVALID_PATTERN",
        "_build_video_download_name",
        "_normalize_task_state",
        "_render_generation_task_snapshot",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in target_names
            for target in node.targets
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in target_names:
            selected_nodes.append(node)

    class FakeColumn:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.downloads = []
            self.videos = []

        def columns(self, count):
            return [FakeColumn() for _ in range(count)]

        def video(self, video_path):
            self.videos.append(video_path)

        def download_button(self, label, data, **kwargs):
            self.downloads.append((label, data.read(), kwargs))

        def success(self, _message):
            pass

        def warning(self, _message):
            pass

        def error(self, _message):
            pass

    video_path = tmp_path / "final-1.mp4"
    video_path.write_bytes(b"video-content")
    fake_st = FakeStreamlit()
    open_task_folder = MagicMock()
    namespace = {
        "Mapping": Mapping,
        "config": SimpleNamespace(ui=ui_config),
        "const": const,
        "logger": MagicMock(),
        "mimetypes": __import__("mimetypes"),
        "open_task_folder": open_task_folder,
        "os": os,
        "re": re,
        "st": fake_st,
        "tr": lambda key: key,
        "_render_generation_logs": lambda _task_id: None,
    }
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)

    namespace["_render_generation_task_snapshot"](
        "download-test",
        {
            "state": const.TASK_STATE_COMPLETE,
            "progress": 100,
            "videos": [str(video_path)],
            "warnings": [],
            "video_subject": "A day: in / Shanghai?",
        },
    )

    assert fake_st.videos == [str(video_path)]
    assert fake_st.downloads == [
        (
            "Download Video",
            b"video-content",
            {
                "file_name": "A day in Shanghai.mp4",
                "mime": "video/mp4",
                "key": "download_generated_video_download-test_0",
                "icon": ":material/download:",
                "on_click": "ignore",
                "use_container_width": True,
            },
        )
    ]
    assert open_task_folder.call_count == expected_open_count
    if expected_open_count:
        open_task_folder.assert_called_once_with("download-test")


def test_submit_generation_returns_while_pipeline_is_still_running():
    """Пока фоновый конвейер ещё работает, функция отправки обязана уже вернуть управление, чтобы Streamlit завершил текущую отрисовку."""
    task_id = "background-submit-test"
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_start(**_kwargs):
        started.set()
        release.wait(timeout=5)
        finished.set()
        return {"videos": ["/tmp/final-1.mp4"]}

    params = VideoParams(video_subject="异步生成测试")
    try:
        with (
            patch.object(webui_task.tm, "start", side_effect=blocking_start),
            patch.object(
                webui_task.config,
                "runtime_config_lock",
                return_value=nullcontext(),
            ),
        ):
            started_at = time.monotonic()
            webui_task.submit_generation(task_id, params, capture_logs=False)
            elapsed = time.monotonic() - started_at

            assert started.wait(timeout=2)
            assert elapsed < 0.5
            assert not finished.is_set()
            task = webui_task.sm.state.get_task(task_id)
            assert task["state"] == const.TASK_STATE_PROCESSING
    finally:
        release.set()
        assert finished.wait(timeout=2)
        webui_task.sm.state.delete_task(task_id)


def test_submit_generation_copies_params_before_starting_worker():
    """Ни последующий rerun страницы, ни правка параметров внутри конвейера не должны загрязнять текущий объект формы."""
    params = VideoParams(video_subject="参数隔离测试")
    with patch.object(webui_task._task_manager, "add_task") as add_task:
        webui_task.submit_generation("copied-params-test", params, capture_logs=False)

    submitted_params = add_task.call_args.kwargs["params"]
    assert submitted_params == params
    assert submitted_params is not params
    webui_task.sm.state.delete_task("copied-params-test")


def test_scheduling_failure_is_saved_as_terminal_task_state():
    """Сбой очереди или запуска потока не должен оставлять менеджер задач в состоянии «генерируется» навсегда."""
    task_id = "scheduling-failure-test"
    params = VideoParams(video_subject="调度失败测试")
    with patch.object(
        webui_task._task_manager,
        "add_task",
        side_effect=RuntimeError("worker unavailable"),
    ):
        with pytest.raises(RuntimeError, match="worker unavailable"):
            webui_task.submit_generation(task_id, params, capture_logs=False)

    task = webui_task.sm.state.get_task(task_id)
    assert task["state"] == const.TASK_STATE_FAILED
    assert task["failed_stage"] == "scheduling"
    assert task["error"] == "RuntimeError: worker unavailable"
    webui_task.sm.state.delete_task(task_id)


def test_worker_logs_are_available_without_streamlit_session_state():
    """Фоновые логи пишутся в потокобезопасный буфер, и странице достаточно опрашивать снимок, чтобы показывать их в реальном времени."""
    task_id = "captured-log-test"
    with webui_task._task_logs_lock:
        webui_task._task_logs.pop(task_id, None)

    def logged_start(**_kwargs):
        logger.info("unique background task log")
        return {"videos": ["/tmp/final-1.mp4"]}

    with (
        patch.object(webui_task.tm, "start", side_effect=logged_start),
        patch.object(
            webui_task.config,
            "runtime_config_lock",
            return_value=nullcontext(),
        ),
    ):
        result = webui_task._run_generation(
            task_id,
            VideoParams(video_subject="日志测试"),
            capture_logs=True,
        )

    assert result == {"videos": ["/tmp/final-1.mp4"]}
    records = webui_task.get_task_logs(task_id)
    assert len(records) == 1
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO \| "
        r'"\./test/services/test_webui_task\.py:\d+": logged_start '
        r"- unique background task log",
        records[0],
    )


def test_log_paths_stay_posix_style_on_every_platform():
    """
    Место вызова обязано всегда отображаться как ``./app/services/task.py``.

    ``os.path.relpath`` в Windows возвращает путь с обратными слэшами, и прямая
    склейка дала бы ``./app\\services\\task.py``: один и тот же лог выглядел бы
    по-разному в разных системах и не совпал бы с расположенным выше регрессным
    тестом фоновых логов, который проверяет прямые слэши.
    """
    record = _log_record(
        os.path.join(logging_utils.PROJECT_ROOT, "app", "services", "task.py")
    )

    logging_utils.format_log_record(record)

    assert record["file"].path == "./app/services/task.py"


def test_log_paths_on_another_mount_do_not_discard_the_record():
    """
    Запуск с примонтированного или ``subst``-диска не должен приводить к
    исчезновению всей строки лога.

    При таком развёртывании путь в стеке вызовов остаётся на ``X:``, тогда как
    ``PROJECT_ROOT`` уже разрешён realpath обратно в ``C:``, и
    ``os.path.relpath`` бросает ``ValueError``. Перехватив исключение
    форматирования, loguru выбрасывает запись, и терминал вместе с панелью логов
    WebUI пустеют одновременно.
    """
    absolute_path = os.path.join(
        logging_utils.PROJECT_ROOT, "app", "services", "task.py"
    )
    record = _log_record(absolute_path)

    with patch.object(
        logging_utils.os.path,
        "relpath",
        side_effect=ValueError("path is on mount 'X:', start on mount 'C:'"),
    ):
        log_format = logging_utils.format_log_record(record)

    assert log_format == logging_utils.LOG_RECORD_FORMAT
    assert record["file"].path == absolute_path


def test_log_paths_outside_the_project_keep_the_absolute_path():
    """Файлы вне каталога проекта сохраняют абсолютный путь, чтобы не выводить обратные пути вида ``./../..``."""
    outside_path = os.path.join(
        os.path.dirname(logging_utils.PROJECT_ROOT), "site-packages", "worker.py"
    )
    record = _log_record(outside_path)

    logging_utils.format_log_record(record)

    assert record["file"].path == outside_path


def test_generation_log_fragment_refreshes_within_half_a_second():
    """Интервал опроса логов не должен возвращаться к секундному обновлению, заметно отстающему от вывода в терминал."""
    assert webui_task.TASK_LOG_REFRESH_INTERVAL_SECONDS <= 0.5

    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_running_generation_task"
    )
    decorator = function.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert _attribute_name(decorator.func) == "st.fragment"
    run_every = next(
        keyword.value for keyword in decorator.keywords if keyword.arg == "run_every"
    )
    assert ast.unparse(run_every) == ("webui_task.TASK_LOG_REFRESH_INTERVAL_SECONDS")


def test_generation_submit_skips_duplicate_config_save():
    """
    После отправки задачи в конце страницы нельзя снова ждать лок конфигурации.

    Фоновая задача держит runtime_config_lock всю генерацию. Ветка генерации уже
    запросила неблокирующее сохранение, и повторять запрос в конце страницы
    незачем; обычные взаимодействия продолжают сохраняться через тот же
    неблокирующий helper и не должны откатываться к config.save_config.
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    controls = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_generation_controls"
    )
    application = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_render_application"
    )

    assert isinstance(controls.body[-1], ast.Return)
    assert ast.unparse(controls.body[-1].value) == "start_button"

    submitted_assignment = next(
        node
        for node in application.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "generation_submitted"
            for target in node.targets
        )
    )
    assert isinstance(submitted_assignment.value, ast.Call)
    assert _attribute_name(submitted_assignment.value.func) == (
        "_render_generation_controls"
    )

    guarded_save = next(
        node
        for node in application.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not generation_submitted"
    )
    guarded_calls = {
        _attribute_name(node.func)
        for node in ast.walk(guarded_save)
        if isinstance(node, ast.Call)
    }
    assert guarded_calls == {"_save_runtime_config"}


def test_terminal_logger_reload_preserves_task_log_handler():
    """Горячая перезагрузка вправе заменить только терминальный handler, но не очищать sink логов фоновой задачи."""
    previous_handler_id = logging_utils._terminal_handler_id
    try:
        with (
            patch.object(logging_utils.logger, "remove") as remove,
            patch.object(logging_utils.logger, "add", return_value=456) as add,
        ):
            logging_utils._terminal_handler_id = 123
            handler_id = logging_utils.configure_terminal_logger(
                sink=object(),
                level="DEBUG",
                colorize=True,
            )

        assert handler_id == 456
        remove.assert_called_once_with(123)
        add.assert_called_once()
        assert logging_utils._terminal_handler_id == 456
    finally:
        logging_utils._terminal_handler_id = previous_handler_id


def test_worker_wrapper_failure_is_saved_instead_of_leaving_processing_state():
    """Исключение в обёртке логов или конфигурации тоже обязано превратиться в наблюдаемый финальный статус ошибки."""
    task_id = "worker-wrapper-failure-test"
    with (
        patch.object(webui_task.tm, "start", side_effect=RuntimeError("lock failed")),
        patch.object(
            webui_task.config,
            "runtime_config_lock",
            return_value=nullcontext(),
        ),
    ):
        result = webui_task._run_generation(
            task_id,
            VideoParams(video_subject="工作线程失败测试"),
            capture_logs=False,
        )

    assert result["state"] == const.TASK_STATE_FAILED
    assert result["failed_stage"] == "webui_worker"
    task = webui_task.sm.state.get_task(task_id)
    assert task["state"] == const.TASK_STATE_FAILED
    assert task["error"] == "RuntimeError: lock failed"
    webui_task.sm.state.delete_task(task_id)
