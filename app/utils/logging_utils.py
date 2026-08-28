import os
import threading

from loguru import logger


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
LOG_RECORD_FORMAT = (
    "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
    "<level>{level}</> | "
    '"{file.path}:{line}":<blue> {function}</> '
    "- <level>{message}</>\n"
)
# При старте Loguru ID терминального handler по умолчанию равен 0. При
# перезагрузке WebUI можно заменить только этот базовый вывод в терминал;
# вызывать logger.remove() и стирать все handler нельзя, иначе удалится и
# временный sink, которым выполняющаяся задача собирает логи для WebUI.
_terminal_handler_id: int | None = 0
_terminal_handler_lock = threading.RLock()


def _project_relative_path(file_path):
    """
    Сокращает абсолютный путь до относительного пути проекта, который начинается
    с ``./`` и всегда использует прямые слэши.

    В Windows проект может быть запущен с примонтированного сетевого диска или
    диска ``subst``. Тогда путь в стеке вызовов остаётся видом
    ``X:\\MoneyPrinterTurbo\\...``, а ``PROJECT_ROOT`` после ``realpath``
    оказывается в ``C:\\...``, и ``os.path.relpath`` сразу бросает ``ValueError``.
    Исключение в функции форматирования loguru перехватит и выбросит запись
    целиком — терминал и панель логов WebUI опустеют одновременно, поэтому здесь
    обязателен откат к исходному пути. С файлами вне каталога проекта то же самое:
    приклеивание ``./`` к пути с ``..`` даёт только менее читаемый результат.
    """
    try:
        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
    except ValueError:
        return file_path
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        return file_path
    # В Windows relpath возвращает путь с обратными слэшами, и прямая склейка даёт
    # смесь разделителей вида ``./app\\utils``, что расходится с логами на других платформах.
    return f"./{relative_path.replace(os.sep, '/')}"


def format_log_record(record):
    """
    Единообразно форматирует логи терминала и WebUI.

    Loguru отдаёт одну и ту же запись нескольким sink. Первый sink мог уже
    превратить абсолютный путь в относительный путь проекта, поэтому здесь
    поддерживаются и абсолютные пути, и уже отформатированные, начинающиеся
    с ``./``. Sink WebUI отключает цвет, но время, уровень, место вызова и текст
    сообщения совпадают с терминалом.
    """
    file_path = record["file"].path
    if os.path.isabs(file_path):
        record["file"].path = _project_relative_path(file_path)

    # В сообщении лога иногда встречается абсолютный путь к файлу задачи. Единое
    # сокращение до пути относительно проекта не даёт WebUI и терминалу показывать разное из-за отличий в точке инициализации.
    record["message"] = record["message"].replace(PROJECT_ROOT, ".")
    return LOG_RECORD_FORMAT


def configure_terminal_logger(sink, level: str, colorize: bool = True) -> int:
    """
    Безопасно заменяет общепроцессный терминальный handler логов, сохраняя
    handler отдельных задач.

    При горячей перезагрузке кода или сбросе кэша Streamlit может заново
    выполнить инициализацию логирования. Здесь старый вывод в терминал удаляется
    точно по сохранённому ID handler, поэтому логи WebUI, которые пишет фоновая
    задача, не прерываются. Лок защищает обновление ID, когда инициализация идёт
    сразу из нескольких сессий браузера.
    """
    global _terminal_handler_id

    with _terminal_handler_lock:
        if _terminal_handler_id is not None:
            try:
                logger.remove(_terminal_handler_id)
            except ValueError:
                # Тест или внешняя точка входа могли уже удалить этот handler. Продолжаем
                # создавать новый вывод в терминал, не затрагивая остальные ещё живые sink.
                pass

        _terminal_handler_id = logger.add(
            sink,
            level=level,
            format=format_log_record,
            colorize=colorize,
        )
        return _terminal_handler_id
