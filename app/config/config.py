import copy
import errno
import os
import shutil
import socket
import tempfile
import threading
from contextlib import contextmanager

import toml
from loguru import logger

from app import __version__

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"
_CONTAINER_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "podman")
_DOCKER_HOST_GATEWAY_NAME = "host.docker.internal"
_config_save_lock = threading.RLock()
_pending_config_lock = threading.RLock()
_pending_config_updates = {}
_pending_config_save_requested = False
_pending_config_flush_scheduled = False
_MISSING = object()
_DELETE = object()
_UTF8_BOM = "\ufeff"


class _SynchronizedConfig(dict):
    """Сохраняет привычный интерфейс dict и одновременно подчиняет запись рантайм-конфигурации одному общему локу."""

    def __setitem__(self, key, value):
        # Каждый полный rerun Streamlit заново записывает в конфигурацию текущие
        # значения виджетов. Пока задача видео держит runtime_config_lock, запись
        # неизменённого значения не имеет побочных эффектов и не должна подвешивать
        # обновлённую страницу посреди формы. Записи, реально меняющие конфигурацию,
        # по-прежнему уходят под лок ниже, поэтому сменить провайдера, ключи или другие
        # глобальные настройки посреди генерации видео невозможно.
        current = super().get(key, _MISSING)
        if current is not _MISSING and current == value:
            return
        with _config_save_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with _config_save_lock:
            super().__delitem__(key)

    def clear(self):
        if not self:
            return
        with _config_save_lock:
            super().clear()

    def pop(self, key, default=_MISSING):
        # ``pop(key, default)`` при отсутствии ключа тоже ничего не меняет. WebUI
        # выражает так «использовать стратегию по умолчанию», и при обновлении страницы это обязано отрабатывать сразу.
        if key not in self:
            if default is _MISSING:
                raise KeyError(key)
            return default
        with _config_save_lock:
            if default is _MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def setdefault(self, key, default=None):
        # Как и у __setitem__, setdefault для уже существующего ключа — операция чтения.
        # Ранний возврат избавляет страницы, которые лишь читают конфигурацию по умолчанию, от влияния лока длинной задачи.
        current = super().get(key, _MISSING)
        if current is not _MISSING:
            return current
        with _config_save_lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs):
        changes = dict(*args, **kwargs)
        if all(
            (current := dict.get(self, key, _MISSING)) is not _MISSING
            and current == value
            for key, value in changes.items()
        ):
            return
        with _config_save_lock:
            super().update(changes)


def _pending_update_key(config_section, key):
    """Формирует ключ отложенного обновления для фиксированной секции конфигурации внутри процесса."""
    return id(config_section), key


def update_config_nonblocking(config_section, key, value):
    """
    Неблокирующее обновление рантайм-конфигурации WebUI.

    Генерация видео держит ``runtime_config_lock``, чтобы одна задача не сменила
    провайдера, ключи или настройки голоса посреди выполнения. Изменение виджета
    Streamlit не может ждать этот лок длинной задачи — иначе браузер выглядит
    зависшим. Если лок свободен, значение применяется сразу; если занят,
    сохраняется только последнее значение каждого параметра и применяется
    целиком, когда текущая задача отпустит лок.

    True означает, что значение уже вступило в силу, False — что оно попало в
    очередь отложенных обновлений.
    """
    # Все обновления сперва попадают в одну очередь и лишь затем пытаются взять лок
    # конфигурации. Тогда при одновременной правке одного параметра с разных страниц
    # итоговый порядок совпадает с порядком записи в очередь, и более ранний поток,
    # получив лок, не затрёт значение, которое более поздний уже поставил в очередь.
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            copy.deepcopy(value),
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        # Вызывающая сторона обычно запрашивает сохранение в конце текущего rerun
        # Streamlit, но полагаться на этот шаг нельзя. Например, при исключении посреди
        # страницы или если обновление пришлось ровно на этап финального сохранения задачи, довести значения из очереди обязан фоновый поток обновления.
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return config_section.get(key, _MISSING) == value
    finally:
        _config_save_lock.release()


def delete_config_nonblocking(config_section, key):
    """
    Неблокирующее удаление параметра конфигурации WebUI.

    «Использовать значение по умолчанию» означает по-настоящему убрать параметр,
    а не записать пустую строку. Пока задача видео занимает лок конфигурации,
    намерение удалить перекрывает ранее поставленные в очередь обновления того же
    параметра и выполняется после завершения задачи.
    """
    with _pending_config_lock:
        _pending_config_updates[_pending_update_key(config_section, key)] = (
            config_section,
            key,
            _DELETE,
        )

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        _apply_pending_config_updates_locked()
        return key not in config_section
    finally:
        _config_save_lock.release()


def _apply_pending_config_updates_locked():
    """Применяет последние отложенные значения конфигурации WebUI, удерживая лок записи."""
    with _pending_config_lock:
        updates = list(_pending_config_updates.values())
        _pending_config_updates.clear()
        # Применяя конфигурацию, продолжаем держать лок очереди обновлений. Поток,
        # читающий снимок «текущие значения плюс отложенные», увидит либо полное состояние до применения, либо полное после — но не набор, обновлённый наполовину.
        for config_section, key, value in updates:
            if value is _DELETE:
                config_section.pop(key, None)
            else:
                config_section[key] = value
    return bool(updates)


def snapshot_config_with_pending(config_section):
    """
    Возвращает актуальный снимок секции конфигурации, объединённый с ещё не
    применёнными обновлениями WebUI.

    Пока задача видео держит лок, менять глобальную конфигурацию нельзя, но
    пользователь уже может готовить следующий ролик. Благодаря снимку только что
    выбранные в интерфейсе провайдер, модель и ключ участвуют в новом запросе к
    LLM и при этом не влияют на выполняющуюся задачу видео.
    """
    with _pending_config_lock:
        snapshot = dict(config_section)
        section_id = id(config_section)
        for (pending_section_id, key), (_, _, value) in _pending_config_updates.items():
            if pending_section_id != section_id:
                continue
            if value is _DELETE:
                snapshot.pop(key, None)
            else:
                snapshot[key] = copy.deepcopy(value)
    return snapshot


def _flush_pending_config_locked(*, suppress_save_errors):
    """Применяет и сохраняет все отложенные настройки, удерживая лок записи конфигурации."""
    global _pending_config_save_requested

    updates_applied = _apply_pending_config_updates_locked()
    with _pending_config_lock:
        save_requested = _pending_config_save_requested
        _pending_config_save_requested = False

    if not updates_applied and not save_requested:
        return True

    try:
        save_config()
        return True
    except Exception as exc:
        # Конфигурация в памяти уже применена успешно, поэтому при неудачном сохранении
        # остаётся только отметка «нужно сохранить». Временно недоступный для записи файл конфигурации не должен превращать задачу видео в упавшую; следующее взаимодействие со страницей снова вызовет сохранение.
        with _pending_config_lock:
            _pending_config_save_requested = True
        if not suppress_save_errors:
            raise
        logger.exception(f"failed to save deferred runtime config: {exc}")
        return False


def _run_deferred_config_flush():
    """Дожидается освобождения лока конфигурации длинной задачей и надёжно разгребает накопившиеся обновления."""
    global _pending_config_flush_scheduled

    while True:
        with _config_save_lock:
            flush_succeeded = _flush_pending_config_locked(
                suppress_save_errors=True
            )

        with _pending_config_lock:
            has_pending_work = bool(
                _pending_config_updates or _pending_config_save_requested
            )
            if not flush_succeeded or not has_pending_work:
                _pending_config_flush_scheduled = False
                return


def _schedule_deferred_config_flush():
    """Гарантирует, что обновления конфигурации ждёт не более одного фонового потока одновременно."""
    global _pending_config_flush_scheduled

    with _pending_config_lock:
        if _pending_config_flush_scheduled:
            return
        _pending_config_flush_scheduled = True

    threading.Thread(
        target=_run_deferred_config_flush,
        name="mpt-config-flush",
        daemon=True,
    ).start()


def try_save_config():
    """
    Неблокирующее сохранение конфигурации WebUI; если лок занят, сохранение
    выполнит текущая длинная задача при завершении.

    Обычные API, CLI и обслуживающие скрипты по-прежнему могут вызывать
    ``save_config`` и получать прежнюю блокирующую семантику записи. Эту функцию
    использует только rerun Streamlit, чтобы страница подолгу не зависала в
    ожидании задачи видео.
    """
    global _pending_config_save_requested

    with _pending_config_lock:
        _pending_config_save_requested = True

    acquired = _config_save_lock.acquire(blocking=False)
    if not acquired:
        _schedule_deferred_config_flush()
        return False

    try:
        return _flush_pending_config_locked(suppress_save_errors=False)
    finally:
        _config_save_lock.release()


@contextmanager
def runtime_config_lock():
    """
    Не даёт другим сессиям WebUI менять конфигурацию на протяжении целой
    операции, которая зависит от глобальных настроек.

    Проект по умолчанию слушает локальный адрес обратной петли, а конфигурация
    остаётся однопользовательской и глобальной. Этот лёгкий лок защищает прежде
    всего долгие операции вроде генерации и прослушивания, чтобы другая вкладка
    не сменила провайдера или ключ посреди работы.
    """
    with _config_save_lock:
        # Если фоновый поток обновления ещё не получил управление после того, как
        # предыдущая короткая операция отпустила лок, новая задача обязана применить очередь до чтения провайдера, ключей и прочих глобальных настроек: гонять весь конвейер на старой конфигурации нельзя.
        _flush_pending_config_locked(suppress_save_errors=True)
        try:
            yield
        finally:
            _flush_pending_config_locked(suppress_save_errors=True)


@contextmanager
def try_runtime_config_lock():
    """
    Пытается взять лок рантайм-конфигурации и сразу сообщает, удалось ли это.

    Прослушивание в WebUI — короткая операция по инициативе пользователя, и ей не
    место в многоминутном ожидании, пока лок держит фоновая задача видео. Не
    получив лок, вызывающая сторона может тут же предложить повторить попытку
    позже; при успешном захвате настройки провайдера, ключа и модели на время
    прослушивания гарантированно не изменит другая сессия.
    """
    acquired = _config_save_lock.acquire(blocking=False)
    try:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
        yield acquired
    finally:
        if acquired:
            _flush_pending_config_locked(suppress_save_errors=True)
            _config_save_lock.release()


def is_running_in_container(
    dockerenv_path: str = "/.dockerenv",
    containerenv_path: str = "/run/.containerenv",
    cgroup_path: str = "/proc/1/cgroup",
) -> bool:
    """
    Определяет, выполняется ли текущий процесс внутри контейнера.

    Проверка нужна главным образом для выбора адреса Ollama по умолчанию:
    - при обычном локальном запуске `localhost` указывает на машину пользователя;
    - внутри контейнера Docker `localhost` указывает на сам контейнер, и чтобы
      достучаться до Ollama на хосте, обычно нужен `host.docker.internal`.

    Нельзя опираться на одно лишь наличие `/proc/1/cgroup`: этот файл есть и в
    обычном Linux. Здесь True возвращается только при явных признаках контейнера,
    чтобы не задеть пользователей Linux без Docker. Путь оставлен внедряемым
    параметром, чтобы юнит-тесты покрывали разные окружения.
    """
    if os.path.isfile(dockerenv_path) or os.path.isfile(containerenv_path):
        return True

    try:
        with open(cgroup_path, mode="r", encoding="utf-8") as fp:
            cgroup_content = fp.read().lower()
    except OSError:
        return False

    return any(marker in cgroup_content for marker in _CONTAINER_CGROUP_MARKERS)


def _can_resolve_hostname(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
    except OSError:
        return False
    return True


def _decode_linux_route_gateway(hex_gateway: str) -> str:
    # Gateway в /proc/net/route записан шестнадцатерично в порядке little-endian:
    # например, 010011AC означает 172.17.0.1. Разбираем его отдельно, чтобы в
    # нативном Linux-Docker без DNS-записи host.docker.internal всё же попробовать достучаться до хоста через шлюз контейнера по умолчанию.
    if len(hex_gateway) != 8:
        raise ValueError("invalid gateway length")

    octets = [
        str(int(hex_gateway[index : index + 2], 16)) for index in range(6, -1, -2)
    ]
    return ".".join(octets)


def get_container_default_gateway_ip(route_path: str = "/proc/net/route") -> str:
    """
    Читает IP шлюза по умолчанию внутри Linux-контейнера.

    Docker Desktop обычно предоставляет `host.docker.internal`, а нативный
    Linux-Docker это DNS-имя по умолчанию может и не давать. Шлюз по умолчанию
    обычно годится как запасной адрес для доступа к сервисам хоста. Если Ollama у
    пользователя слушает только 127.0.0.1, ему всё равно придётся заставить её
    слушать сетевой интерфейс хоста или вручную задать `ollama_base_url`.
    """
    try:
        with open(route_path, mode="r", encoding="utf-8") as fp:
            route_lines = fp.readlines()
    except OSError:
        return ""

    for line in route_lines[1:]:
        fields = line.strip().split()
        if len(fields) < 3:
            continue

        destination = fields[1]
        gateway = fields[2]
        if destination != "00000000" or gateway == "00000000":
            continue

        try:
            return _decode_linux_route_gateway(gateway)
        except ValueError:
            logger.warning(f"invalid container gateway route entry: {line.strip()}")
            return ""

    return ""


def get_default_ollama_base_url() -> str:
    """
    Возвращает base_url OpenAI-совместимого API Ollama по умолчанию.

    Если пользователь задал `ollama_base_url` явно, сюда управление не попадает:
    здесь определяется только «лучшее значение по умолчанию, когда настройки
    нет». Внутри контейнера оно указывает на хост, при обычном локальном запуске —
    на localhost.
    """
    if not is_running_in_container():
        return "http://localhost:11434/v1"

    if _can_resolve_hostname(_DOCKER_HOST_GATEWAY_NAME):
        return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"

    gateway_ip = get_container_default_gateway_ip()
    if gateway_ip:
        logger.info(
            "host.docker.internal is not resolvable, fallback to container "
            f"default gateway for Ollama: {gateway_ip}"
        )
        return f"http://{gateway_ip}:11434/v1"

    logger.warning(
        "failed to resolve host.docker.internal and container default gateway; "
        "fallback to host.docker.internal for Ollama"
    )
    return f"http://{_DOCKER_HOST_GATEWAY_NAME}:11434/v1"


def _load_toml_config(config_path: str):
    """
    Загружает TOML, корректно обрабатывая повторный UTF-8 BOM, который могут
    записать редакторы под Windows.

    ``utf-8-sig`` убирает только один BOM в начале файла. Некоторые редакторы
    Windows либо процедуры распаковки и сохранения могут записать BOM ещё раз, и
    второй невидимый символ попадёт в парсер TOML и вызовет ошибку на первой
    строке. Здесь нормализация только на чтение выполняется единожды и лишь после
    неудачного стандартного разбора; исходный файл не переписывается, чтобы
    случайно не затереть уже введённые пользователем API-ключи.
    """
    try:
        return toml.load(config_path)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "load config failed, retry with UTF-8 BOM compatibility: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )

    try:
        with open(config_path, mode="r", encoding="utf-8-sig") as fp:
            config_content = fp.read()

        normalized_content = config_content.lstrip(_UTF8_BOM)
        removed_bom_count = len(config_content) - len(normalized_content)
        if removed_bom_count:
            logger.warning(
                "removed repeated UTF-8 BOM characters while loading config: "
                f"path={config_path}, count={removed_bom_count}"
            )
        return toml.loads(normalized_content)
    except (toml.TomlDecodeError, UnicodeDecodeError) as exc:
        logger.error(
            "config file is not valid TOML after UTF-8 BOM normalization: "
            f"path={config_path}, error={type(exc).__name__}: {exc}"
        )
        raise


def load_config():
    # fix: IsADirectoryError: [Errno 21] Is a directory: '/MoneyPrinterTurbo/config.toml'
    if os.path.isdir(config_file):
        shutil.rmtree(config_file)

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            shutil.copyfile(example_file, config_file)
            logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")

    return _load_toml_config(config_file)


def save_config():
    """
    Атомарно сохраняет рантайм-конфигурацию.

    Разные сессии Streamlit могут запустить сохранение почти одновременно. При
    прямой перезаписи config.toml другой поток может прочитать частично
    записанный TOML. Здесь сохранение сериализуется реентерабельным локом внутри
    процесса: сперва пишется временный файл в том же каталоге, затем os.replace
    атомарно подменяет целевой.

    В Docker Desktop bind mount одиночного файла делает точкой монтирования сам
    config.toml, а ядро Linux не позволяет заменить точку монтирования через
    rename/replace и возвращает EBUSY. В этом случае остаётся перезаписать файл
    на месте под локом; прочие исключения по-прежнему пробрасываются, чтобы не
    скрыть ошибки прав, диска или пути.

    Это сохраняет существующую в проекте семантику однопользовательской
    глобальной конфигурации и не тянет за собой сложную многопользовательскую
    систему настроек; задача — не испортить файл конфигурации при нескольких
    вкладках или быстрых rerun.
    """
    with _config_save_lock:
        config_to_save = dict(_cfg)
        config_to_save["app"] = dict(app)
        config_to_save["azure"] = dict(azure)
        config_to_save["siliconflow"] = dict(siliconflow)
        config_to_save["minimax_tts"] = dict(minimax_tts)
        config_to_save["elevenlabs"] = dict(elevenlabs)
        config_to_save["chatterbox"] = dict(chatterbox)
        config_to_save["fish_audio"] = dict(fish_audio)
        config_to_save["ui"] = dict(ui)
        serialized_config = toml.dumps(config_to_save)

        # По завершении полного rerun WebUI вызывает сохранение. Если содержимое не
        # изменилось, сразу выходим: незачем на каждый клик по обычному виджету делать запись на диск и fsync.
        try:
            with open(config_file, mode="r", encoding="utf-8") as f:
                if f.read() == serialized_config:
                    _cfg.clear()
                    _cfg.update(config_to_save)
                    return
        except (OSError, UnicodeError):
            pass

        temp_path = ""
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".config-",
                suffix=".toml.tmp",
                dir=root_dir,
            )
            with os.fdopen(fd, mode="w", encoding="utf-8") as f:
                f.write(serialized_config)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(temp_path, config_file)
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise

                logger.warning(
                    "atomic config replacement is unavailable for the mounted "
                    f"file, fallback to in-place write: {config_file}"
                )
                with open(config_file, mode="w", encoding="utf-8") as f:
                    f.write(serialized_config)
                    f.flush()
                    os.fsync(f.fileno())
            _cfg.clear()
            _cfg.update(config_to_save)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


_cfg = load_config()
app = _SynchronizedConfig(_cfg.get("app", {}))
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _SynchronizedConfig(_cfg.get("azure", {}))
siliconflow = _SynchronizedConfig(_cfg.get("siliconflow", {}))
minimax_tts = _SynchronizedConfig(_cfg.get("minimax_tts", {}))
elevenlabs = _SynchronizedConfig(_cfg.get("elevenlabs", {}))
chatterbox = _SynchronizedConfig(_cfg.get("chatterbox", {}))
fish_audio = _SynchronizedConfig(_cfg.get("fish_audio", {}))
ui = _SynchronizedConfig(
    _cfg.get(
        "ui",
        {
            "hide_log": False,
        },
    )
)

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", __version__)
reload_debug = False

app["redis_host"] = os.getenv(
    "MPT_APP_REDIS_HOST",
    os.getenv("REDIS_HOST", app.get("redis_host", "localhost")),
)

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
