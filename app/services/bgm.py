import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from loguru import logger

from app.utils import file_security, utils


# Streamlit по умолчанию разрешает довольно крупные загрузки, но фоновая музыка
# обычно весит несколько МБ. Явный серверный лимит не даёт API или WebUI записать
# на диск огромный файл целиком и помешать задачам видео в том же процессе.
MAX_BGM_UPLOAD_BYTES = 30 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_INTERNAL_UPLOAD_PREFIX = ".bgm-upload-"
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
# Фоновую музыку в итоге декодирует FFmpeg через MoviePy, поэтому искусственно
# ограничиваться MP3 незачем. Открываем только распространённые и однозначные
# аудиорасширения, чтобы контейнеры с видео вроде MP4 не загружались как музыка.
# Тот же кортеж — единственный источник данных для контроля загрузки в WebUI, так
# что при добавлении или удалении формата бэкенд и фронтенд не разойдутся.
SUPPORTED_BGM_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
)


class BgmUploadError(ValueError):
    """Загруженный файл не удовлетворяет требованиям безопасности или формата для фоновой музыки."""


class BgmServiceError(RuntimeError):
    """Серверный сбой выполнения: недоступен FFmpeg, файловая система и тому подобное."""


def should_use_bgm(bgm_type: str | None, bgm_volume: float | None) -> bool:
    """
    Единообразно решает, нужно ли текущей задаче вообще обрабатывать фоновую
    музыку.

    Правило не зависит от источника: если источник не выбран, громкость
    некорректна или не превышает 0, то и случайный, и пользовательский, и Sonilo,
    и любой будущий поставщик обязаны пропустить разбор файла, внешнюю генерацию
    и финальное сведение. Место в общем сервисе BGM избавляет от копирования
    проверки нулевой громкости под каждого нового поставщика.
    """
    if not str(bgm_type or "").strip():
        return False
    try:
        normalized_volume = float(bgm_volume or 0)
    except (TypeError, ValueError):
        return False
    return math.isfinite(normalized_volume) and normalized_volume > 0


def uploaded_bgm_dir(create: bool = True) -> str:
    """
    Возвращает каталог постоянного хранения пользовательской фоновой музыки.

    Встроенные композиции — ресурс кода и остаются в resource/songs. Загруженное
    пользователем относится к данным рантайма и обязано лежать под смонтированным
    в Docker каталогом storage: только так оно переживёт пересоздание контейнера
    и не засорит рабочую копию Git.
    """
    return utils.storage_dir("bgm", create=create)


def _remove_staged_file(file_path: str) -> None:
    """По возможности удаляет временный файл загрузки, не подменяя исходное исключение, которое обрабатывает вызывающая сторона."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        # У временного файла зарезервированный префикс, и в список BGM он не попадает.
        # Неудачная уборка не должна затирать более точное исходное исключение вроде «некорректное аудио», но обязана оставить путь и системную ошибку для эксплуатации.
        logger.warning(
            f"failed to remove staged background music: path={file_path}, "
            f"error={str(exc)}"
        )


def sanitize_upload_filename(filename: str) -> str:
    """Извлекает имя аудиофайла, пригодное для показа на любой платформе, отклоняя недопустимые имена и неподдерживаемые расширения."""
    safe_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if (
        not safe_name
        or safe_name in {".", ".."}
        or len(safe_name) > 255
        or any(ord(character) < 32 for character in safe_name)
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in safe_name)
        or safe_name.lower().startswith(_INTERNAL_UPLOAD_PREFIX)
    ):
        raise BgmUploadError("invalid background music filename")

    # Windows считает первую часть имени до расширения именем устройства: например,
    # CON.mp3 и LPT1.wav обычными файлами не создать. Даже при том, что сервер в
    # итоге использует UUID, ранний отказ по таким именам делает поведение API на разных платформах одинаковым.
    windows_basename = safe_name.split(".", 1)[0].rstrip(" .").upper()
    if windows_basename in _WINDOWS_RESERVED_FILENAMES:
        raise BgmUploadError("invalid background music filename")
    if Path(safe_name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_BGM_EXTENSIONS
        )
        raise BgmUploadError(
            f"unsupported background music format; supported formats: {supported_formats}"
        )
    return safe_name


def _validate_audio(file_path: str, timeout_seconds: int = 30) -> None:
    """
    Проверяет, что файл содержит полностью декодируемый аудиопоток, используя
    только тот FFmpeg, который настроен в проекте.

    Проект допускает переносимый FFmpeg от imageio-ffmpeg, а такая установка не
    гарантирует наличие FFprobe, поэтому новую двоичную зависимость добавлять
    нельзя. `-map 0:a:0` завершится ошибкой, если аудиопотока нет, а `-xerror`
    превращает ошибку декодирования в отказ. Полное декодирование заодно ловит
    случаи, когда зашифрованный файл или случайные данные ненароком совпали с
    заголовком аудиокадра. Файл может содержать дополнительные потоки вроде
    обложки альбома, но проверяется только первый аудиопоток.
    """
    try:
        decoded = subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                file_path,
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BgmServiceError("FFmpeg background music validation timed out") from exc
    except OSError as exc:
        raise BgmServiceError("failed to run FFmpeg for background music validation") from exc
    if decoded.returncode != 0:
        raise BgmUploadError("uploaded file must contain a decodable audio stream")


def validate_audio_file(file_path: str, timeout_seconds: int = 120) -> None:
    """
    Проверяет, что аудиофайл на диске полностью декодируется проектным FFmpeg.

    Предпроверке при загрузке обычно хватает 30 секунд, а музыка от Sonilo может
    длиться до 6 минут, поэтому наружу отдаётся переиспользуемая точка входа с
    настраиваемым таймаутом. Сервис зависит только от FFmpeg и не требует
    отдельно устанавливать FFprobe.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
        raise BgmUploadError("background music file is empty or missing")
    _validate_audio(file_path, timeout_seconds=timeout_seconds)


def _stage_bgm_upload(filename: str, source: BinaryIO) -> tuple[str, str, int]:
    """
    Пишет поток загрузки во временный файл в том же каталоге и возвращает
    безопасное имя, временный путь и число байт.

    Предпроверка в WebUI и финальное сохранение обязаны использовать ровно
    одинаковые правила чтения по частям, ограничения размера и формирования
    имени: иначе интерфейс покажет файл пригодным, а сервер отклонит его после
    нажатия «Сгенерировать». Временный файл удаляет или атомарно подменяет
    вызывающая сторона после проверки аудио.
    """
    safe_name = sanitize_upload_filename(filename)
    try:
        target_dir = uploaded_bgm_dir(create=True)
    except OSError as exc:
        raise BgmServiceError("failed to prepare background music storage") from exc
    temp_path = ""
    total_bytes = 0

    try:
        try:
            source.seek(0)
        except (AttributeError, OSError) as exc:
            raise BgmUploadError("background music upload is not seekable") from exc

        # Сохраняем исходное расширение, чтобы FFmpeg выбрал верный demuxer для форматов
        # без заголовка контейнера вроде AAC. Временный файл остаётся в целевом каталоге, чтобы финальный os.replace был атомарным.
        descriptor, temp_path = tempfile.mkstemp(
            prefix=_INTERNAL_UPLOAD_PREFIX,
            suffix=Path(safe_name).suffix.lower(),
            dir=target_dir,
        )
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = source.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise BgmUploadError("background music upload must be binary")
                total_bytes += len(chunk)
                if total_bytes > MAX_BGM_UPLOAD_BYTES:
                    raise BgmUploadError("background music file exceeds the 30 MB limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise BgmUploadError("background music file is empty")
        return safe_name, temp_path, total_bytes
    except Exception as exc:
        _remove_staged_file(temp_path)
        if isinstance(exc, BgmUploadError):
            raise
        if isinstance(exc, OSError):
            raise BgmServiceError("failed to stage background music upload") from exc
        raise
    finally:
        # Тот же UploadedFile нужен Streamlit для прослушивания в браузере. Возврат
        # файлового указателя не даёт плееру или финальному сохранению прочитать пустоту после проверки.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass


def validate_bgm_upload(filename: str, source: BinaryIO) -> str:
    """Полностью проверяет загруженное аудио без сохранения — предпроверка WebUI перед показом статуса «готово»."""
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    try:
        _validate_audio(temp_path)
        logger.debug(
            f"background music upload validated: name={safe_name}, "
            f"size={total_bytes} bytes"
        )
        return safe_name
    finally:
        _remove_staged_file(temp_path)


def save_bgm_upload(filename: str, source: BinaryIO) -> str:
    """
    Сохраняет пользовательскую фоновую музыку по частям, с ограничением объёма и
    атомарной подменой.

    Сценарии использования — UploadFile из FastAPI и UploadedFile из Streamlit;
    оба дают двоичный файловый интерфейс. Сначала пишется и проверяется временный
    файл в том же каталоге, затем os.replace атомарно кладёт его на место. Так
    параллельные загрузки или прерывание процесса не оставляют половину
    аудиофайла, а загрузка файла с тем же именем получает другой ключ хранения на
    основе UUID — поэтому уже поставленные в очередь и выполняющиеся задачи
    всегда ссылаются на прежний неизменяемый файл.
    """
    safe_name, temp_path, total_bytes = _stage_bgm_upload(filename, source)
    stored_name = f"{uuid4().hex}{Path(safe_name).suffix.lower()}"
    target_path = os.path.join(os.path.dirname(temp_path), stored_name)

    try:
        _validate_audio(temp_path)
        try:
            os.replace(temp_path, target_path)
        except OSError as exc:
            raise BgmServiceError("failed to persist background music upload") from exc
        temp_path = ""
        logger.info(
            f"background music uploaded: original_name={safe_name}, "
            f"stored_name={stored_name}, size={total_bytes} bytes"
        )
        return stored_name
    finally:
        _remove_staged_file(temp_path)


def list_bgm_files() -> list[str]:
    """Перечисляет доступную фоновую музыку: загруженную пользователем и встроенную."""
    files_by_name: dict[str, str] = {}
    for directory in (utils.song_dir(), uploaded_bgm_dir(create=True)):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory), key=str.lower):
            # И предпроверка загрузки, и финальное сохранение ненадолго создают файл в том же
            # каталоге. У временного файла корректное аудиорасширение, но проверку он ещё не прошёл, и попадать в выбор случайного BGM раньше времени не должен.
            if name.startswith(_INTERNAL_UPLOAD_PREFIX):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS:
                continue
            file_path = os.path.join(directory, name)
            try:
                # Результаты перечисления тоже нуждаются в проверке реального пути: иначе
                # атакующий положит в разрешённый каталог аудиосимлинк на внешний файл и передаст его в MoviePy через путь случайного BGM.
                resolved_path = file_security.resolve_path_within_directory(
                    directory, file_path
                )
            except ValueError as exc:
                logger.warning(
                    f"skip unsafe background music file: name={name}, error={str(exc)}"
                )
                continue
            files_by_name[name] = resolved_path
    return [files_by_name[name] for name in sorted(files_by_name, key=str.lower)]


def resolve_bgm_file(unsafe_path: str) -> str:
    """
    Разрешает путь к BGM в каталоге пользовательских загрузок и в каталоге
    встроенных композиций, отклоняя всё за пределами этих двух белых списков.

    Имя файла сперва ищется в пользовательском каталоге; при этом сохраняются
    прежние варианты записи — `output000.mp3`, абсолютный путь из белого списка и
    `./resource/songs/output000.mp3`. Новые загрузки используют UUID, поэтому в
    норме не конфликтуют по имени со встроенными композициями и прошлыми
    загрузками.
    """
    if (
        not unsafe_path
        or Path(unsafe_path).suffix.lower() not in SUPPORTED_BGM_EXTENSIONS
    ):
        raise ValueError("unsupported background music path")

    candidates = [unsafe_path]
    if not os.path.isabs(unsafe_path):
        candidates.append(os.path.join(utils.root_dir(), unsafe_path))

    last_error = ValueError("background music file does not exist")
    for directory in (uploaded_bgm_dir(create=True), utils.song_dir()):
        for candidate in candidates:
            try:
                return file_security.resolve_path_within_directory(directory, candidate)
            except ValueError as exc:
                last_error = exc
    raise ValueError(str(last_error)) from last_error
