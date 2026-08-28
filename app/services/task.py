import math
import os
import re
import socket
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from os import path
from uuid import uuid4

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import bgm as bgm_service
from app.services import (
    elevenlabs_music,
    llm,
    loomloom,
    material,
    sonilo,
    subtitle,
    task_artifacts,
    twelvelabs,
    video,
    volcengine_seedance,
    voice,
)
from app.services import upload_post
from app.services import state as sm
from app.utils import file_security, utils


# Запрос публикации может ждать несколько минут, и занимать слот параллелизма
# генерации видео он не вправе. Пул потоков фиксированного размера удерживает
# пропускную способность публикации в разумных пределах и позволяет задаче видео
# переходить в завершённое состояние сразу после получения результата.
_cross_post_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="mpt-cross-post",
)
_cross_post_max_pending_tasks = max(
    1,
    int(config.app.get("upload_post_max_pending_tasks", 10)),
)
_cross_post_slots = threading.BoundedSemaphore(_cross_post_max_pending_tasks)
_cross_post_registry_lock = threading.RLock()
_cross_post_futures: dict[str, Future] = {}
_cross_post_process_owner = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
_ACTIVE_CROSS_POST_STATES = {
    const.CROSS_POST_STATE_PENDING,
    const.CROSS_POST_STATE_PROCESSING,
}
_CROSS_POST_STATE_WRITE_ATTEMPTS = 3
_CROSS_POST_STATE_RETRY_DELAY_SECONDS = 0.1
_LOOMLOOM_STATE_WRITE_ATTEMPTS = 3
_LOOMLOOM_STATE_RETRY_DELAY_SECONDS = 0.1
_INTERRUPTED_CROSS_POST_ERROR = (
    "cross-posting was interrupted before the process completed"
)
# Map upload-post platform ids to the social platform names llm.py accepts.
_CROSS_POST_SOCIAL_PLATFORMS = {
    "tiktok": "tiktok",
    "instagram": "instagram_reels",
    "facebook": "facebook_reels",
}
# Сервису музыки достаточно реализовать ``is_enabled`` и ``generate_bgm``. Различия
# поставщиков сосредоточены в расширении файла, доменных исключениях и коде
# предупреждения для WebUI; оркестрация задачи, короткое замыкание при нулевой
# громкости и мягкая деградация при сбое идут по одному и тому же пути — не придётся
# вести несколько похожих процедур при добавлении новых поставщиков.
_VIDEO_MUSIC_PROVIDERS = {
    "sonilo": {
        "service": sonilo,
        "error_type": sonilo.SoniloError,
        "suffix": ".m4a",
        "warning_code": "sonilo_bgm_failed",
        "display_name": "Sonilo",
    },
    "elevenlabs": {
        "service": elevenlabs_music,
        "error_type": elevenlabs_music.ElevenLabsMusicError,
        "suffix": ".mp3",
        "warning_code": "elevenlabs_bgm_failed",
        "display_name": "ElevenLabs",
    },
}


def _get_video_music_prompt(params: VideoParams) -> str:
    """
    Читает промпт, который реально использует текущий поставщик музыки.

    Новые задачи используют поле, не зависящее от поставщика; у старых параметров
    Sonilo в CLI и у прошлых задач может быть только ``sonilo_bgm_prompt``,
    поэтому старое поле читается лишь тогда, когда общее поле Sonilo пусто.
    """
    prompt = str(params.video_music_prompt or "").strip()
    if params.bgm_type == "sonilo" and not prompt:
        prompt = str(params.sonilo_bgm_prompt or "").strip()
    return prompt


def is_task_busy(task: dict | None) -> bool:
    """Определяет, идёт ли ещё генерация или публикация задачи; переиспользуется всеми точками удаления."""
    if not task:
        return False

    state = task.get("state")
    try:
        state = int(state)
    except (TypeError, ValueError):
        pass

    # И генерация видео, и кросспостинг могут продолжать читать каталог задачи.
    # Считаем оба состояния занятостью: иначе API и WebUI, ведя правила по отдельности,
    # разойдутся — один разрешит удаление, другой запретит.
    return (
        state == const.TASK_STATE_PROCESSING
        or task.get("cross_post_state") in _ACTIVE_CROSS_POST_STATES
    )


def _register_cross_post_future(task_id: str, future: Future) -> None:
    """Регистрирует Future публикации, принадлежащий текущему процессу, чтобы восстановление при старте и тесты видели реальное состояние выполнения."""
    with _cross_post_registry_lock:
        _cross_post_futures[task_id] = future


def _unregister_cross_post_future(task_id: str, future: Future | None = None) -> None:
    """Удаляет только совпадающий Future, чтобы старый колбэк не снял новую работу, зарегистрированную позже для той же задачи."""
    with _cross_post_registry_lock:
        current = _cross_post_futures.get(task_id)
        if current is None or (future is not None and current is not future):
            return
        _cross_post_futures.pop(task_id, None)


def _is_cross_post_active_in_process(task_id: str) -> bool:
    """Определяет, остались ли у текущего процесса незавершённые задачи публикации."""
    with _cross_post_registry_lock:
        future = _cross_post_futures.get(task_id)
        return future is not None and not future.done()


def _is_windows_process_alive(process_id: int) -> bool:
    """Определяет состояние процесса через Win32 API только на чтение, чтобы случайно не убить процесс через os.kill."""
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    error_invalid_parameter = 87
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # ctypes по умолчанию считает необъявленное возвращаемое значение 32-битным int.
    # Дескриптор 64-битного процесса Windows из-за этого может быть обрезан, поэтому сигнатуры функций Win32 нужно объявить явно перед вызовом.
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        error_code = ctypes.get_last_error()
        if error_code == error_invalid_parameter:
            return False
        if error_code == error_access_denied:
            # Если процесс существует, но у текущего пользователя нет прав на запрос,
            # консервативно считаем его живым, чтобы ошибочно не подобрать задачу публикации, выполняемую под другой учётной записью.
            return True
        logger.warning(
            "failed to open cross-post owner process on Windows, "
            f"process_id: {process_id}, error_code: {error_code}"
        )
        return True

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error_code = ctypes.get_last_error()
            logger.warning(
                "failed to read cross-post owner process state on Windows, "
                f"process_id: {process_id}, error_code: {error_code}"
            )
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _is_cross_post_owner_alive(owner: str | None) -> bool:
    """Определяет, существует ли ещё локальный процесс сохранённой задачи публикации."""
    if not owner:
        return False

    try:
        hostname, process_id_text, _ = owner.split(":", 2)
        process_id = int(process_id_text)
    except (TypeError, ValueError):
        logger.warning(f"invalid cross-post owner metadata: {owner}")
        return False

    # Надёжно определить процесс на другом хосте невозможно. При развёртывании на
    # нескольких хостах с общим Redis консервативно считаем задачу выполняющейся, чтобы текущий узел не удалил видеофайл, который читает другой узел.
    if hostname != socket.gethostname():
        return True

    # Есть ли в текущем процессе реальная работа по публикации, точно определяет
    # реестр Future. Если управление дошло сюда, соответствующего Future в реестре нет,
    # и даже при полном совпадении owner с текущим процессом задачу считаем прерванной. Это покрывает случай, когда запись финального статуса стабильно падает, а Future уже завершился.
    if process_id == os.getpid():
        return False

    # В Windows os.kill(pid, 0) ведёт себя иначе, чем в POSIX, и может прямо завершить
    # целевой процесс. Используем Win32 API, запрашивающий только право на опрос, и не шлём процессу никаких сигналов.
    if os.name == "nt":
        return _is_windows_process_alive(process_id)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        logger.warning(
            f"failed to inspect cross-post owner process, owner: {owner}, error: {exc}"
        )
        return True
    return True


def _mark_task_failed(
    task_id: str,
    stage: str,
    error: str,
    details: dict | None = None,
) -> dict:
    """Записывает структурированные сведения об ошибке, сохраняя прогресс, достигнутый задачей до сбоя."""
    existing_task = None
    try:
        existing_task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.warning(f"failed to read task state before failure update: {exc}")

    # Конкретная сервисная функция обычно знает причину ошибки точнее, чем слой
    # оркестрации. Последующая проверка пустого результата не вправе перекрыть её общим текстом — иначе вызывающая сторона API снова увидит лишь размытую формулировку.
    if (
        existing_task
        and existing_task.get("state") == const.TASK_STATE_FAILED
        and existing_task.get("error")
    ):
        return existing_task

    message = str(error or "unknown task error").strip()
    progress = int((existing_task or {}).get("progress", 0) or 0)
    logger.error(f"task failed, task_id: {task_id}, stage: {stage}, error: {message}")
    failure = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "progress": progress,
        "failed_stage": stage,
        "error": message,
    }
    # Некоторые внешние задачи уже создали удалённый ID, пригодный для восстановления
    # или разбора. Статус ошибки обязан сохранить эти нечувствительные поля, но не позволять вызывающей стороне перекрывать единую структуру статуса, прогресса и ошибки.
    failure_details = {
        key: value for key, value in dict(details or {}).items() if key not in failure
    }
    failure.update(failure_details)
    sm.state.update_task(
        task_id,
        state=failure["state"],
        progress=failure["progress"],
        failed_stage=failure["failed_stage"],
        error=failure["error"],
        **failure_details,
    )
    return failure


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        _mark_task_failed(task_id, "script", "failed to generate video script")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        # Когда включён подбор материалов по порядку текста, сами ключевые слова тоже
        # обязаны генерироваться в повествовательном порядке сценария. Иначе даже при
        # упорядоченных загрузке и склейке останется один набор общих тематических слов, и проблема «кадры из конца появляются раньше времени» не решится.
        video_terms = llm.generate_terms(
            video_subject=params.video_subject,
            video_script=video_script,
            amount=8 if params.match_materials_to_script else 5,
            match_script_order=params.match_materials_to_script,
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        _mark_task_failed(
            task_id,
            "terms",
            "failed to generate video search terms",
        )
        return None

    # Опциональное семантическое переупорядочивание TwelveLabs Marengo: если выключено,
    # порядок возвращается прежним, без побочных эффектов. В режиме упорядоченного подбора порядок ключевых слов и есть порядок повествования, поэтому шаг пропускается.
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }
    task_artifacts.write_script_data(task_id, script_data)


def resolve_custom_audio_file(
    task_id: str,
    custom_audio_file: str | None,
    *,
    allow_server_file_input: bool = False,
) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    # A missing path that otherwise stays inside the task directory is safe to
    # report precisely. Paths outside that boundary use the same generic error
    # regardless of whether they exist, so callers cannot probe the host filesystem.
    if str(task_dir_error) == "file does not exist":
        raise task_dir_error

    # HTTP requests and other untrusted callers must never turn a submitted path
    # into a server-side file read. WebUI uploads already live in the task directory;
    # only the local CLI explicitly opts into resolving files elsewhere on the host.
    if not allow_server_file_input:
        raise ValueError(
            "custom audio file must be stored within the current task directory"
        ) from task_dir_error

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def _resolve_reusable_voice_preview(
    task_id: str,
    params,
    video_script: str,
    voice_preview: dict | None,
) -> tuple[str, float, object] | None:
    """
    Проверяет и разбирает кэш полного прослушивания, отправленный из WebUI.

    Эта нагрузка не является параметром публичного API и может прийти только от
    WebUI текущего процесса. Тем не менее фоновая задача заново сверяет текст и
    все параметры озвучки и требует, чтобы аудио лежало в каталоге текущей задачи;
    при любом несовпадении происходит откат к обычному TTS, чтобы устаревшее
    прослушивание не испортило финальный ролик.
    """
    if not voice_preview:
        return None

    expected_values = {
        "script": str(video_script or "").strip(),
        "voice_name": params.voice_name,
        "voice_rate": float(params.voice_rate),
        "voice_volume": float(params.voice_volume),
    }
    if not math.isclose(float(params.voice_volume), 1.0) or any(
        voice_preview.get(key) != value for key, value in expected_values.items()
    ):
        logger.info(
            f"skip stale voice preview cache, task_id: {task_id}, "
            "reason: voice parameters changed"
        )
        return None

    preview_file = path.realpath(str(voice_preview.get("audio_file") or ""))
    task_root = path.realpath(utils.task_dir(task_id))
    try:
        preview_is_task_local = path.commonpath([task_root, preview_file]) == task_root
    except ValueError:
        preview_is_task_local = False

    duration = voice_preview.get("duration")
    sub_maker = voice_preview.get("sub_maker")
    if (
        not preview_is_task_local
        or not path.isfile(preview_file)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or sub_maker is None
    ):
        logger.warning(
            f"skip invalid voice preview cache, task_id: {task_id}, "
            f"audio_file: {preview_file or '<empty>'}"
        )
        return None

    logger.info(
        f"using full voice preview audio, task_id: {task_id}, duration: {duration:.2f}s"
    )
    return preview_file, math.ceil(duration), sub_maker


def generate_audio(
    task_id,
    params,
    video_script,
    voice_preview=None,
    *,
    allow_server_file_input: bool = False,
):
    """
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    """
    logger.info("\n\n## generating audio")
    # Модели запросов /audio и /subtitle не содержат custom_audio_file,
    # поэтому читаем поле совместимым способом, чтобы прямой вызов эндпоинта не давал ошибку атрибута.
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id,
            requested_custom_audio_file,
            allow_server_file_input=allow_server_file_input,
        )
    except ValueError as exc:
        _mark_task_failed(
            task_id,
            "audio",
            f"invalid custom audio file: {exc}",
        )
        return None, None, None

    if not custom_audio_file:
        reusable_preview = _resolve_reusable_voice_preview(
            task_id,
            params,
            video_script,
            voice_preview,
        )
        if reusable_preview:
            return reusable_preview

        logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        sub_maker = voice.tts(
            text=video_script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            _mark_task_failed(
                task_id,
                "audio",
                "failed to synthesize audio; verify the selected voice and TTS connectivity",
            )
            return None, None, None
        # Measure the real written audio_file, not sub_maker.cues[-1].end:
        # the latter is the last WORD BOUNDARY, and TTS leaves a fixed tail
        # past it (Edge TTS: ~0.88s at any length - 19% of a 7-word clip but
        # 1.4% of a 153-word one, so short scripts suffer most). The
        # under-count sizes paid generate_bgm() calls, is reported as
        # audio_duration to the API/WebUI, and under-sources
        # download_videos() material, scaled by video_count.
        file_duration = voice.get_audio_duration(audio_file)
        audio_duration = math.ceil(
            file_duration if file_duration > 0 else voice.get_audio_duration(sub_maker)
        )
        if audio_duration == 0:
            _mark_task_failed(task_id, "audio", "generated audio duration is zero")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            _mark_task_failed(
                task_id,
                "audio",
                "custom audio duration is zero",
            )
            return None, None, None
        return custom_audio_file, audio_duration, None


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    """
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    """
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    if not subtitle_provider:
        logger.info("subtitle provider is empty, skip subtitle generation")
        return ""

    if sub_maker is None and subtitle_provider != "whisper":
        # Пользовательское аудио не проходит через TTS, поэтому у него нет таймлайна
        # sub_maker, который возвращают Edge, Azure и другие TTS. Расшифровать субтитры
        # прямо из аудиофайла умеет только Whisper; остальные поставщики субтитров сохраняют прежнее поведение, чтобы не породить неверный пустой таймлайн.
        logger.warning(
            "subtitle maker is missing, skip subtitle generation for provider: "
            f"{subtitle_provider}"
        )
        return ""

    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            # Субтитры Edge иногда не дают файла, потому что таймлайн не удаётся сопоставить
            # с текстом. Автоматически переключаться на Whisper здесь нельзя: первая же
            # неудача незаметно для пользователя скачала бы модель на несколько ГБ. Модель
            # загружается только при явно настроенном Whisper; при сбое Edge остаётся видео
            # без субтитров и запись о причине — так не возникнет неожиданных трат сети и диска.
            logger.warning(
                "edge subtitle generation did not produce a subtitle file; "
                "skip subtitles without falling back to whisper"
            )
            return ""

    if subtitle_provider == "whisper":
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            _mark_task_failed(
                task_id,
                "materials",
                "no valid local video materials were found",
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "loomloom":
        if not isinstance(
            loomloom_video_request, loomloom.LoomLoomConfirmedVideoRequest
        ):
            _mark_task_failed(
                task_id,
                "materials",
                "LoomLoom video generation requires a confirmed quote",
            )
            return None

        request = loomloom_video_request
        logger.info(
            "\n\n## generating "
            f"{len(request.batch.input_rows)} video materials with LoomLoom"
        )
        run_id = ""
        try:
            request.validate()
            backend = loomloom.LoomLoomVideoBackend(request.settings)
            execution = backend.execute(
                request.batch,
                client_request_id=request.client_request_id,
                listing_version_id=request.listing_version_id,
                confirm=True,
            )
            run_id = execution.run_id
            # Возврат из execute означает, что платная задача уже принята удалённой стороной.
            # ID запуска необходимо сперва записать в лог процесса: даже если бэкенд статусов
            # вроде Redis затем станет недоступен, по логам можно будет найти задачу на стороне Shengsuanyun — единственный идентификатор не должен жить только в локальной переменной.
            logger.info(
                "LoomLoom paid video run created: "
                f"task_id={task_id}, run_id={run_id}, "
                f"listing_version_id={request.listing_version_id}"
            )
            # Как только платная задача создана, сразу фиксируем удалённый ID. Даже если
            # последующий опрос упрётся в таймаут, логи и статус задачи помогут пользователю
            # или поддержке платформы найти и забрать уже сгенерированный результат. Сбой бэкенда статусов может лишь ухудшить наблюдаемость, но не должен обрывать уже оплачиваемую удалённую задачу и загрузку результата.
            _record_loomloom_run_reference(
                task_id=task_id,
                run_id=run_id,
                listing_version_id=request.listing_version_id,
            )
            backend.wait_for_run(run_id)
            return list(
                backend.download_video_results(
                    run_id,
                    utils.task_dir(task_id),
                )
            )
        except (loomloom.LoomLoomError, ValueError) as exc:
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details={
                    "loomloom_run_id": run_id,
                    "loomloom_listing_version_id": request.listing_version_id,
                },
            )
            return None
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        # Режим упорядоченного подбора включается только явно пользователем. Здесь
        # загрузка материалов принудительно идёт по кругу в порядке ключевых слов, чтобы одно раннее слово не набрало столько материала, что поздние темы сценария вытеснит с итогового таймлайна.
        try:
            downloaded_videos = material.download_videos(
                task_id=task_id,
                search_terms=video_terms,
                source=params.video_source,
                video_aspect=params.video_aspect,
                video_concat_mode=(
                    VideoConcatMode.sequential
                    if params.match_materials_to_script
                    else params.video_concat_mode
                ),
                audio_duration=audio_duration * params.video_count,
                max_clip_duration=params.video_clip_duration,
                match_script_order=params.match_materials_to_script,
            )
        except volcengine_seedance.VolcEngineSeedanceError as exc:
            # И неподтверждённый статус, и «сгенерировано, но не скачалось» соответствуют
            # удалённой задаче, которую можно восстановить в консоли Ark. Единообразно пишем
            # статус ошибки из task_id, приложенного к исключению, чтобы разные ветки исключений не вели сведения о восстановлении по-своему и снова не растеряли их при доработках.
            remote_task_id = str(getattr(exc, "task_id", "") or "").strip()
            details = (
                {"volcengine_seedance_task_id": remote_task_id}
                if remote_task_id
                else None
            )
            _mark_task_failed(
                task_id,
                "materials",
                str(exc),
                details=details,
            )
            return None
        if not downloaded_videos:
            _mark_task_failed(
                task_id,
                "materials",
                f"failed to download video materials from {params.video_source}",
            )
            return None
        return downloaded_videos


def _record_loomloom_run_reference(
    *, task_id: str, run_id: str, listing_version_id: str
) -> bool | None:
    """
    Изо всех сил старается сохранить уже созданный платный LoomLoom Run, не давая
    сбою хранилища статусов оборвать удалённую задачу.

    True — сохранение удалось, False — записи задачи больше нет, None — бэкенд
    статусов недоступен даже после ограниченного числа повторов. При любом исходе
    вызывающая сторона обязана продолжить опрос и загрузку: execute уже вызвал
    внешний платный побочный эффект, и остановка локального процесса лишь
    затруднит получение результата.
    """
    fields = {
        "loomloom_run_id": run_id,
        "loomloom_listing_version_id": listing_version_id,
    }
    for attempt in range(1, _LOOMLOOM_STATE_WRITE_ATTEMPTS + 1):
        try:
            updated = sm.state.patch_task(task_id, **fields)
        except Exception as exc:
            if attempt >= _LOOMLOOM_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    "failed to persist LoomLoom paid run after retries: "
                    f"task_id={task_id}, run_id={run_id}, attempts={attempt}, "
                    f"error={exc}"
                )
                return None
            logger.warning(
                "retry LoomLoom paid run state update: "
                f"task_id={task_id}, run_id={run_id}, attempt={attempt}, "
                f"error={exc}"
            )
            time.sleep(_LOOMLOOM_STATE_RETRY_DELAY_SECONDS)
            continue

        if updated is False:
            logger.warning(
                "could not persist LoomLoom paid run because task is missing: "
                f"task_id={task_id}, run_id={run_id}"
            )
        return updated

    return None


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, audio_duration
):
    final_video_paths = []
    combined_video_paths = []
    warnings = []
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_requested = (
        video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    # При генерации нескольких видео материалы по умолчанию перемешиваются ради
    # разнообразия, но «подбор материалов по порядку текста» нацелен на стабильный и объяснимый таймлайн, поэтому при включении все выводы используют последовательную склейку.
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            clip_speed=params.video_clip_speed,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        # В режиме музыки для видео сначала явно отключаем разрешение BGM по умолчанию,
        # чтобы не подхватить bgm_file, оставшийся от старой задачи. Прокси создаётся и платный API вызывается только при громкости больше 0; при нулевой шаг пропускается.
        bgm_file_override = "" if video_music_provider else None
        if video_music_requested:
            service = video_music_provider["service"]
            display_name = video_music_provider["display_name"]
            warning_code = video_music_provider["warning_code"]
            generated_bgm_path = path.join(
                utils.task_dir(task_id),
                (f"{params.bgm_type}-bgm-{index}{video_music_provider['suffix']}"),
            )
            try:
                service.generate_bgm(
                    video_path=combined_video_path,
                    output_path=generated_bgm_path,
                    video_duration=audio_duration,
                    prompt=_get_video_music_prompt(params),
                )
                bgm_file_override = generated_bgm_path
            except video_music_provider["error_type"] as exc:
                # Когда видео, озвучка и субтитры уже готовы, временный сбой стороннего
                # сервиса музыки не должен обесценить всю задачу. Для текущего видео BGM явно отключается, а результат деградации возвращается в WebUI как предупреждение пользователю.
                logger.warning(
                    f"{display_name} BGM generation failed: task_id={task_id}, "
                    f"video_index={index}, error={exc}"
                )
                bgm_file_override = ""
                warnings.append({"code": warning_code, "video_index": index})

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        bgm_mix_succeeded = video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
            bgm_file_override=bgm_file_override,
        )
        if (
            video_music_provider is not None
            and bgm_file_override
            and not bgm_mix_succeeded
        ):
            # Сторонний сервис уже успешно ответил и прошёл проверку FFmpeg, но финальное
            # сведение в MoviePy всё ещё может упасть из-за окружения. Сервис видео сохранит
            # ролик без BGM; при неудачной генерации через API override пуст, поэтому предупреждение не задвоится.
            warnings.append(
                {
                    "code": video_music_provider["warning_code"],
                    "video_index": index,
                }
            )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths, warnings


def _patch_cross_post_state(task_id: str, **kwargs) -> bool | None:
    """Безопасно обновляет поля публикации с ограниченным числом повторов при кратковременном сбое бэкенда статусов."""
    for attempt in range(1, _CROSS_POST_STATE_WRITE_ATTEMPTS + 1):
        try:
            return sm.state.patch_task(task_id, **kwargs)
        except Exception as exc:
            # Кратковременный обрыв Redis не должен навсегда оставлять задачу в pending или
            # processing. Статус публикации пишется редко, поэтому фиксированного числа
            # попыток с короткой паузой хватает на мгновенный сбой и фоновый поток не блокируется бесконечно. У последней неудачи сохраняется полный стек для разбора.
            if attempt >= _CROSS_POST_STATE_WRITE_ATTEMPTS:
                logger.exception(
                    f"failed to update cross-post state after retries, "
                    f"task_id: {task_id}, fields: {', '.join(kwargs)}, "
                    f"attempts: {attempt}, error: {exc}"
                )
                return None

            logger.warning(
                f"retry cross-post state update, task_id: {task_id}, "
                f"fields: {', '.join(kwargs)}, attempt: {attempt}, error: {exc}"
            )
            time.sleep(_CROSS_POST_STATE_RETRY_DELAY_SECONDS)

    return None


def _record_cross_post_failure(
    task_id: str,
    error: Exception,
    results: list[dict] | None = None,
) -> None:
    """Изо всех сил старается сохранить факт неудачной публикации; если бэкенд статусов недоступен, диагностика остаётся в логах."""
    updated = _patch_cross_post_state(
        task_id,
        cross_post_state=const.CROSS_POST_STATE_FAILED,
        cross_post_results=results or None,
        cross_post_error=str(error),
        cross_post_owner=None,
    )
    if updated is False:
        logger.warning(f"discard cross-post failure for missing task: {task_id}")


def _ensure_cross_post_terminal_state(task_id: str) -> None:
    """После завершения Future сводит задачи, оставшиеся в активном состоянии, к статусу ошибки."""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        # Это уже финальный колбэк Future, и синхронной вызывающей стороны, способной
        # обработать исключение, дальше нет. После восстановления бэкенда статусов зависшие записи всё равно разберёт логика восстановления при следующем старте процесса.
        logger.exception(
            f"failed to verify final cross-post state, task_id: {task_id}, error: {exc}"
        )
        return

    if not task or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES:
        return

    logger.warning(
        f"cross-post worker ended without terminal state, task_id: {task_id}, "
        f"state: {task.get('cross_post_state')}"
    )
    _record_cross_post_failure(
        task_id,
        RuntimeError("cross-post worker ended without persisting a terminal state"),
        task.get("cross_post_results"),
    )


def recover_interrupted_cross_posts(page_size: int = 100) -> int | None:
    """
    Помечает ошибкой задачи публикации, которые невозможно возобновить после
    перезапуска процесса.

    Кросспостинг использует пул потоков внутри текущего процесса, а не постоянную
    очередь задач. При старте процесса оставшиеся в Redis pending и processing сами
    собой не продолжатся; если и дальше считать их выполняющимися, пользователь
    никогда не сможет удалить задачу. Здесь статусы сканируются постранично,
    обрабатываются только активные записи без соответствующего Future в текущем
    процессе, а уже сгенерированные результаты видео сохраняются.
    """
    recovered = 0
    page = 1

    while True:
        try:
            tasks, total = sm.state.get_all_tasks(page, page_size)
        except Exception as exc:
            logger.exception(f"failed to recover interrupted cross-post tasks: {exc}")
            return None

        for task in tasks:
            task_id = str(task.get("task_id") or "")
            if (
                not task_id
                or task.get("cross_post_state") not in _ACTIVE_CROSS_POST_STATES
                or _is_cross_post_active_in_process(task_id)
                or _is_cross_post_owner_alive(task.get("cross_post_owner"))
            ):
                continue

            updated = _patch_cross_post_state(
                task_id,
                cross_post_state=const.CROSS_POST_STATE_FAILED,
                cross_post_error=_INTERRUPTED_CROSS_POST_ERROR,
                cross_post_owner=None,
            )
            if updated is True:
                recovered += 1

        if page * page_size >= total or not tasks:
            break
        page += 1

    if recovered:
        logger.warning(f"recovered interrupted cross-post tasks: {recovered}")
    return recovered


def _run_cross_post(
    task_id: str,
    video_paths: tuple[str, ...],
    video_subject: str,
    video_script: str,
    video_language: str,
    platforms: tuple[str, ...],
    youtube_privacy_status: str,
) -> None:
    """Выполняет кросспостинг в фоне, дополняя только поля задачи, относящиеся к публикации."""
    results = []
    try:
        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_PROCESSING,
            cross_post_error=None,
            cross_post_owner=_cross_post_process_owner,
        )
        if state_updated is not True:
            # False означает, что задача удалена, None — что бэкенд статусов временно
            # недоступен. В обоих случаях обращаться к стороннему эндпоинту нельзя: иначе пользователь не сможет ни отследить эту публикацию, ни управлять ею.
            if state_updated is False:
                logger.warning(f"skip cross-post for missing task: {task_id}")
            else:
                _record_cross_post_failure(
                    task_id,
                    RuntimeError("failed to persist cross-post processing state"),
                )
            return

        logger.info(
            f"cross-post started, task_id: {task_id}, platforms: {', '.join(platforms)}"
        )
        youtube_extra = None
        post_title = video_subject or "Check out this video! #shorts #viral"
        if platforms:
            has_youtube = any(platform.startswith("youtube") for platform in platforms)
            social_platform = "youtube_shorts"
            if not has_youtube:
                first = (platforms[0] or "").strip().lower()
                # llm.py resolves unknown ids to its default platform.
                social_platform = _CROSS_POST_SOCIAL_PLATFORMS.get(first, first)
            metadata = llm.generate_social_metadata(
                video_subject=video_subject,
                video_script=video_script,
                language=video_language or "",
                platform=social_platform,
            )
            if has_youtube:
                youtube_extra = {
                    "youtube_title": metadata.get("title", video_subject),
                    "youtube_description": metadata.get("caption", ""),
                    "tags": metadata.get("hashtags", []),
                    "privacyStatus": youtube_privacy_status,
                    "containsSyntheticMedia": True,
                }
            post_title = (
                metadata.get("caption")
                or metadata.get("title")
                or video_subject
                or "Check out this video! #shorts #viral"
            )

        for video_path in video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=post_title,
                platforms=list(platforms),
                youtube_extra=youtube_extra,
            )
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": "Upload-Post returned an invalid response",
                }
            results.append(result)

        failures = [result for result in results if not result.get("success")]
        if failures:
            error_messages = [
                str(
                    result.get("error")
                    or result.get("message")
                    or "unknown upload error"
                )
                for result in failures
            ]
            cross_post_state = const.CROSS_POST_STATE_FAILED
            cross_post_error = "; ".join(error_messages)
            logger.warning(
                f"cross-post completed with failures, task_id: {task_id}, "
                f"failed: {len(failures)}, total: {len(results)}"
            )
        else:
            cross_post_state = const.CROSS_POST_STATE_COMPLETE
            cross_post_error = None
            logger.success(
                f"cross-post completed, task_id: {task_id}, videos: {len(results)}"
            )

        state_updated = _patch_cross_post_state(
            task_id,
            cross_post_state=cross_post_state,
            cross_post_results=results,
            cross_post_error=cross_post_error,
            cross_post_owner=None,
        )
        if state_updated is False:
            logger.warning(f"discard cross-post result for missing task: {task_id}")
        elif state_updated is None:
            # Если загрузка завершилась, а результат не сохранился, оставлять processing
            # нельзя. Запись статуса ошибки снова пройдёт ограниченные повторы — вызывающая сторона хотя бы получит однозначный финальный статус.
            _record_cross_post_failure(
                task_id,
                RuntimeError("failed to persist final cross-post result"),
                results,
            )
    except Exception as exc:
        # Неудачная публикация влияет только на статус публикации и не вправе перекрыть
        # уже завершённую задачу видео. Текст исключения пишется в статус задачи, чтобы вызывающей стороне API не пришлось лезть в серверные логи.
        logger.exception(f"cross-post failed, task_id: {task_id}, error: {exc}")
        _record_cross_post_failure(task_id, exc, results)


def _run_cross_post_with_slot(*args) -> None:
    """Выполняет задачу публикации, гарантированно возвращая ёмкость очереди при успехе, неудаче и исключении."""
    try:
        _run_cross_post(*args)
    except Exception as exc:
        # _run_cross_post уже обрабатывает ожидаемые исключения; здесь последний рубеж
        # защиты, чтобы исключение из будущей логики не осело в Future, который никто не читает.
        task_id = str(args[0]) if args else "unknown"
        logger.exception(f"cross-post worker crashed, task_id: {task_id}, error: {exc}")
        if args:
            _record_cross_post_failure(task_id, exc)
    finally:
        _cross_post_slots.release()


def _finalize_cross_post_future(task_id: str, future: Future) -> None:
    """Снимает регистрацию Future и доводит до финального состояния отмену, исключение и неудачную запись статуса."""
    _unregister_cross_post_future(task_id, future)

    try:
        error = future.exception()
    except CancelledError:
        logger.warning(f"cross-post future was cancelled, task_id: {task_id}")
        # Если Future отменён до начала выполнения, finally воркера не отработает,
        # поэтому ёмкость очереди возвращается в колбэке, а сохранённый статус переводится в ошибку.
        _cross_post_slots.release()
        _record_cross_post_failure(
            task_id,
            RuntimeError("cross-post job was cancelled before execution"),
        )
        return
    except Exception as exc:
        logger.exception(
            f"failed to inspect cross-post future, task_id: {task_id}, error: {exc}"
        )
        _ensure_cross_post_terminal_state(task_id)
        return

    if error is not None:
        logger.error(
            f"cross-post future failed, task_id: {task_id}, "
            f"error: {type(error).__name__}: {error}"
        )

    _ensure_cross_post_terminal_state(task_id)


def _schedule_cross_post(
    task_id: str,
    video_paths: list[str],
    params: VideoParams,
    video_script: str,
    platforms: list[str],
    youtube_privacy_status: str,
) -> str | None:
    """Отправляет фоновую задачу публикации: при успехе возвращает None, при сбое планирования — наблюдаемую причину ошибки."""
    if not _cross_post_slots.acquire(blocking=False):
        error = "cross-post queue is full; publishing was skipped"
        logger.warning(
            f"skip cross-post because queue is full, task_id: {task_id}, "
            f"capacity: {_cross_post_max_pending_tasks}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=error,
            cross_post_owner=None,
        )
        return error

    try:
        future = _cross_post_executor.submit(
            _run_cross_post_with_slot,
            task_id,
            tuple(video_paths),
            params.video_subject or "",
            video_script,
            params.video_language or "",
            tuple(platforms),
            youtube_privacy_status,
        )
        _register_cross_post_future(task_id, future)
        future.add_done_callback(partial(_finalize_cross_post_future, task_id))
    except RuntimeError as exc:
        _unregister_cross_post_future(task_id)
        _cross_post_slots.release()
        logger.exception(
            f"failed to schedule cross-post, task_id: {task_id}, error: {exc}"
        )
        _patch_cross_post_state(
            task_id,
            cross_post_state=const.CROSS_POST_STATE_FAILED,
            cross_post_error=f"failed to schedule cross-post: {exc}",
            cross_post_owner=None,
        )
        return f"failed to schedule cross-post: {exc}"

    return None


def _run_pipeline(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    if (
        stop_at in {"materials", "video"}
        and params.video_source == "volcengine_seedance"
        and not volcengine_seedance.is_enabled()
    ):
        return _mark_task_failed(
            task_id,
            "preflight",
            "Volcano Engine Seedance requires an Ark API key",
        )

    # Поставщик музыки нужен только полному конвейеру сборки ролика. Останавливаем
    # полную задачу без ключа как можно раньше, чтобы не расходовать сперва квоты LLM, TTS и сервисов материалов; эндпоинты промежуточных результатов по-прежнему доступны отдельно.
    video_music_provider = _VIDEO_MUSIC_PROVIDERS.get(params.bgm_type)
    video_music_enabled = (
        stop_at == "video"
        and video_music_provider is not None
        and bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    )
    if video_music_enabled:
        service = video_music_provider["service"]
        display_name = video_music_provider["display_name"]
        if not service.is_enabled():
            return _mark_task_failed(
                task_id,
                "preflight",
                f"{display_name} background music requires an API key",
            )

        # WebUI ограничивает длину ввода, но API, CLI и старые задачи могут обойти
        # элементы фронтенда. Перед генерацией сценария, озвучки и материалов ещё раз
        # проверяем длину по лимиту поставщика, чтобы отказ от стороннего сервиса не пришёл уже после полной сборки видео. Ту же проверку сохраняет и сервисный слой — как последний рубеж при прямом вызове.
        music_prompt = _get_video_music_prompt(params)
        max_prompt_length = int(getattr(service, "MAX_PROMPT_LENGTH", 0) or 0)
        if max_prompt_length and len(music_prompt) > max_prompt_length:
            return _mark_task_failed(
                task_id,
                "preflight",
                (f"{display_name} music prompt exceeds {max_prompt_length} characters"),
            )

        # Поставщик может предложить бесплатную предварительную проверку аккаунта.
        # Функция проверки обязана бросать только детерминированные ошибки; при сетевых колебаниях или неопределимой области прав сервисный слой пишет предупреждение и продолжает реальную генерацию.
        validate_access = getattr(service, "validate_generation_access", None)
        if callable(validate_access):
            try:
                validate_access()
            except video_music_provider["error_type"] as exc:
                return _mark_task_failed(task_id, "preflight", str(exc))

    # FFmpeg не нужен только промежуточным результатам script и terms (они не создают
    # ни аудио, ни видео). API, CLI и WebUI выполняют задачи через эту общую точку
    # входа, поэтому проверяем здесь единожды, а не повторяем в каждой из них — так
    # поведение всех трёх путей совпадает. Расположено после проверки ключа музыки,
    # чтобы не изменить исходный порядок «падать раньше всех» и тексты тех ошибок.
    if stop_at not in ("script", "terms") and not utils.check_ffmpeg_ready():
        return _mark_task_failed(
            task_id,
            "preflight",
            "ffmpeg is not available; install ffmpeg or set app.ffmpeg_path "
            "in config.toml to a working ffmpeg executable",
        )

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        error = (
            video_script.removeprefix("Error: ").strip()
            if isinstance(video_script, str) and "Error: " in video_script
            else "failed to generate video script"
        )
        return _mark_task_failed(task_id, "script", error)

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            return _mark_task_failed(
                task_id,
                "terms",
                "failed to generate video search terms",
            )

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id,
        params,
        video_script,
        voice_preview=voice_preview,
        allow_server_file_input=allow_server_file_input,
    )
    if not audio_file:
        return _mark_task_failed(
            task_id,
            "audio",
            "failed to prepare narration audio",
        )

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id,
        params,
        video_terms,
        audio_duration,
        loomloom_video_request=loomloom_video_request,
    )
    if not downloaded_videos:
        return _mark_task_failed(
            task_id,
            "materials",
            "failed to prepare video materials",
        )

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # Режим склейки видео нужно обрабатывать только в полном конвейере генерации:
    # так запросы вроде /subtitle и /audio не обратятся к несуществующим полям.
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths, generation_warnings = (
        generate_final_videos(
            task_id,
            params,
            downloaded_videos,
            audio_file,
            subtitle_path,
            audio_duration,
        )
    )

    if not final_video_paths:
        return _mark_task_failed(
            task_id,
            "video",
            "failed to generate final video",
        )

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. Сначала завершаем задачу генерации видео и лишь затем при необходимости
    # отправляем кросспостинг. Загрузка к стороннему сервису может занять минуты: она не должна задерживать возврат результата видео и влиять на уже собранный ролик.
    cross_post_enabled = (
        upload_post.upload_post_service.is_configured()
        and upload_post.upload_post_service.auto_upload
    )
    platforms = (
        list(upload_post.upload_post_service.platforms) if cross_post_enabled else []
    )
    should_cross_post = cross_post_enabled and bool(platforms)
    if cross_post_enabled and not platforms:
        logger.warning(
            f"skip cross-post because no platforms are configured, task_id: {task_id}"
        )
    cross_post_state = const.CROSS_POST_STATE_PENDING if should_cross_post else None

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_state": cross_post_state,
        "cross_post_results": None,
        "cross_post_error": None,
        "cross_post_owner": _cross_post_process_owner if should_cross_post else None,
        "warnings": generation_warnings or None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )

    if should_cross_post:
        scheduling_error = _schedule_cross_post(
            task_id=task_id,
            video_paths=final_video_paths,
            params=params,
            video_script=video_script,
            platforms=platforms,
            youtube_privacy_status=(
                upload_post.upload_post_service.youtube_privacy_status
            ),
        )
        # Переполненная очередь и закрытый пул потоков — синхронно наблюдаемые сбои
        # планирования. Статус задачи уже обновила функция планирования, здесь синхронно правим возвращаемый снимок, чтобы вызывающая сторона не получила pending, расходящийся с последующим запросом.
        if scheduling_error:
            kwargs["cross_post_state"] = const.CROSS_POST_STATE_FAILED
            kwargs["cross_post_error"] = scheduling_error
            kwargs["cross_post_owner"] = None

    return kwargs


def start(
    task_id,
    params: VideoParams,
    stop_at: str = "video",
    voice_preview: dict | None = None,
    loomloom_video_request: loomloom.LoomLoomConfirmedVideoRequest | None = None,
    allow_server_file_input: bool = False,
):
    """
    Выполняет конвейер задачи и гарантирует, что даже непредвиденное исключение
    превратится в наблюдаемый статус ошибки.

    ``allow_server_file_input`` предназначен только для локального CLI. HTTP API и
    WebUI обязаны оставлять значение по умолчанию, чтобы пользовательское аудио
    всегда оставалось в границах каталога текущей задачи.
    """
    try:
        return _run_pipeline(
            task_id,
            params,
            stop_at=stop_at,
            voice_preview=voice_preview,
            loomloom_video_request=loomloom_video_request,
            allow_server_file_input=allow_server_file_input,
        )
    except Exception as exc:
        logger.exception(
            f"unexpected task pipeline failure, task_id: {task_id}, error: {exc}"
        )
        return _mark_task_failed(
            task_id,
            "pipeline",
            f"{type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
