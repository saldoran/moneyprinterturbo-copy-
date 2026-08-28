import itertools
import io
import os
import random
import gc
import subprocess
import sys
import tempfile
import unicodedata
from contextlib import ExitStack, redirect_stdout
from functools import lru_cache
from typing import List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageDraw, ImageFont

from app.config import config
from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, utils

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# Связка ffmpeg и AAC в Docker при настройках по умолчанию чаще даёт колебания
# качества звука, поэтому битрейт аудио повышен явно: иначе слишком низкое умолчание внесёт заметные искажения на этапе сборки ролика.
audio_bitrate = "192k"
fps = 30
# При склейке и перекодировании по частоте кадров FFmpeg может дать итоговую
# длительность на десятки миллисекунд короче теоретической, которую видит MoviePy.
# Даём видеоматериалу небольшой запас, чтобы в конце аудио не появились чёрный
# экран, подвисание или кусочек закадрового текста без картинки.
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# Мессенджеры и часть кодировщиков округляют размеры кадра вниз: например, WhatsApp
# сжимает материал 9:16 до 478x850 — на два пикселя меньше 480. Жёсткая планка в
# 480 отбраковала бы такие материалы целиком, и всё падало бы с "no valid materials
# found". Небольшой допуск пропускает материалы, которые ниже порога лишь из-за
# округления, и по-прежнему отсекает действительно низкое разрешение.
_MIN_DIMENSION_TOLERANCE = 10
_DEFAULT_VIDEO_CODEC = "libx264"
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()


def _get_required_video_duration(audio_duration: float) -> float:
    """
    Возвращает целевую длительность склейки видеоматериалов.

    Сценарий: при сборке видео длительность материалов должна перекрывать
    закадровое аудио. Если сделать её ровно равной длительности аудио, FFmpeg
    из-за округления по частоте кадров может выдать чуть более короткое видео,
    поэтому единообразно добавляется небольшой запас. Функция вынесена отдельно,
    чтобы её было удобно тестировать и подстраивать запас по реальным отзывам.
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    Определяет, достаточно ли разрешения материала для сборки.

    Номинальный минимум — 480x480, но допускается отклонение на
    `_MIN_DIMENSION_TOLERANCE` пикселей: это совместимость с размерами, которые
    кодировщики и мессенджеры округляют вниз (например, 478x850 у WhatsApp).
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
) -> List[SubClippedVideoClip]:
    """
    Старается использовать каждый исходный материал по одному разу, снижая
    вероятность повторов одного и того же материала в готовом ролике.

    В онлайн-материалах часто встречается ситуация «одно длинное видео нарезано на
    несколько коротких фрагментов». Прежняя логика в режиме random просто
    перемешивала все фрагменты, из-за чего куски одного исходного видео могли
    оказаться и в начале, и в середине, — пользователь воспринимал это как повтор
    материала. Функция меняет только порядок фрагментов: сначала идёт самый
    длинный фрагмент каждого исходного файла, остальные остаются про запас; если
    суммарной длительности материалов не хватает, последующие фрагменты
    по-прежнему могут добрать длину аудио, чтобы не снижать долю успешных
    генераций. Самый длинный фрагмент выбирается первым, чтобы случайно не взять
    мелкие обрывки из хвоста видео и не начать переиспользовать материал, когда
    его на самом деле достаточно.
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        return subclipped_items

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    Совместимость с теми местами, которые исторически читали путь к FFmpeg прямо
    из сервиса video.

    Настоящая логика определения вынесена в `app.utils.utils.get_ffmpeg_binary()`,
    и видео, речь и будущие цепочки обязаны использовать один и тот же приоритет.
    Здесь остаётся тонкая обёртка, чтобы прямой импорт
    `app.services.video.get_ffmpeg_binary` из внешних скриптов или старых тестов
    не приводил к AttributeError.
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    Читает выбранный пользователем видеокодировщик.

    Настройка рассчитана на опытных пользователей и позволяет попробовать
    аппаратное кодирование NVENC, AMF, QSV, VideoToolbox. Здесь намеренно
    допускается только фиксированный белый список: открой мы произвольные
    параметры FFmpeg, ошибка пользователя сделала бы формат вывода
    непредсказуемым и уронила бы задачу на более позднем этапе.
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    Проверяет, заявляет ли текущий FFmpeg поддержку указанного кодировщика.

    Это доказывает лишь то, что encoder включён в сборку FFmpeg, но не то, что
    железо и драйверы на этой машине его потянут. Поэтому при реальном сбое
    кодирования всё равно происходит откат к libx264.
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    Возвращает кодировщик, который будет использован в этот раз.

    Когда пользователь выбрал аппаратный кодировщик, сперва проверяется список
    encoder у FFmpeg; если в этом процессе кодирование уже падало на практике,
    происходит немедленный откат, чтобы в рамках одной задачи не повторять сбой на
    каждом фрагменте.
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    После сбоя аппаратного кодирования повторяет попытку через libx264 и отключает
    аппаратный кодировщик только при успешном повторе.

    В Windows причины сбоя FFmpeg разнообразны: и неподдерживаемые видеокарта или
    драйвер, и общие проблемы ввода-вывода — занятый выходной файл, права на
    каталог, вмешательство антивируса. Лишь когда libx264 успешно записывает
    результат, можно заключить, что исходный сбой с большой вероятностью вызван
    самим аппаратным кодировщиком, и не задеть последующие задачи.
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(clip, output_file: str, codec: str, **kwargs):
    """
    Записывает видео указанным кодировщиком, при сбое автоматически повторяя
    попытку через libx264.

    Доступность аппаратного кодировщика зависит не только от FFmpeg, но и от
    видеокарты, драйвера и текущего окружения. Задача генерации не должна падать
    целиком из-за недоступности продвинутого кодировщика, поэтому откат собран
    здесь в одном месте.
    """
    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # Демультиплексор concat заключает пути в одинарные кавычки, поэтому одинарные кавычки в пути нужно предварительно экранировать.
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    Формирует путь для списка файлов демультиплексора concat.

    Официальная документация FFmpeg требует экранировать спецсимволы и пробелы в
    списке concat; обратные слэши в абсолютных путях Windows к тому же легко
    принимаются за экранирующие символы. Здесь всё единообразно переводится в
    прямые слэши, так что `C:\\Users\\...` становится `C:/Users/...`, затем
    обрабатываются одинарные кавычки — это совместимо и с macOS, и с Linux.
    """
    absolute_path = os.path.abspath(file_path)
    return _escape_ffmpeg_concat_path(absolute_path.replace("\\", "/"))


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    concat_list_file = os.path.join(output_dir, "ffmpeg-concat-list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as fp:
        for clip_file in clip_files:
            fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
            "-pix_fmt",
            "yuv420p",
        ]
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(output_file)
        return command

    def run_concat(codec: str):
        command = build_command(codec)
        # Через ffmpeg делаем склейку и кодирование за один проход, чтобы MoviePy не
        # перекодировал каждый фрагмент по очереди: так ниже риск потери качества и смещения цвета.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
        return codec

    try:
        effective_codec = _get_effective_video_codec()
        try:
            return run_concat(effective_codec)
        except Exception as exc:
            if effective_codec == _DEFAULT_VIDEO_CODEC:
                raise
            result_codec = run_concat(_DEFAULT_VIDEO_CODEC)
            _disable_runtime_video_codec(effective_codec, str(exc))
            return result_codec
    finally:
        delete_files(concat_list_file)


def _sanitize_image_file(image_path: str) -> str:
    # Некоторые локальные изображения Pillow открывает, но из-за повреждённых
    # метаданных EXIF/eXIf ImageClip бросает исключение уже на разборе. Здесь пересохраняем «чистую картинку», отбрасывая испорченные метаданные.
    image_root, _ = os.path.splitext(image_path)
    sanitized_path = f"{image_root}.sanitized.png"

    with Image.open(image_path) as image:
        image.load()
        # Сохраняем единообразно в PNG, чтобы разные пути метаданных JPEG и PNG не протащили испорченный блок дальше.
        cleaned_image = Image.new(image.mode, image.size)
        cleaned_image.putdata(list(image.getdata()))
        cleaned_image.save(sanitized_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # Сначала пробуем открыть исходное изображение; если помешали повреждённые метаданные, пробуем сделать копию без них.
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    Тихо открывает видеофайл, не давая MoviePy 2.1.x печатать сведения от ffmpeg
    прямо в stdout.

    Предыстория:
    в текущей версии зависимости внутри `FFMPEG_VideoReader` есть
    `print(self.infos)` и `print(ffmpeg command)`, и при чтении промежуточного
    видео без звуковой дорожки выводится `audio_found: False`. Это лишь метаданные
    входного материала, а не признак того, что в готовом ролике не будет звука, но
    пользователь WebUI или терминала решит, что генерация провалилась.

    Реализация:
    1. stdout перенаправляется только на короткое окно открытия VideoFileClip;
    2. по умолчанию `audio=False`, поскольку на этапе видеоматериалов исходный
       звук проекту не нужен — финальное аудио подключается на этапе
       `generate_video()`;
    3. если зависимость всё же что-то вывела, это понижается до debug-лога, чтобы
       при необходимости можно было разобраться.
    """
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # При зацикливании для добора длительности один и тот же путь временного
    # фрагмента встречается в списке склейки FFmpeg несколько раз. Склейке дубликаты
    # нужны, а удалять файл можно только один раз; здесь дубликаты убираются с
    # сохранением исходного порядка, чтобы поведение было идемпотентным для всех
    # вызывающих сторон и после первого успешного удаления не сыпались FileNotFoundError.
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # Уборка допускает, что файла уже нет — например, при сбое FFmpeg или если его
            # уже забрала параллельная очистка; это не проблема, требующая внимания пользователя, и засорять лог генерации ей не стоит.
            continue
        except OSError as e:
            # Права, файловая система только для чтения или сбой диска оставят настоящий
            # временный файл, поэтому предупреждение сохраняем: по конкретному пути и системной ошибке проще найти проблему окружения.
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # bgm_file в запросе API приходит от пользователя, поэтому разрешать его можно
            # только в каталоге пользовательских BGM или встроенных композиций — чтобы MoviePy не прочитал конфигурацию, ключи и любые другие файлы сервера.
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # Когда каталог фоновой музыки пуст, откатываемся к «без BGM», чтобы random.choice([]) не бросил исключение.
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
) -> str:
    audio_clip = AudioFileClip(audio_file)
    try:
        # Здесь нужна только длительность закадрового аудио, чтобы определить длину
        # склейки материалов; дальше audio_clip не используется. Закрываем сразу после чтения, чтобы ранний выход или ветка с исключением не утекли файловым дескриптором.
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # Совместимость с прямым вызовом API без указания режима переходов, чтобы обращение к .value дальше не упало.
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # Записываем итоговое значение один раз: так проще заметить нормализацию
        # выходящих за диапазон параметров API и не дублировать одинаковый лог в горячем цикле по фрагментам.
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    # max_clip_duration ограничивает итоговую длительность воспроизведения в ролике,
    # а не длительность чтения исходного видео. MoviePy на скорости 0.5x из 1.5 секунды
    # исходника даст 3-секундный фрагмент, и на 2x из 6 секунд — тоже 3-секундный.
    # Значит, перед нарезкой длительность исходника нужно пересчитать по скорости:
    # иначе, читая фиксированные 3 секунды с последующим замедлением и обрезкой, мы
    # начнём следующий фрагмент с третьей секунды исходника и пропустим 1.5 секунды
    # картинки. Этот расчёт заодно делает исходную временную шкалу непрерывной и без
    # перекрытий при любой скорости.
    source_clip_duration = max_clip_duration * normalized_clip_speed
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = _open_video_clip_quietly(video_path)
        clip_duration = clip.duration
        clip_w, clip_h = clip.size
        close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)

            # Сохраняем все пригодные отрезки.
            # Так мы не потеряем материалы, которые сами по себе короче max_clip_duration,
            # и не проглотим небольшой остаток в конце длинного видео.
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                        source_file_path=video_path,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
    )
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    for i, subclipped_item in enumerate(subclipped_items):
        if video_duration >= required_video_duration:
            break
        
        logger.debug(
            f"processing clip {i+1}: {subclipped_item.width}x{subclipped_item.height}, "
            f"source: {os.path.basename(subclipped_item.source_file_path)}, "
            f"current duration: {video_duration:.2f}s, "
            f"remaining: {required_video_duration - video_duration:.2f}s"
        )
        
        try:
            clip = _open_video_clip_quietly(subclipped_item.file_path).subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            # Скорость воспроизведения — свойство самого материала, и применять её нужно до
            # переходов. Тогда секундные переходы вроде Fade и Slide не превратятся вслед за
            # скоростью материала в 0.5 или 2 секунды; последующая обрезка по максимальной
            # длительности остаётся подстраховкой от погрешности с плавающей точкой и
            # аномальной длительности материала, гарантируя, что фрагмент не выйдет за лимит.
            if normalized_clip_speed != 1.0:
                clip = clip.with_speed_scaled(normalized_clip_speed)
            clip_duration = clip.duration
            # Not all videos are same size, so we need to resize them
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(f"resizing clip, source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, target: {video_width}x{video_height}, ratio: {video_ratio:.2f}")
                
                if clip_ratio == video_ratio:
                    clip = clip.resized(new_size=(video_width, video_height))
                else:
                    if clip_ratio > video_ratio:
                        scale_factor = video_width / clip_w
                    else:
                        scale_factor = video_height / clip_h

                    new_width = int(clip_w * scale_factor)
                    new_height = int(clip_h * scale_factor)

                    background = ColorClip(size=(video_width, video_height), color=(0, 0, 0)).with_duration(clip_duration)
                    clip_resized = clip.resized(new_size=(new_width, new_height)).with_position("center")
                    clip = CompositeVideoClip([background, clip_resized])
                    
            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if transition_value in (None, VideoTransitionMode.none.value):
                clip = clip
            elif transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.zoom_in.value:
                clip = video_effects.zoomin_transition(clip, 1)
            elif transition_value == VideoTransitionMode.zoom_out.value:
                clip = video_effects.zoomout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                    lambda c: video_effects.zoomin_transition(c, 1),
                    lambda c: video_effects.zoomout_transition(c, 1),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)
                
            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i+1}.mp4"
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            # Store clip duration before closing
            clip_duration_saved = clip.duration
            close_clip(clip)

            processed_clips.append(
                SubClippedVideoClip(
                    file_path=clip_file,
                    duration=clip_duration_saved,
                    width=clip_w,
                    height=clip_h,
                    source_file_path=subclipped_item.source_file_path,
                )
            )
            video_duration += clip_duration_saved
            
        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")
    
    # loop processed clips until the video duration covers the audio duration and the small safety margin.
    if video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s), looping clips to match audio length."
        )
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= required_video_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(
            f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, "
            f"required duration: {required_video_duration:.2f}s, "
            f"looped {len(processed_clips)-len(base_clips)} clips"
        )
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        return combined_video_path
    
    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    concat_video_clips_with_ffmpeg(
        clip_files=clip_files,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
        max_duration=audio_duration,
    )
    
    # clean temp files
    delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # Перенос строк в субтитрах обязан произойти до реального создания TextClip:
    # иначе MoviePy рассчитает область отрисовки по исходному тексту. Здесь ширина
    # измеряется через PIL текущим шрифтом и кеглем, чтобы каждая строка по возможности укладывалась в доступную ширину видео и крупный кегль или длинная китайская фраза не вылезали за кадр.
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    # getbbox() возвращает «высоту видимого следа текущих глифов», а не межстрочный
    # интервал шрифта. Например, у английского текста только из символов вроде A, m, n
    # нет выносных элементов вниз, и descent отсутствует; на нескольких строках эта
    # погрешность накапливается и в итоге обрезает последнюю строку TextClip холстом.
    # ascent + descent берутся из самого шрифта, не зависят от языка и набора символов
    # и согласуются с baseline-моделью отрисовки в MoviePy.
    ascent, descent = font.getmetrics()
    line_height = int(ascent + descent)
    if line_height <= 0:
        # Нормальные шрифты TrueType и OpenType сюда не попадают; диагностический лог и
        # запасной кегль оставлены, чтобы повреждённый или нестандартный шрифт с аномальными метриками не дал субтитры нулевой высоты.
        logger.warning(
            "invalid subtitle font metrics, fallback to font size: "
            f"ascent={ascent}, descent={descent}, fontsize={fontsize}"
        )
        line_height = max(1, int(fontsize))

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, line_height
        left, top, right, bottom = font.getbbox(inner_text)
        # bbox по-прежнему годится, чтобы измерить реальную ширину для переноса; для высоты всегда используем стабильный межстрочный интервал шрифта.
        return right - left, line_height

    width, height = get_text_size(text)
    if width <= max_width:
        # В записи SRT автор вправе поставить перенос вручную. Даже если по ширине весь
        # текст переносить больше не нужно, высоту холста всё равно нужно считать по имеющемуся числу строк — иначе вторая и следующие строки обрежутся.
        return text, (text.count("\n") + 1) * line_height

    def split_long_token(token):
        # Когда сам token шире доступной ширины (типично для длинных китайских фраз без
        # пробелов или сверхдлинных английских слов), переходим к посимвольному разбиению.
        # Ключевой момент: заметив, что candidate стал слишком широким, сначала фиксируем
        # предыдущий ещё допустимый current, а текущий символ переносим на следующую
        # строку — возвращать слишком широкий символ назад нельзя.
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # При посимвольном разбиении длинной китайской фразы последняя закрывающая
        # пунктуация — точка, запятая — может уехать на отдельную строку, из-за чего фон
        # субтитров неестественно растягивается и визуально выглядит как точка, упавшая
        # под текст. Не переделывая алгоритм переноса, переносим последний символ
        # предыдущей строки к строке с пунктуацией, чтобы знак шёл вместе с текстом; это
        # работает и для китайской, и для английской закрывающей пунктуации.
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        candidate = f"{lines[index - 1][-1]}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-1]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    # Высоту определяем по итоговому результату. Явный перенос строки из исходного
    # текста мог остаться внутри одного из token, и тогда длина временного списка lines не совпадёт с числом строк, которые реально отрисует MoviePy.
    height = (result.count("\n") + 1) * line_height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # Цвет фона субтитров приходит из параметров API и WebUI и может быть пустым или
    # записанным неверно. Принимаем только форму #RRGGBB, а недопустимое значение откатываем к чёрному, чтобы PIL не бросил исключение на этапе отрисовки и не оборвал задачу.
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        except ValueError:
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # Новый фон субтитров используется только при явном включении пользователем:
    # полупрозрачная подложка со скруглёнными углами рисуется картинкой RGBA и
    # передаётся в MoviePy прозрачным ImageClip. Путь по умолчанию при этом не меняется вовсе, а более мягкую подачу субтитров можно опробовать малой ценой.
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    Помещает TextClip в центр контейнера фона по реальным видимым пикселям текста.

    TextClip в MoviePy создаёт прозрачный холст по межстрочному интервалу шрифта и
    baseline. У многих шрифтов видимые глифы не совпадают с геометрическим центром
    этого холста, и прямой `with_position("center")` отцентрирует весь прозрачный
    холст — субтитры визуально уедут вверх или вниз. Здесь читается прозрачная
    маска TextClip, а смещение считается только по bbox реально закрашенных
    пикселей, чтобы видимый пользователю текст стоял по центру фона субтитров.
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """Определяет, совпадают ли цвета текста и фона субтитров, чтобы предупредить пользователя о нечитаемых субтитрах."""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """Проверяет наличие в шрифте глифов, нужных для образца текста, и кеширует результат повторных проверок."""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # Неудачная проверка шрифта не должна мешать пользователю генерировать; лог оставляем для разбора проблем совместимости окружения.
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """Проверяет, может ли шрифт отрисовать буквы и цифры из текста, игнорируя пробелы и знаки препинания."""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    Собирает итоговое видео и сообщает, успешно ли обработана фоновая музыка.

    Возвращаемое значение описывает только состояние BGM: True, если музыка не
    запрашивалась либо успешно сведена; False, если музыка запрашивалась, но
    загрузка, эффекты или сведение не удались. Даже при сбое BGM видео с одним
    закадровым голосом всё равно записывается, а показывать ли пользователю
    предупреждение о деградации, решает слой оркестрации задач.
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # Совместимость с историческим параметром: в API `text_background_color` может
        # быть как булевым значением, так и строкой с цветом. Нормализуем здесь, чтобы
        # передача True/False прямо в TextClip не дала непредсказуемый результат отрисовки.
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # Скруглённый фон строится по реальной ширине текста, поэтому отступы слева и
        # справа должны быть скромнее; у прежнего прямоугольного фона запас остаётся больше, чтобы длинные субтитры из старых конфигураций не липли к краю и не обрезались.
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # Фону субтитров нужны заметные внутренние отступы слева и справа. Сначала
        # вычитаем padding из доступной ширины и лишь потом переносим строки: иначе
        # длинный английский текст или крупный кегль ровно заполнят 90% ширины видео,
        # текст прижмётся к краю рамки и будет выглядеть обрезанным. По этой ветке идут
        # и обычный прямоугольный, и скруглённый фон; у субтитров без фона максимальная
        # ширина остаётся прежней.
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # Pillow и MoviePy расширяют обводку вверх и вниз от глифов и учитывают это в
        # шаге каждой строки. Если добавить запас под обводку один раз на весь блок
        # субтитров, при толстой обводке и нескольких строках погрешность всё равно
        # накопится построчно. Здесь место под двустороннюю обводку считается по реальному
        # числу строк: тонкая обводка по умолчанию добавляет совсем немного высоты, а
        # сочетание «мелкий кегль + толстая обводка + много строк» отображается целиком.
        stroke_padding = int(params.stroke_width * 2 * line_count)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # В режиме `method=label` MoviePy автоматически ужимает высоту текстового блока и
        # при многострочных субтитрах, обводке или цветном фоне легко обрезает нижнюю
        # половину последней строки. Поэтому передаём более консервативную высоту явно,
        # закладывая межстрочный интервал и дополнительные отступы сверху и снизу, чтобы
        # и рамка фона субтитров, и сам текст отрисовались полностью.
        clip_h = int(
            txt_height
            + vertical_padding
            + (interline * line_count)
            + stroke_padding
        )

        if rounded_bg_enabled:
            # Скруглённый фон должен облегать ширину текста, а не наследовать 90% ширины
            # видео. Сначала измеряем самую длинную строку через PIL, затем добавляем горизонтальные внутренние отступы — так у коротких субтитров не будет чрезмерно широкой подложки.
            try:
                font = ImageFont.truetype(font_path, params.font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)
        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    # CompositeAudioClip.close() в MoviePy не закрывает вложенные AudioFileClip. Здесь
    # через ExitStack явно удерживаются все исходные файловые reader: так пути успеха,
    # сбоя субтитров, неудачного сведения и неудачной записи видео одинаково освобождают дочерние процессы FFmpeg — особенно важно, чтобы в Windows файлы не оставались занятыми.
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if subtitle_path and os.path.exists(subtitle_path):
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=subtitle_path,
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            text_clips = []
            for item in sub.subtitles:
                clip = create_text_clip(subtitle_item=item)
                text_clips.append(clip)
            video_clip = CompositeVideoClip([video_clip, *text_clips])
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # Это правило короткого замыкания общее для всех источников BGM. При громкости не
            # больше 0 нельзя ни разрешать случайный или пользовательский файл, ни загружать файл от поставщика — незачем зря делать ввод-вывод и сведение.
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # Музыку от поставщика слой оркестрации задач может передать файлом напрямую.
        # None означает «использовать обычное разрешение случайного или пользовательского BGM», пустая строка явно отключает BGM для этого ролика; но любой источник сперва обязан пройти общее правило громкости.
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                ]
                # Случайная или пользовательская музыка, разрешённая внутри сервиса, может быть
                # короче ролика, и её нужно зациклить; файл, переданный слоем задач через override,
                # означает, что поставщик уже подогнал длительность. Решение о зацикливании принимаем по источнику файла, чтобы не править белый список имён при каждом новом поставщике.
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # Записываем полный стек и устойчивый контекст, чтобы отличить сбой декодирования
                # файла от сбоя эффектов MoviePy и от сбоя CompositeAudioClip; содержимое файла и API-ключ в лог не попадают.
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # Явно наследуем частоту дискретизации входного аудио; если её не получить,
        # откатываемся к принятым в MoviePy 44100 Гц. Это уменьшает колебания качества из-за повторной передискретизации в разных окружениях, особенно в Docker.
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        _write_videofile_with_codec_fallback(
            final_video_clip,
            output_file=output_file,
            codec=_get_configured_video_codec(),
            audio_codec=audio_codec,
            audio_fps=output_audio_fps,
            audio_bitrate=audio_bitrate,
            temp_audiofile_path=_get_temp_audio_dir(output_dir),
            threads=params.n_threads or 2,
            logger=None,
            fps=fps,
        )
        return bgm_mix_succeeded


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    # В некоторых сценариях повторной генерации WebUI может передать пустой список материалов; сразу возвращаем пустой результат, чтобы не получить исключение NoneType.
    if not materials:
        return []

    # Возвращаем только материалы, прошедшие предварительную проверку, чтобы изображения низкого разрешения не попали в дальнейшую сборку видео.
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # Пути материалов для local video_source приходят из параметров API и обязаны
            # оставаться в выделенном каталоге материалов. Пользователь может передать имя
            # файла, поддерживается и исторический абсолютный путь, но выход в другие каталоги системы запрещён — иначе возможно чтение произвольных файлов или зондирование локальных секретов через MoviePy.
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        try:
            # Изображения читаем сразу как изображения, чтобы не пройти сперва через VideoFileClip, ошибиться в определении типа и попасть в нестабильную запасную ветку.
            if ext in const.FILE_TYPE_IMAGES:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # При нестандартном расширении или неудачном определении откатываемся к режиму изображения — это совместимо с исторической передачей путей к локальным картинкам.
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # Обнаружив материал низкого разрешения, сразу освобождаем ресурсы и не отдаём его в дальнейшую обработку.
                close_clip(clip)
                continue

            if ext in const.FILE_TYPE_IMAGES:
                logger.info(f"processing image: {material_source_path}")
                # Материал уже открывался при определении размера, поэтому сперва освобождаем этот дескриптор и лишь затем заново создаём image clip для экспорта.
                close_clip(clip)
                # Create an image clip and set its duration to 3 seconds
                clip = (
                    ImageClip(material_source_path)
                    .with_duration(clip_duration)
                    .with_position("center")
                )
                # Apply a zoom effect using the resize method.
                # A lambda function is used to make the zoom effect dynamic over time.
                # The zoom effect starts from the original size and gradually scales up to 120%.
                # t represents the current time, and clip.duration is the total duration of the clip (3 seconds).
                # Note: 1 represents 100% size, so 1.2 represents 120% size.
                zoom_clip = clip.resized(
                    lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
                )

                # Optionally, create a composite video clip containing the zoomed clip.
                # This is useful when you want to add other elements to the video.
                final_clip = CompositeVideoClip([zoom_clip])

                # Output the video to a file.
                video_file = f"{material_source_path}.mp4"
                final_clip.write_videofile(video_file, fps=30, logger=None)
                close_clip(clip)
                close_clip(final_clip)
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # У обычного видеоматериала нужно лишь прочитать размеры для проверки, после чего дескриптор можно сразу освободить.
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
