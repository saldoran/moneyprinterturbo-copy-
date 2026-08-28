from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
from typing import TYPE_CHECKING, Any, Sequence
from uuid import UUID, uuid4

from loguru import logger

if TYPE_CHECKING:
    from app.models.schema import MaterialInfo, VideoParams


DEFAULT_VOICE_NAME = "zh-CN-XiaoxiaoNeural-Female"
# Соответствует VOICE_MODE_NONE и VOICE_MODE_UPLOAD из webui/Main.py. Общих
# констант у этих двух сторон пока нет, поэтому здесь оставлены литералы с указанием источника.
UI_VOICE_MODE_NONE = "none"
# Встроенное значение позиции субтитров по умолчанию. В VideoParams есть
# одноимённое умолчание, но это умолчание поля Pydantic: config.ui читается один
# раз при импорте модуля, и недопустимое значение из config.toml замораживается
# прямо в модель — без проверки и без возможности подменить в тесте. Поэтому CLI
# держит собственную копию, чтобы правило «недопустимое сохранённое значение
# откатывается к умолчанию» всегда работало для командной строки.
DEFAULT_SUBTITLE_POSITION = "bottom"
DEFAULT_CUSTOM_POSITION = 70.0
UI_VOICE_MODE_UPLOAD = "upload"
# Оба этих сохранённых способа озвучки означают отказ от автоозвучки.
UI_VOICE_MODES_WITHOUT_TTS = frozenset({UI_VOICE_MODE_NONE, UI_VOICE_MODE_UPLOAD})
_PIPELINE_STAGES = ("script", "terms", "audio", "subtitle", "materials", "video")
_CUSTOM_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_BATCH_FILE_MAX_BYTES = 1024 * 1024
_BATCH_TASK_MAX_COUNT = 100


class _CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Сохраняет многострочную вёрстку примеров и при этом автоматически показывает осмысленные значения по умолчанию."""

    def _get_help_string(self, action):
        help_text = action.help or ""
        if (
            "%(default)" not in help_text
            and action.default not in (None, "", argparse.SUPPRESS)
            and action.option_strings
            and "default:" not in help_text.lower()
        ):
            help_text += " (default: %(default)s)"
        return help_text


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"value must be >= 1, got {parsed}")
    return parsed


def _paragraph_count(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 10:
        raise argparse.ArgumentTypeError(
            f"paragraph-number must be between 1 and 10, got {parsed}"
        )
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"value must be a finite number >= 0, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"value must be a finite number > 0, got {value!r}")
    return parsed


def _percent_position(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 100:
        raise argparse.ArgumentTypeError(
            f"custom-position must be a finite number between 0 and 100, got {value!r}"
        )
    return parsed


def _hex_color(value: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise argparse.ArgumentTypeError(
            f"color must use #RRGGBB format, got {value!r}"
        )
    return value


def _subtitle_position(value: str) -> str:
    """Проверяет сохранённую позицию субтитров; диапазон значений совпадает с аргументом командной строки."""
    if value not in ("top", "center", "bottom", "custom"):
        raise argparse.ArgumentTypeError(
            f"subtitle-position must be one of: top, center, bottom, custom, got {value!r}"
        )
    return value


def _task_id(value: str) -> str:
    """Пользовательский идентификатор задачи в CLI принимается только как UUID, чтобы значение не истолковали как путь в файловой системе."""
    try:
        return str(UUID(value.strip()))
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"task-id must be a valid UUID, got {value!r}"
        ) from exc


_TRANSITION_MODE_VALUES = {
    "none": None,
    "shuffle": "Shuffle",
    "fade-in": "FadeIn",
    "fade-out": "FadeOut",
    "slide-in": "SlideIn",
    "slide-out": "SlideOut",
}


def _transition_mode(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized not in _TRANSITION_MODE_VALUES:
        allowed = ", ".join(_TRANSITION_MODE_VALUES)
        raise argparse.ArgumentTypeError(
            f"video-transition-mode must be one of: {allowed}"
        )
    return _TRANSITION_MODE_VALUES[normalized]


def _bgm_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "none":
        return ""
    if normalized in {"", "random", "custom", "sonilo"}:
        return normalized
    raise argparse.ArgumentTypeError(
        "bgm-type must be one of: none, random, custom, sonilo"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MoneyPrinterTurbo videos without the WebUI.\n\n"
            "Provider settings and credentials are read from config.toml.\n"
            "Default full-video generation requires a configured LLM and Pexels API key.\n"
            "The default Edge TTS voice requires no API key."
        ),
        epilog="""
Examples:
  Generate a complete video with the default Edge TTS voice:
    uv run python cli.py --video-subject "How AI is changing everyday life"

  Generate from local files. Relative paths use the current working directory;
  absolute paths are also accepted:
    uv run python cli.py --video-subject "How AI is changing everyday life" \\
      --video-source local --video-materials "./1.mp4,./2.mp4"

  Generate with a prepared script and no voiceover:
    uv run python cli.py --video-script "Your complete script" \\
      --voice-name no-voice --stop-at video

  Stop after script generation:
    uv run python cli.py --video-subject "How AI is changing everyday life" --stop-at script

  Run a JSON array or JSONL manifest. CLI options provide defaults and each object
  overrides VideoParams fields for one task:
    uv run python cli.py --batch-file ./tasks.jsonl --stop-at script

Pipeline stages:
  script     Generate or return the script.
  terms      Generate material search terms; unavailable with local materials.
  audio      Generate TTS, silent audio, or use --custom-audio-file.
  subtitle   Generate subtitles when enabled.
  materials  Download online materials or preprocess local files.
  video      Generate the final video and run configured cross-posting.
  The command stops immediately after the selected stage and prints that stage's result.

Output and exit status:
  Task files are written to storage/tasks/<task-id>/. A successful command prints one
  JSON object to stdout and exits with 0. Batch mode prints one JSON summary after all
  tasks finish. Task failures exit with 1; argument or manifest errors exit with 2
  before any batch task starts. Runtime logs are written to stderr.

Batch manifests:
  --batch-file accepts a UTF-8 JSON array or JSONL file with one object per non-empty
  line (up to 100 tasks and 1 MiB). Objects may override VideoParams fields; unknown
  fields are rejected. Every merged task needs video_subject or video_script.
  The manifest path is relative to the current working directory. Relative
  custom_audio_file and local video_materials[].url values declared in a manifest are
  relative to the manifest directory. Relative paths supplied by CLI options remain
  relative to the current working directory. bgm_file and font_name keep their managed
  storage/resource lookup rules. --stop-at is a batch-wide CLI option. The summary has
  total, succeeded, failed, and tasks keys; every task entry has index, task_id,
  status, result, failed_stage, and error.
""",
        formatter_class=_CliHelpFormatter,
    )

    content_group = parser.add_argument_group("script and content")
    content_group.add_argument(
        "--video-subject",
        default="",
        help="video topic; required unless --video-script is provided",
    )
    content_group.add_argument(
        "--video-script",
        default="",
        help="complete script; skips LLM script generation when provided",
    )
    content_group.add_argument(
        "--video-terms",
        default=None,
        help="comma-separated material search terms; generated automatically when omitted",
    )
    content_group.add_argument(
        "--video-language",
        default=None,
        help=(
            "script language code, such as zh-CN or en-US (default: auto-detect)"
        ),
    )
    content_group.add_argument(
        "--paragraph-number",
        type=_paragraph_count,
        default=None,
        help="number of generated script paragraphs, from 1 to 10 (default: 1)",
    )
    content_group.add_argument(
        "--video-script-prompt",
        default=None,
        help="additional requirements for LLM script generation",
    )
    content_group.add_argument(
        "--custom-system-prompt",
        default=None,
        help="replace the default LLM system prompt for script generation",
    )

    material_group = parser.add_argument_group("materials and pipeline")
    material_group.add_argument(
        "--video-source",
        default="pexels",
        choices=["pexels", "pixabay", "coverr", "volcengine_seedance", "local"],
        help="video material provider; online providers require matching API keys in config.toml",
    )
    material_group.add_argument(
        "--video-materials",
        default="",
        metavar="PATH[,PATH...]",
        help=(
            "comma-separated local image/video paths for --video-source local; relative "
            "paths use the current working directory, then storage/local_videos as a "
            "compatibility fallback; absolute paths are accepted"
        ),
    )
    material_group.add_argument(
        "--stop-at",
        default="video",
        choices=_PIPELINE_STAGES,
        help="stop after this pipeline stage; see the stage order below",
    )
    material_group.add_argument(
        "--confirm-seedance-charge",
        action="store_true",
        help=(
            "confirm that Volcano Engine Seedance creates paid Ark tasks; required "
            "with --video-source volcengine_seedance for materials or video output"
        ),
    )

    video_group = parser.add_argument_group("video output")
    video_group.add_argument(
        "--video-count",
        type=_positive_int,
        default=1,
        help="number of output videos, at least 1",
    )
    video_group.add_argument(
        "--video-aspect",
        choices=["9:16", "16:9", "1:1"],
        default="9:16",
        help="output aspect ratio: portrait, landscape, or square",
    )
    video_group.add_argument(
        "--video-concat-mode",
        choices=["random", "sequential"],
        default=None,
        help="source clip concatenation order (default: random)",
    )
    video_group.add_argument(
        "--video-transition-mode",
        type=_transition_mode,
        default=None,
        metavar="{none,shuffle,fade-in,fade-out,slide-in,slide-out}",
        help="transition applied between source clips (default: none)",
    )
    video_group.add_argument(
        "--video-clip-duration",
        type=_positive_int,
        default=None,
        help=(
            "maximum duration of each source clip in seconds, at least 1 (default: 5)"
        ),
    )
    video_group.add_argument(
        "--match-materials-to-script",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "preserve script keyword order while selecting and concatenating materials "
            "(default: disabled)"
        ),
    )
    video_group.add_argument(
        "--n-threads",
        type=_positive_int,
        default=None,
        help="FFmpeg worker thread count, at least 1 (default: 2)",
    )

    audio_group = parser.add_argument_group("voiceover and background music")
    audio_group.add_argument(
        "--voice-name",
        default=None,
        help=(
            f"TTS voice identifier. Defaults to [ui].voice_name from "
            f"config.toml, otherwise {DEFAULT_VOICE_NAME}. A saved "
            "[ui].voice_mode of 'none' or 'upload' resolves to no-voice "
            "instead, unless this option is given. "
            "Use 'no-voice' for silent output. Provider-specific identifiers "
            "use prefixes such as gemini:, mimo:, elevenlabs:, and chatterbox:"
        ),
    )
    audio_group.add_argument(
        "--voice-volume",
        type=_non_negative_float,
        default=None,
        help=(
            "final voiceover volume multiplier, a finite number >= 0 (default: "
            "[ui].voice_volume from config.toml; 1.0 when unset)"
        ),
    )
    audio_group.add_argument(
        "--voice-rate",
        type=_positive_float,
        default=None,
        help=(
            "speech rate multiplier, a finite number > 0 (default: "
            "[ui].voice_rate from config.toml; 1.0 when unset)"
        ),
    )
    audio_group.add_argument(
        "--custom-audio-file",
        default=None,
        metavar="PATH",
        help=(
            "existing MP3/WAV/M4A/AAC/FLAC/OGG voiceover; relative paths use the "
            "current working directory. This skips TTS; set subtitle_provider=whisper "
            "to transcribe it"
        ),
    )
    audio_group.add_argument(
        "--bgm-type",
        type=_bgm_type,
        default=None,
        metavar="{none,random,custom,sonilo}",
        help=(
            "background music mode; Sonilo reads its API key from config.toml or "
            "SONILO_API_KEY; --bgm-file implies custom when omitted "
            "(default: random)"
        ),
    )
    audio_group.add_argument(
        "--sonilo-bgm-prompt",
        default=None,
        help="optional music style prompt for Sonilo, up to 2000 characters",
    )
    audio_group.add_argument(
        "--bgm-file",
        default=None,
        metavar="PATH",
        help=(
            "custom supported audio file inside storage/bgm or resource/songs; "
            "accepts a filename or an allowed managed path"
        ),
    )
    audio_group.add_argument(
        "--bgm-volume",
        type=_non_negative_float,
        default=None,
        help=(
            "background music volume multiplier, a finite number >= 0 (default: 0.2)"
        ),
    )

    subtitle_group = parser.add_argument_group("subtitles")
    subtitle_group.add_argument(
        "--subtitle-enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "enable subtitles; use --no-subtitle-enabled to disable "
            "(default: [ui].subtitle_enabled from config.toml; enabled when "
            "unset)"
        ),
    )
    subtitle_group.add_argument(
        "--font-name",
        default=None,
        help=(
            "subtitle font filename inside resource/fonts "
            "(default: [ui].font_name from config.toml; "
            "STHeitiMedium.ttc when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-position",
        choices=["top", "center", "bottom", "custom"],
        default=None,
        help=(
            "subtitle vertical position (default: [ui].subtitle_position from "
            "config.toml; bottom when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--custom-position",
        type=_percent_position,
        default=None,
        help=(
            "custom position as percent from top, 0-100; requires "
            "--subtitle-position custom (default: [ui].custom_position from "
            "config.toml; 70 when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--text-fore-color",
        type=_hex_color,
        default=None,
        help=(
            "subtitle text color in #RRGGBB format; quote the value in shells "
            "that treat # as a comment (default: [ui].text_fore_color from "
            "config.toml; #FFFFFF when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--font-size",
        type=_positive_int,
        default=None,
        help=(
            "subtitle font size (default: [ui].font_size from config.toml; "
            "60 when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--stroke-color",
        type=_hex_color,
        default=None,
        help=(
            "subtitle outline color in #RRGGBB format (default: "
            "[ui].stroke_color from config.toml; #000000 when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--stroke-width",
        type=_non_negative_float,
        default=None,
        help=(
            "subtitle outline width, a finite number >= 0 (default: "
            "[ui].stroke_width from config.toml; 1.5 when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-background-enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "enable subtitle background; use --no-subtitle-background-enabled to "
            "disable (default: [ui].subtitle_background_enabled from "
            "config.toml; disabled when unset)"
        ),
    )
    subtitle_group.add_argument(
        "--subtitle-background-color",
        type=_hex_color,
        default=None,
        help=(
            "subtitle background color in #RRGGBB format (default: "
            "[ui].subtitle_background_color from config.toml)"
        ),
    )
    subtitle_group.add_argument(
        "--rounded-subtitle-background",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "use a rounded subtitle background (default: "
            "[ui].rounded_subtitle_background from config.toml; "
            "disabled when unset)"
        ),
    )

    execution_group = parser.add_argument_group("execution")
    execution_mode = execution_group.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--task-id",
        type=_task_id,
        default=None,
        help="custom UUID used for storage/tasks/<task-id>; generated automatically when omitted",
    )
    execution_mode.add_argument(
        "--batch-file",
        default=None,
        metavar="PATH",
        help=(
            "UTF-8 JSON array or JSONL task manifest; conflicts with --task-id and "
            "generates a UUID for every task"
        ),
    )
    args = parser.parse_args(argv)

    if (
        not args.batch_file
        and not args.video_subject.strip()
        and not args.video_script.strip()
    ):
        parser.error("one of --video-subject or --video-script is required")

    if not args.batch_file and args.video_source == "local" and args.stop_at == "terms":
        parser.error(
            "--stop-at terms has no effect with --video-source local "
            "(search terms are not generated for local sources)"
        )

    stage_requires_materials = args.stop_at in {"materials", "video"}
    has_video_materials = bool((args.video_materials or "").strip())
    if (
        not args.batch_file
        and args.video_source == "local"
        and stage_requires_materials
        and not has_video_materials
    ):
        parser.error(
            "--video-materials is required with --video-source local when "
            "--stop-at is materials or video"
        )
    if not args.batch_file and args.video_source != "local" and has_video_materials:
        parser.error("--video-materials can only be used with --video-source local")
    if (
        not args.batch_file
        and args.video_source == "volcengine_seedance"
        and stage_requires_materials
        and not args.confirm_seedance_charge
    ):
        parser.error(
            "--confirm-seedance-charge is required with "
            "--video-source volcengine_seedance"
        )

    if args.bgm_file:
        if args.bgm_type in (None, "custom"):
            args.bgm_type = "custom"
        elif not args.batch_file:
            parser.error("--bgm-file can only be combined with --bgm-type custom")

    if args.sonilo_bgm_prompt:
        if args.bgm_type in (None, "sonilo"):
            args.bgm_type = "sonilo"
        elif not args.batch_file:
            parser.error(
                "--sonilo-bgm-prompt can only be combined with --bgm-type sonilo"
            )

    if (
        not args.batch_file
        and args.custom_position is not None
        and args.subtitle_position != "custom"
    ):
        parser.error("--custom-position requires --subtitle-position custom")
    # Конфликтом считается только явный --no-subtitle-enabled. Значение по умолчанию
    # теперь None, а сохранённое выключенное состояние обрабатывается в build_video_params — сообщать здесь об ошибке аргументов не нужно.
    if (
        not args.batch_file
        and args.stop_at == "subtitle"
        and args.subtitle_enabled is False
    ):
        parser.error("--stop-at subtitle cannot be combined with --no-subtitle-enabled")
    if not args.batch_file and args.subtitle_background_enabled is False and (
        args.subtitle_background_color is not None
        or args.rounded_subtitle_background is True
    ):
        parser.error(
            "subtitle background color or rounding cannot be enabled together with "
            "--no-subtitle-background-enabled"
        )

    return args


def _ui_config_value(ui_config, key: str, expected_type, checker=None):
    """
    Читает сохранённые настройки WebUI из ``[ui]``; если пригодного значения нет,
    возвращает ``None``.

    Эту секцию могли править и вручную, поэтому непригодные записи здесь просто
    отбрасываются, а вызывающая сторона откатывается к встроенному умолчанию —
    вместо того чтобы тащить грязные данные дальше и получить traceback или
    ошибку валидации ``VideoParams``.

    ``checker`` переиспользует те же функции типов, что и командная строка
    (например, ``_hex_color``), поэтому сохранённые значения и аргументы
    подчиняются совершенно одинаковым правилам: громкость не может быть
    отрицательной, цвет обязан быть ``#RRGGBB``.
    """
    value = ui_config.get(key)
    if value is None:
        return None
    # ``isinstance(True, int)`` истинно, поэтому там, где ожидается число, bool нужно исключать явно.
    if isinstance(value, bool) != (expected_type is bool):
        return None
    # ``1`` в TOML — целое число, но для полей вроде громкости и скорости речи это тоже допустимое значение.
    if expected_type is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, expected_type):
        return None
    if expected_type is str and not value.strip():
        return None
    if checker is not None:
        try:
            return checker(str(value))
        except argparse.ArgumentTypeError:
            return None
    return value


def _resolve_subtitle_enabled(args: argparse.Namespace, ui_config) -> bool:
    """
    Разбирает переключатель субтитров по приоритету: командная строка > значение,
    сохранённое в WebUI > включено по умолчанию.

    ``--stop-at subtitle`` прямо требует сгенерировать субтитры, поэтому
    сохранённое выключенное состояние не должно превращать этот этап в пустую
    операцию. Сочетание явного --no-subtitle-enabled с этим этапом уже
    отклоняется при проверке аргументов, так что здесь остаётся обработать только
    сохранённое значение.
    """
    if args.subtitle_enabled is not None:
        return args.subtitle_enabled
    if args.stop_at == "subtitle":
        return True
    saved = _ui_config_value(ui_config, "subtitle_enabled", bool)
    return True if saved is None else saved


def _resolve_voice_name(args: argparse.Namespace, ui_config) -> str:
    """
    Разбирает голос по приоритету: командная строка > сохранённые в WebUI способ
    озвучки и голос > встроенное умолчание.

    WebUI сохраняет «без озвучки» отдельным voice_mode, попутно оставляя
    последний реально выбранный пользователем голос, чтобы его можно было вернуть
    при переключении обратно на автоозвучку. Поэтому читать только voice_name
    нельзя: сохранённое состояние «без озвучки» было бы проигнорировано и могло
    бы снова вызвать запрос к платному поставщику.
    """
    from app.services.voice import NO_VOICE_NAME

    if args.voice_name:
        return args.voice_name
    # И «без озвучки», и загрузка собственного аудио означают, что автоозвучка не
    # нужна. Путь к файлу при загрузке в [ui] не пишется, и CLI не может повторить
    # эту загрузку; использование сохранённого голоса молча вызвало бы платный запрос к TTS. Поэтому оба режима отображаются в no-voice, а нужная озвучка задаётся явным --voice-name.
    if _ui_config_value(ui_config, "voice_mode", str) in UI_VOICE_MODES_WITHOUT_TTS:
        return NO_VOICE_NAME
    return _ui_config_value(ui_config, "voice_name", str) or DEFAULT_VOICE_NAME


def build_video_params(args: argparse.Namespace) -> VideoParams:
    # Справке по аргументам и их проверке конфигурация приложения не нужна. Модель
    # импортируется только при реальной сборке параметров задачи, чтобы ``cli.py -h`` не печатал логи инициализации конфигурации.
    from app.config import config
    from app.models.schema import MaterialInfo, VideoParams

    ui_config = config.ui

    video_terms = args.video_terms
    if video_terms:
        video_terms = [
            term.strip() for term in re.split(r"[,，]", video_terms) if term.strip()
        ]

    video_materials = None
    materials_arg = args.video_materials or ""
    if materials_arg.strip():
        video_materials = [
            # Actual duration will be detected during video processing; use 0 as placeholder.
            MaterialInfo(provider="local", url=item.strip(), duration=0)
            for item in materials_arg.split(",")
            if item.strip()
        ]

    params_kwargs = {
        "video_subject": args.video_subject.strip(),
        "video_script": args.video_script,
        "video_terms": video_terms,
        "video_source": args.video_source,
        "video_materials": video_materials,
        "video_count": args.video_count,
        "video_aspect": args.video_aspect,
        "voice_name": _resolve_voice_name(args, ui_config),
        "subtitle_enabled": _resolve_subtitle_enabled(args, ui_config),
    }

    optional_arg_names = [
        "video_language",
        "paragraph_number",
        "video_script_prompt",
        "custom_system_prompt",
        "video_concat_mode",
        "video_transition_mode",
        "video_clip_duration",
        "match_materials_to_script",
        "n_threads",
        "voice_volume",
        "voice_rate",
        "custom_audio_file",
        "bgm_type",
        "bgm_file",
        "bgm_volume",
        "sonilo_bgm_prompt",
        "font_name",
        "subtitle_position",
        "custom_position",
        "text_fore_color",
        "font_size",
        "stroke_color",
        "stroke_width",
        "rounded_subtitle_background",
    ]
    for name in optional_arg_names:
        value = getattr(args, name)
        if value is not None:
            params_kwargs[name] = value

    # Если аргумент командной строки не задан явно, берём значение, сохранённое в
    # WebUI. Дополняем только поля, которые выше не установила командная строка; при отсутствии сохранённого значения остаётся умолчание VideoParams.
    ui_defaults = (
        ("font_name", str, None),
        ("text_fore_color", str, _hex_color),
        ("font_size", int, _positive_int),
        ("rounded_subtitle_background", bool, None),
        ("voice_volume", float, _non_negative_float),
        ("voice_rate", float, _positive_float),
        ("stroke_color", str, _hex_color),
        ("stroke_width", float, _non_negative_float),
    )
    for name, expected_type, checker in ui_defaults:
        if name in params_kwargs:
            continue
        value = _ui_config_value(ui_config, name, expected_type, checker)
        if value is not None:
            params_kwargs[name] = value

    # Позицию субтитров CLI обязан задать определённо и не полагаться на описанное выше умолчание поля, вычисляемое при импорте.
    if "subtitle_position" not in params_kwargs:
        params_kwargs["subtitle_position"] = (
            _ui_config_value(ui_config, "subtitle_position", str, _subtitle_position)
            or DEFAULT_SUBTITLE_POSITION
        )
    if "custom_position" not in params_kwargs:
        saved_custom_position = _ui_config_value(
            ui_config, "custom_position", float, _percent_position
        )
        params_kwargs["custom_position"] = (
            DEFAULT_CUSTOM_POSITION
            if saved_custom_position is None
            else saved_custom_position
        )

    if args.subtitle_background_enabled is False:
        params_kwargs["text_background_color"] = False
        params_kwargs["rounded_subtitle_background"] = False
    elif args.subtitle_background_color is not None:
        params_kwargs["text_background_color"] = args.subtitle_background_color
    elif args.subtitle_background_enabled is True:
        # Пользователь только включил фон, не переопределяя цвет, поэтому в первую
        # очередь берём цвет, сохранённый в WebUI, и лишь при его отсутствии откатываемся к фону по умолчанию.
        params_kwargs["text_background_color"] = (
            _ui_config_value(
                ui_config, "subtitle_background_color", str, _hex_color
            )
            or True
        )
    else:
        # Сочетание «фон выключен» и цвета в командной строке — ошибка аргументов, но
        # то же сочетание в сохранённых настройках не должно прерывать работу: оно просто означает отключённый фон.
        ui_enabled = _ui_config_value(
            ui_config, "subtitle_background_enabled", bool
        )
        ui_color = _ui_config_value(
            ui_config, "subtitle_background_color", str, _hex_color
        )
        if ui_enabled is False:
            params_kwargs["text_background_color"] = False
            if args.rounded_subtitle_background is None:
                params_kwargs["rounded_subtitle_background"] = False
        elif ui_color is not None:
            params_kwargs["text_background_color"] = ui_color
        elif ui_enabled is True:
            params_kwargs["text_background_color"] = True

    return VideoParams(**params_kwargs)


def _load_batch_manifest(raw_path: str) -> tuple[str, list[dict[str, Any]]]:
    expanded_path = os.path.expanduser(raw_path.strip())
    if not expanded_path:
        raise ValueError("--batch-file path cannot be empty")

    candidate = (
        expanded_path
        if os.path.isabs(expanded_path)
        else os.path.join(os.getcwd(), expanded_path)
    )
    manifest_path = os.path.realpath(candidate)
    if not os.path.isfile(manifest_path):
        raise ValueError(f"batch manifest does not exist or is not a file: {raw_path}")

    with open(manifest_path, "rb") as manifest_file:
        payload = manifest_file.read(_BATCH_FILE_MAX_BYTES + 1)
    if len(payload) > _BATCH_FILE_MAX_BYTES:
        raise ValueError("batch manifest exceeds the 1 MiB limit")

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("batch manifest must be UTF-8 encoded") from exc
    if not text.strip():
        raise ValueError("batch manifest must contain at least one task")

    if text.lstrip().startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON array in batch manifest: {exc.msg}"
            ) from exc
        if not isinstance(parsed, list):
            raise ValueError("batch manifest JSON must be an array")
        entries = parsed
    else:
        entries = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL in batch manifest at line {line_number}: {exc.msg}"
                ) from exc

    if not entries:
        raise ValueError("batch manifest must contain at least one task")
    if len(entries) > _BATCH_TASK_MAX_COUNT:
        raise ValueError(
            f"batch manifest contains {len(entries)} tasks; "
            f"the limit is {_BATCH_TASK_MAX_COUNT}"
        )

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"batch task {index} must be a JSON object")
    return manifest_path, entries


def _validate_batch_entry_fields(
    entry: dict[str, Any],
    *,
    index: int,
    allowed_fields: set[str],
) -> None:
    unknown_fields = sorted(set(entry) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"batch task {index} contains unknown VideoParams fields: "
            f"{', '.join(unknown_fields)}"
        )

    materials = entry.get("video_materials")
    if not isinstance(materials, list):
        return
    allowed_material_fields = {"provider", "url", "duration", "source_info"}
    for material_index, material in enumerate(materials, start=1):
        if not isinstance(material, dict):
            continue
        unknown_material_fields = sorted(set(material) - allowed_material_fields)
        if unknown_material_fields:
            raise ValueError(
                f"batch task {index} material {material_index} contains unknown "
                f"MaterialInfo fields: {', '.join(unknown_material_fields)}"
            )


def _manifest_relative_path(raw_path: str, manifest_directory: str) -> str:
    expanded_path = os.path.expanduser(raw_path.strip())
    if not expanded_path or os.path.isabs(expanded_path):
        return expanded_path
    return os.path.realpath(os.path.join(manifest_directory, expanded_path))


def _resolve_batch_entry_paths(
    params: VideoParams,
    *,
    override_fields: set[str],
    manifest_directory: str,
) -> None:
    if "custom_audio_file" in override_fields and params.custom_audio_file:
        params.custom_audio_file = _manifest_relative_path(
            params.custom_audio_file,
            manifest_directory,
        )

    if (
        "video_materials" in override_fields
        and params.video_source == "local"
        and params.video_materials
    ):
        for material in params.video_materials:
            material.url = _manifest_relative_path(
                material.url,
                manifest_directory,
            )


def _validate_batch_task_params(
    params: VideoParams,
    *,
    stop_at: str,
    custom_position_is_explicit: bool,
    seedance_charge_confirmed: bool,
) -> None:
    if not params.video_subject.strip() and not params.video_script.strip():
        raise ValueError("one of video_subject or video_script is required")

    if params.video_source not in {
        "pexels",
        "pixabay",
        "coverr",
        "volcengine_seedance",
        "local",
    }:
        raise ValueError(
            "video_source must be one of: pexels, pixabay, coverr, "
            "volcengine_seedance, local"
        )
    for field_name, value in (
        ("video_aspect", params.video_aspect),
        ("video_concat_mode", params.video_concat_mode),
    ):
        # These schema fields remain Optional for compatibility with historical
        # API payloads, but the video pipeline always dereferences their enum
        # values. A manifest's explicit null must fail before any batch task starts.
        if value is None:
            raise ValueError(f"{field_name} cannot be null")
    if params.video_source == "local" and stop_at == "terms":
        raise ValueError(
            "stop_at=terms has no effect with video_source=local"
        )
    if (
        params.video_source == "local"
        and stop_at in {"materials", "video"}
        and not params.video_materials
    ):
        raise ValueError(
            "video_materials is required with video_source=local when "
            "stop_at is materials or video"
        )
    if params.video_source != "local" and params.video_materials:
        raise ValueError("video_materials can only be used with video_source=local")
    if (
        params.video_source == "volcengine_seedance"
        and stop_at in {"materials", "video"}
        and not seedance_charge_confirmed
    ):
        raise ValueError(
            "--confirm-seedance-charge is required for Volcano Engine Seedance"
        )

    if stop_at == "subtitle" and not params.subtitle_enabled:
        raise ValueError("stop_at=subtitle cannot be combined with disabled subtitles")
    if params.subtitle_position not in {"top", "center", "bottom", "custom"}:
        raise ValueError(
            "subtitle_position must be one of: top, center, bottom, custom"
        )
    if custom_position_is_explicit and params.subtitle_position != "custom":
        raise ValueError("custom_position requires subtitle_position=custom")
    if not math.isfinite(params.custom_position) or not 0 <= params.custom_position <= 100:
        raise ValueError("custom_position must be a finite number between 0 and 100")
    if params.video_clip_speed is not None and (
        not math.isfinite(params.video_clip_speed)
        or not 0.5 <= params.video_clip_speed <= 2.0
    ):
        raise ValueError("video_clip_speed must be a finite number between 0.5 and 2.0")
    if params.text_background_color is False and params.rounded_subtitle_background:
        raise ValueError(
            "rounded_subtitle_background requires an enabled subtitle background"
        )

    color_values = {
        "text_fore_color": params.text_fore_color,
        "stroke_color": params.stroke_color,
        "text_background_color": (
            params.text_background_color
            if isinstance(params.text_background_color, str)
            else None
        ),
    }
    for name, value in color_values.items():
        if value is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            raise ValueError(f"{name} must use #RRGGBB format")

    numeric_constraints = (
        ("voice_volume", params.voice_volume, 0, False),
        ("voice_rate", params.voice_rate, 0, True),
        ("bgm_volume", params.bgm_volume, 0, False),
        ("stroke_width", params.stroke_width, 0, False),
    )
    for name, value, minimum, exclusive in numeric_constraints:
        if value is None:
            continue
        if not math.isfinite(value) or (
            value <= minimum if exclusive else value < minimum
        ):
            comparison = ">" if exclusive else ">="
            raise ValueError(f"{name} must be a finite number {comparison} {minimum}")

    for name, value in (
        ("n_threads", params.n_threads),
        ("font_size", params.font_size),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1")

    if params.bgm_type not in {"", "random", "custom", "sonilo"}:
        raise ValueError("bgm_type must be one of: none, random, custom, sonilo")
    if params.bgm_file and params.bgm_type != "custom":
        raise ValueError("bgm_file requires bgm_type=custom")
    if params.sonilo_bgm_prompt and params.bgm_type != "sonilo":
        raise ValueError("sonilo_bgm_prompt requires bgm_type=sonilo")


def _build_batch_tasks(args: argparse.Namespace) -> list[VideoParams]:
    from app.models.schema import VideoParams

    manifest_path, entries = _load_batch_manifest(args.batch_file)
    manifest_directory = os.path.dirname(manifest_path)
    base_params = build_video_params(args)
    allowed_fields = set(VideoParams.model_fields)
    base_values = {
        field_name: getattr(base_params, field_name) for field_name in allowed_fields
    }
    tasks: list[VideoParams] = []

    for index, entry in enumerate(entries, start=1):
        _validate_batch_entry_fields(
            entry,
            index=index,
            allowed_fields=allowed_fields,
        )
        try:
            params = VideoParams.model_validate(
                {**copy.deepcopy(base_values), **entry}
            )
            override_fields = set(entry)
            _resolve_batch_entry_paths(
                params,
                override_fields=override_fields,
                manifest_directory=manifest_directory,
            )
            _validate_batch_task_params(
                params,
                stop_at=args.stop_at,
                custom_position_is_explicit=(
                    args.custom_position is not None
                    or "custom_position" in override_fields
                ),
                seedance_charge_confirmed=args.confirm_seedance_charge,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid batch task {index}: {exc}") from exc
        tasks.append(params)

    validation_plans = []
    for index, params in enumerate(tasks, start=1):
        try:
            validation_plans.append(_validate_cli_files(params, stop_at=args.stop_at))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid batch task {index}: {exc}") from exc

    prepared_paths: dict[str, str] = {}
    created_paths: list[str] = []
    try:
        for index, (local_videos_dir, resolved_materials) in enumerate(
            validation_plans, start=1
        ):
            try:
                _prepare_cli_materials(
                    local_videos_dir,
                    resolved_materials,
                    prepared_paths=prepared_paths,
                    created_paths=created_paths,
                )
            except OSError as exc:
                raise ValueError(f"invalid batch task {index}: {exc}") from exc
    except Exception:
        _remove_cli_material_copies(created_paths)
        raise
    return tasks


def _resolve_cli_file(
    raw_path: str,
    *,
    description: str,
    fallback_dir: str | None = None,
) -> str:
    """
    Приводит файловые аргументы CLI к абсолютным путям относительно текущего
    рабочего каталога и проверяет их существование до старта задачи.

    Прежние версии всегда разрешали локальные материалы относительно
    ``storage/local_videos``. Ради совместимости с существующими скриптами
    относительный путь, не найденный в текущем каталоге, может откатиться к тому
    каталогу; абсолютные пути всегда разрешаются ровно так, как их ввёл
    пользователь.
    """
    expanded_path = os.path.expanduser(raw_path.strip())
    if not expanded_path:
        raise ValueError(f"{description} path cannot be empty")

    candidate = (
        expanded_path
        if os.path.isabs(expanded_path)
        else os.path.join(os.getcwd(), expanded_path)
    )
    resolved_path = os.path.realpath(candidate)
    if not os.path.isfile(resolved_path) and fallback_dir and not os.path.isabs(expanded_path):
        resolved_path = os.path.realpath(os.path.join(fallback_dir, expanded_path))

    if not os.path.isfile(resolved_path):
        raise ValueError(f"{description} file does not exist: {raw_path}")
    return resolved_path


def _path_is_within_directory(file_path: str, directory: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.realpath(directory), os.path.realpath(file_path)]
        ) == os.path.realpath(directory)
    except ValueError:
        # Для разных букв дисков в Windows commonpath не вычисляется — значит, файл заведомо вне целевого каталога.
        return False


def _resolve_managed_resource_file(
    raw_path: str,
    *,
    resource_dir: str,
    description: str,
) -> str:
    """Разрешает файл ресурсов проекта и следит, чтобы абсолютный путь оставался внутри соответствующего каталога ресурсов."""
    from app.utils import utils

    expanded_path = os.path.expanduser(raw_path.strip())
    candidates = (
        [expanded_path]
        if os.path.isabs(expanded_path)
        else [
            os.path.join(resource_dir, expanded_path),
            os.path.join(utils.root_dir(), expanded_path),
        ]
    )
    for candidate in candidates:
        resolved_path = os.path.realpath(candidate)
        if os.path.isfile(resolved_path) and _path_is_within_directory(
            resolved_path, resource_dir
        ):
            return resolved_path
    raise ValueError(
        f"{description} file must exist inside {resource_dir}: {raw_path}"
    )


def _validate_cli_files(
    params: VideoParams, stop_at: str
) -> tuple[str, list[tuple[MaterialInfo, str, str]]]:
    """
    Разрешает и проверяет файлы CLI без побочных эффектов, чтобы пакетная
    предпроверка не оставляла копии материалов.

    Пользовательское аудио, BGM и шрифты приводятся к пути или имени, пригодным
    для сервисного слоя; для локальных материалов разбираются только источник и
    расширение, а реальное копирование выполняет ``_prepare_cli_materials`` уже
    после того, как все элементы пакета прошли проверку.
    """
    from app.models import const
    from app.services import bgm as bgm_service
    from app.utils import utils

    # Проверка FFmpeg перенесена в общий конвейер задач в app/services/task.py
    # (task.start), где выполняется единой жёсткой проверкой: при неудаче задача
    # завершается ошибкой на этапе preflight, и run_cli() возвращает ненулевой код.
    # Повторять здесь неблокирующую проверку не нужно — её результат мог бы разойтись с решением конвейера.

    local_material_extensions = {
        *(f".{extension}" for extension in const.FILE_TYPE_VIDEOS),
        *(f".{extension}" for extension in const.FILE_TYPE_IMAGES),
        ".avi",
        ".flv",
    }

    if params.custom_audio_file:
        params.custom_audio_file = _resolve_cli_file(
            params.custom_audio_file,
            description="custom audio",
        )
        audio_extension = os.path.splitext(params.custom_audio_file)[1].lower()
        if audio_extension not in _CUSTOM_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(_CUSTOM_AUDIO_EXTENSIONS))
            raise ValueError(
                f"unsupported custom audio type {audio_extension or '<none>'}; "
                f"allowed extensions: {allowed}"
            )

    if params.bgm_type == "custom":
        if not bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume):
            # При нулевой громкости ниже по потоку любой BGM пропускается. Заодно очищаем
            # файловый аргумент, чтобы CLI не разрешал путь, не проверял существование и
            # формат файла, который всё равно не будет прочитан.
            params.bgm_file = ""
        elif not params.bgm_file:
            # Считать ли отсутствие файла ошибкой, решает общий переключатель BGM. Безусловно
            # отсекать это на этапе argparse нельзя: иначе ``custom + 0%`` разойдётся с поведением WebUI и сервисного слоя.
            raise ValueError("--bgm-file is required when --bgm-type is custom")
        else:
            try:
                # CLI, WebUI и сервис задач обязаны использовать одну и ту же границу файлов BGM.
                # Переиспользуем разрешение из сервисного слоя: оно поддерживает и каталог
                # пользовательских загрузок, и каталог встроенных композиций, а также
                # автоматически подхватывает новые аудиоформаты и правила безопасности путей — не приходится вести белый список в нескольких точках входа.
                params.bgm_file = bgm_service.resolve_bgm_file(params.bgm_file)
            except ValueError as exc:
                supported_extensions = ", ".join(
                    bgm_service.SUPPORTED_BGM_EXTENSIONS
                )
                raise ValueError(
                    "background music must be a supported audio file inside "
                    f"storage/bgm or resource/songs ({supported_extensions}): "
                    f"{params.bgm_file}"
                ) from exc

    if params.subtitle_enabled and params.font_name and stop_at == "video":
        font_path = _resolve_managed_resource_file(
            params.font_name,
            resource_dir=utils.font_dir(),
            description="subtitle font",
        )
        if not font_path.lower().endswith((".ttf", ".ttc")):
            raise ValueError("subtitle font must use the .ttf or .ttc extension")
        # Ниже по потоку путь собирается из имени файла внутри resource/fonts, поэтому оставляем чистое имя файла.
        params.font_name = os.path.basename(font_path)

    if params.video_source != "local" or stop_at not in {"materials", "video"}:
        return "", []

    local_videos_dir = utils.storage_dir("local_videos")
    resolved_materials: list[tuple[MaterialInfo, str, str]] = []
    for material in params.video_materials or []:
        source_path = _resolve_cli_file(
            material.url,
            description="local material",
            fallback_dir=local_videos_dir,
        )
        extension = os.path.splitext(source_path)[1].lower()
        if extension not in local_material_extensions:
            allowed = ", ".join(sorted(local_material_extensions))
            raise ValueError(
                f"unsupported local material type {extension or '<none>'}: "
                f"{material.url}; allowed extensions: {allowed}"
            )
        resolved_materials.append((material, source_path, extension))

    return local_videos_dir, resolved_materials


def _remove_cli_material_copies(created_paths: Sequence[str]) -> None:
    """Best-effort cleanup for managed copies created by one CLI preparation."""
    for file_path in reversed(created_paths):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError as exc:
            logger.warning(
                f"failed to remove prepared CLI material: path={file_path}, "
                f"error={exc}"
            )


def _prepare_cli_materials(
    local_videos_dir: str,
    resolved_materials: Sequence[tuple[MaterialInfo, str, str]],
    *,
    prepared_paths: dict[str, str] | None = None,
    created_paths: list[str] | None = None,
) -> None:
    """Copy validated local materials once and update every matching reference."""
    if not resolved_materials:
        return

    os.makedirs(local_videos_dir, exist_ok=True)
    if prepared_paths is None:
        prepared_paths = {}
    if created_paths is None:
        created_paths = []

    for material, source_path, extension in resolved_materials:
        prepared_path = prepared_paths.get(source_path)
        if prepared_path is None:
            if _path_is_within_directory(source_path, local_videos_dir):
                prepared_path = source_path
            else:
                prepared_path = os.path.join(
                    local_videos_dir,
                    f"cli-material-{uuid4().hex}{extension}",
                )
                try:
                    shutil.copy2(source_path, prepared_path)
                except OSError:
                    try:
                        if os.path.exists(prepared_path):
                            os.remove(prepared_path)
                    except OSError:
                        pass
                    raise
                created_paths.append(prepared_path)
                logger.info(
                    "copied CLI local material into managed storage: "
                    f"source={source_path}, target={prepared_path}"
                )
            prepared_paths[source_path] = prepared_path

        material.url = prepared_path


def prepare_cli_files(params: VideoParams, stop_at: str) -> None:
    """Validate and prepare files for one trusted local CLI task."""
    local_videos_dir, resolved_materials = _validate_cli_files(params, stop_at)
    created_paths: list[str] = []
    try:
        _prepare_cli_materials(
            local_videos_dir,
            resolved_materials,
            created_paths=created_paths,
        )
    except Exception:
        _remove_cli_material_copies(created_paths)
        raise


def _run_batch_tasks(args: argparse.Namespace, tasks: list[VideoParams]) -> int:
    from app.services import task as tm
    from app.utils import utils

    task_summaries: list[dict[str, Any]] = []
    used_task_ids: set[str] = set()
    failed_count = 0

    for index, params in enumerate(tasks, start=1):
        task_id = utils.get_uuid()
        if task_id in used_task_ids:
            task_id = str(uuid4())
        used_task_ids.add(task_id)
        logger.info(
            f"start CLI batch task: index={index}, task_id={task_id}, "
            f"stop_at={args.stop_at}"
        )

        result = None
        failed_stage = None
        error = None
        try:
            result = tm.start(
                task_id=task_id,
                params=params,
                stop_at=args.stop_at,
                allow_server_file_input=True,
            )
        except Exception as exc:
            failed_stage = "runtime"
            error = str(exc)
            logger.exception(
                "CLI batch task failed with an unexpected error: "
                f"index={index}, task_id={task_id}, error={exc}"
            )
        else:
            if not isinstance(result, dict) or not result:
                failed_stage = "unknown"
                error = "empty or invalid task result"
            elif result.get("state") == tm.const.TASK_STATE_FAILED:
                failed_stage = result.get("failed_stage") or "unknown"
                error = result.get("error") or "unknown task error"

            if error is not None:
                logger.error(
                    f"CLI batch task failed: index={index}, task_id={task_id}, "
                    f"stop_at={args.stop_at}, stage={failed_stage}, error={error}"
                )

        status = "failed" if error is not None else "succeeded"
        if status == "failed":
            failed_count += 1
        task_summaries.append(
            {
                "index": index,
                "task_id": task_id,
                "status": status,
                "result": result,
                "failed_stage": failed_stage,
                "error": error,
            }
        )

    print(
        json.dumps(
            {
                "total": len(task_summaries),
                "succeeded": len(task_summaries) - failed_count,
                "failed": failed_count,
                "tasks": task_summaries,
            },
            ensure_ascii=False,
        )
    )
    return 1 if failed_count else 0


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_file:
        try:
            tasks = _build_batch_tasks(args)
        except (ValueError, OSError) as exc:
            logger.error(f"invalid CLI batch input: {exc}")
            return 2
        return _run_batch_tasks(args, tasks)

    try:
        params = build_video_params(args)
        prepare_cli_files(params, stop_at=args.stop_at)
    except (ValueError, OSError) as exc:
        logger.error(f"invalid CLI input: {exc}")
        return 2

    # Аргумент справки завершает работу прямо в parse_args. Импорт бизнес-сервисов
    # отложен сюда, чтобы вывод -h/--help оставался чистым и при этом инициализация реальной задачи не менялась.
    from app.services import task as tm
    from app.utils import utils

    task_id = args.task_id or utils.get_uuid()
    logger.info(f"start CLI task: task_id={task_id}, stop_at={args.stop_at}")
    try:
        result = tm.start(
            task_id=task_id,
            params=params,
            stop_at=args.stop_at,
            # CLI inputs come from the local operator rather than an HTTP client.
            # Preserve support for arbitrary local audio paths without weakening the
            # task service's secure default for API and WebUI callers.
            allow_server_file_input=True,
        )
    except Exception as exc:
        logger.exception(
            f"CLI task failed with an unexpected error: task_id={task_id}, error={exc}"
        )
        return 1
    if not result or result.get("state") == tm.const.TASK_STATE_FAILED:
        failed_stage = result.get("failed_stage", "unknown") if result else "unknown"
        error = result.get("error", "unknown task error") if result else "empty result"
        logger.error(
            f"CLI task failed: task_id={task_id}, stop_at={args.stop_at}, "
            f"stage={failed_stage}, error={error}"
        )
        return 1

    print(json.dumps({"task_id": task_id, "result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
