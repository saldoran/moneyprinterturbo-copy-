import math
import os
import time
from typing import Any, Mapping
from urllib.parse import quote_plus

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL_ID = "doubao-seedance-1-0-pro-250528"
DEFAULT_RESOLUTION = "1080p"
DEFAULT_MIN_DURATION_SECONDS = 2
DEFAULT_MAX_DURATION_SECONDS = 12
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1800.0
MAX_POLL_RETRIES = 5
RETRY_BASE_SECONDS = 1.0
MAX_ERROR_TEXT_LENGTH = 500
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "expired"}
)
ACTIVE_STATUSES = frozenset({"queued", "running"})
SUPPORTED_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})


class VolcEngineSeedanceError(RuntimeError):
    """Детерминированная ошибка конфигурации, запроса или ответа."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        # Как только удалённая задача создана, ID задачи несут все типы ошибок. Верхнему
        # слою не нужна отдельная логика восстановления под каждый подкласс исключения, а WebUI и API стабильно показывают данные для разбора.
        self.task_id = task_id


class VolcEngineSeedanceUnconfirmedTaskError(VolcEngineSeedanceError):
    """Платная задача, возможно, создана на удалённой стороне, но локально её итоговый статус подтвердить не удалось."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message, task_id=task_id)


class VolcEngineSeedanceDownloadError(VolcEngineSeedanceError):
    """Платная удалённая задача завершилась успешно, но сгенерированное видео не удалось скачать локально."""

    def __init__(self, message: str, task_id: str):
        super().__init__(message, task_id=task_id)


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    """
    Читает учётные данные Ark по явному и единственному приоритету.

    Наивысший приоритет — у отдельной настройки Seedance. Единственная
    поддерживаемая переменная окружения — однозначная по смыслу
    ``VOLCENGINE_ARK_API_KEY``. Историческая ``volcengine_api_key`` остаётся лишь
    запасным вариантом из общей конфигурации, чтобы пользователям, уже
    подключившим модели Ark, не пришлось после обновления вводить тот же ключ заново.
    """
    settings = config.app if settings is None else settings
    configured = str(settings.get("volcengine_seedance_api_key", "") or "").strip()
    environment_key = os.getenv("VOLCENGINE_ARK_API_KEY", "").strip()
    shared_ark_key = str(settings.get("volcengine_api_key", "") or "").strip()
    return configured or environment_key or shared_ark_key


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    return bool(get_api_key(settings))


def _base_url() -> str:
    return str(
        config.app.get("volcengine_seedance_base_url", DEFAULT_BASE_URL)
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    return str(
        config.app.get("volcengine_seedance_model", DEFAULT_MODEL_ID)
        or DEFAULT_MODEL_ID
    ).strip()


def _resolution() -> str:
    configured = config.app.get("volcengine_seedance_resolution", DEFAULT_RESOLUTION)
    value = str(configured).strip().lower()
    if value not in SUPPORTED_RESOLUTIONS:
        # Разрешение напрямую задаёт тариф платной задачи. Некорректное значение нельзя
        # молча откатывать к максимальному по умолчанию. Если параметра нет, значение
        # по умолчанию подставит get; но если пользователь явно записал пустую строку,
        # None, 0 или другое недопустимое значение, нужна ошибка — иначе счёт окажется больше ожидаемого.
        supported = ", ".join(sorted(SUPPORTED_RESOLUTIONS))
        raise VolcEngineSeedanceError(
            f"Unsupported Seedance resolution {value!r}; expected one of: {supported}"
        )
    return value


def _config_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _bounded_float(key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return min(max(value, minimum), maximum)


def _duration_bounds() -> tuple[int, int]:
    def read(key: str, default: int) -> int:
        try:
            value = int(config.app.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value >= 1 else default

    minimum = read(
        "volcengine_seedance_min_duration", DEFAULT_MIN_DURATION_SECONDS
    )
    maximum = read(
        "volcengine_seedance_max_duration", DEFAULT_MAX_DURATION_SECONDS
    )
    return minimum, max(minimum, maximum)


def _tls_verify() -> bool:
    return _config_bool("tls_verify", True)


def _status_code(response: Any) -> int:
    try:
        return int(getattr(response, "status_code", 200))
    except (TypeError, ValueError):
        return 200


def _redact_secret(value: Any, secret: str) -> str:
    text = str(value or "")
    if secret:
        text = text.replace(secret, "***")
        encoded = quote_plus(secret)
        if encoded != secret:
            text = text.replace(encoded, "***")
    for proxy_url in config.proxy.values():
        proxy_secret = str(proxy_url or "")
        if proxy_secret:
            text = text.replace(proxy_secret, "***")
    return text[:MAX_ERROR_TEXT_LENGTH]


def _response_error(response: Any, api_key: str) -> str:
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {_status_code(response)}"
    if not isinstance(payload, dict):
        return f"HTTP {_status_code(response)}"
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or payload.get("code")
        message = error.get("message") or payload.get("message")
    else:
        code = payload.get("code")
        message = payload.get("message") or error
    detail = ": ".join(str(item) for item in (code, message) if item not in (None, ""))
    return _redact_secret(detail or f"HTTP {_status_code(response)}", api_key)


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    response = getattr(error, "response", None)
    return response is not None and _status_code(response) in RETRYABLE_STATUS_CODES


def _rendition_size(aspect: VideoAspect, resolution: str) -> tuple[int, int]:
    # У Ark длинная сторона видео 480p из-за выравнивания кодека фактически равна
    # 864, а не 854, как даёт арифметика; 720p и 1080p по официальным пропорциям
    # дают 1280 и 1920. Реальный вызов подтвердил, что вертикальный ролик 480p
    # имеет размер 480x864. Запись об источнике обязана описывать фактический
    # результат, иначе аудит или диагностика материалов увидят размеры, не совпадающие с файлом.
    short_edge = {"480p": 480, "720p": 720, "1080p": 1080}[resolution]
    long_edge = {"480p": 864, "720p": 1280, "1080p": 1920}[resolution]
    if aspect == VideoAspect.portrait:
        return short_edge, long_edge
    if aspect == VideoAspect.square:
        return short_edge, short_edge
    return long_edge, short_edge


def generate_videos(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list[MaterialInfo]:
    """Отправляет задачу Ark Seedance «текст в видео» и дожидается адреса результата, доступного для скачивания."""
    api_key = get_api_key()
    if not api_key:
        raise VolcEngineSeedanceError(
            "Volcano Engine Seedance requires an Ark API key"
        )

    term = str(search_term or "").strip()
    if not term:
        # Пустой промпт может прийти из-за сбоя разбиения сценария выше по потоку.
        # Платный генератор не должен отправлять его на удалённую сторону: даже если эндпоинт примет запрос, получится непригодное, но уже оплаченное видео.
        raise VolcEngineSeedanceError("Seedance search term must not be empty")

    aspect = VideoAspect(video_aspect)
    requested_duration = max(int(minimum_duration), 1)
    minimum, maximum = _duration_bounds()
    duration = min(max(requested_duration, minimum), maximum)
    resolution = _resolution()
    payload = {
        "model": _model_id(),
        "content": [{"type": "text", "text": term}],
        "ratio": aspect.value,
        "duration": duration,
        "resolution": resolution,
        "watermark": _config_bool("volcengine_seedance_watermark", False),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    tasks_url = f"{_base_url()}/contents/generations/tasks"
    logger.info(
        "generating video with Volcano Engine Seedance: "
        f"model={payload['model']}, term={term!r}, duration={duration}s"
    )

    # Эндпоинт отправки не повторяет запрос автоматически: таймаут или 5xx могут
    # случиться уже после создания платной задачи, и слепой повтор приведёт к двойному списанию. Детерминированным отказом считается только явный отказ в ответе.
    try:
        response = requests.post(
            tasks_url,
            json=payload,
            headers=headers,
            proxies=config.proxy,
            verify=_tls_verify(),
            timeout=(30, 60),
        )
    except Exception as exc:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            "Seedance submission returned no response; a paid task may already "
            "exist remotely: "
            f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}"
        ) from exc

    status_code = _status_code(response)
    if status_code >= 500:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            f"Seedance submission failed with HTTP {status_code}; a paid task may "
            "already exist remotely"
        )
    if not 200 <= status_code < 300:
        raise VolcEngineSeedanceError(
            "Seedance video generation request rejected: "
            f"HTTP {status_code}, {_response_error(response, api_key)}"
        )
    try:
        body = response.json()
    except Exception as exc:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            "Seedance submission returned an unreadable response; a paid task may "
            f"already exist remotely: error={type(exc).__name__}"
        ) from exc
    task_id = str(body.get("id") or "").strip() if isinstance(body, dict) else ""
    if not task_id:
        raise VolcEngineSeedanceUnconfirmedTaskError(
            "Seedance accepted the submission without returning a task id"
        )
    logger.info(f"Volcano Engine Seedance task created: id={task_id}")

    task = _wait_for_task(
        task_id=task_id,
        tasks_url=tasks_url,
        headers=headers,
        api_key=api_key,
    )
    if task is None:
        return []
    content = task.get("content")
    video_url = content.get("video_url") if isinstance(content, dict) else None
    if not isinstance(video_url, str) or not video_url.startswith(("http://", "https://")):
        raise VolcEngineSeedanceError(
            f"Seedance task succeeded without a downloadable video: id={task_id}",
            task_id=task_id,
        )

    width, height = _rendition_size(aspect, resolution)
    return [
        MaterialInfo(
            provider="volcengine_seedance",
            url=video_url,
            duration=duration,
            source_info={
                "provider": "volcengine_seedance",
                "search_term": term,
                "asset_id": task_id,
                "rendition": {
                    "id": task_id,
                    "width": width,
                    "height": height,
                },
            },
        )
    ]


def _wait_for_task(
    *,
    task_id: str,
    tasks_url: str,
    headers: dict[str, str],
    api_key: str,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + _bounded_float(
        "volcengine_seedance_run_timeout",
        DEFAULT_RUN_TIMEOUT_SECONDS,
        60.0,
        7200.0,
    )
    poll_interval = _bounded_float(
        "volcengine_seedance_poll_interval",
        DEFAULT_POLL_INTERVAL_SECONDS,
        0.5,
        60.0,
    )
    consecutive_failures = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VolcEngineSeedanceUnconfirmedTaskError(
                "Seedance task is still running after the configured local wait "
                f"timeout: id={task_id}",
                task_id=task_id,
            )

        # У requests таймауты connect и read отсчитываются раздельно, поэтому каждому
        # отдаётся половина оставшегося времени. Даже если оба упрутся в предел, один
        # круг запроса заведомо не выйдет за общий дедлайн; небольшая погрешность
        # планирования в сетевой библиотеке возможна, но следующая проверка дедлайна не даст повторить попытку.
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = requests.get(
                f"{tasks_url}/{task_id}",
                headers=headers,
                proxies=config.proxy,
                verify=_tls_verify(),
                timeout=(phase_timeout, phase_timeout),
            )
            status_code = _status_code(response)
            if status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(
                    f"HTTP {status_code}", response=response
                )
            if not 200 <= status_code < 300:
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "Seedance task status is unknown: "
                    f"http_status={status_code}, detail={_response_error(response, api_key)}",
                    task_id=task_id,
                )
            body = response.json()
            if not isinstance(body, dict):
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "Seedance task status response is malformed", task_id=task_id
                )
        except VolcEngineSeedanceUnconfirmedTaskError:
            raise
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "Seedance polling failed and the paid task state is unknown: "
                    f"error={type(exc).__name__}, detail={_redact_secret(exc, api_key)}",
                    task_id=task_id,
                ) from exc

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "Seedance task is still running after the configured local wait "
                    f"timeout: id={task_id}",
                    task_id=task_id,
                ) from exc
            consecutive_failures += 1
            if consecutive_failures > MAX_POLL_RETRIES:
                raise VolcEngineSeedanceUnconfirmedTaskError(
                    "Seedance polling failed after retries; the paid task may still "
                    f"be running remotely: id={task_id}",
                    task_id=task_id,
                ) from exc
            delay = min(RETRY_BASE_SECONDS * consecutive_failures, remaining)
            logger.warning(
                "Seedance polling hit a transient error; retrying the same task: "
                f"id={task_id}, attempt={consecutive_failures}/{MAX_POLL_RETRIES}, "
                f"retry_in={delay:.1f}s"
            )
            time.sleep(delay)
            continue

        consecutive_failures = 0
        status = str(body.get("status") or "").strip().lower()
        if status == "succeeded":
            return body
        if status in TERMINAL_FAILURE_STATUSES:
            raise VolcEngineSeedanceError(
                "Seedance task did not produce a video: "
                f"id={task_id}, status={status}, "
                f"detail={_redact_secret(body.get('error'), api_key)}",
                task_id=task_id,
            )
        if status not in ACTIVE_STATUSES:
            raise VolcEngineSeedanceUnconfirmedTaskError(
                f"Seedance returned an unknown task status: id={task_id}, status={status!r}",
                task_id=task_id,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VolcEngineSeedanceUnconfirmedTaskError(
                "Seedance task is still running after the configured local wait "
                f"timeout: id={task_id}",
                task_id=task_id,
            )
        time.sleep(min(poll_interval, remaining))
