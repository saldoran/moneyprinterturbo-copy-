"""Дисковый кэш результатов онлайн-поиска материалов."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils


MATERIAL_SEARCH_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_FORMAT_VERSION = 2
_CACHE_CLEANUP_INTERVAL_SECONDS = 60 * 60
_CACHE_FILE_PATTERN = re.compile(r"^[0-9a-f]{64}\.json$")

# API по умолчанию допускает параллельное выполнение нескольких задач видео.
# Фиксированное число шардов лока позволяет одинаковым условиям поиска делить
# один лок и при этом не хранить Lock под каждое ключевое слово вечно, наращивая
# память. Шарды лишь объединяют параллельные запросы внутри текущего процесса;
# целостность межпроцессной записи по-прежнему обеспечивают временный файл и os.replace.
_CACHE_LOCKS = tuple(threading.Lock() for _ in range(256))
_cleanup_state_lock = threading.Lock()
_last_cleanup_monotonic: float | None = None


def _safe_public_url(value) -> str | None:
    """Убирает из URL публичной страницы параметры запроса и учётные данные, чтобы в кэш случайно не попал токен."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _cached_source_info(item: MaterialInfo) -> dict | None:
    """
    Собирает сведения об источнике для записи на диск по белому списку.

    Поисковый запрос уже входит в ключ кэша и в содержимое открытым текстом
    больше не пишется — при чтении он восстанавливается из параметров вызова.
    URL для скачивания хранится отдельно в ``MaterialInfo.url``. Здесь
    допускаются только публичная страница материала, публичная страница автора и
    устойчивые бизнес-идентификаторы, чтобы произвольные дополнительные поля не
    попали в дисковый кэш.
    """
    source = item.source_info
    if not isinstance(source, dict) or not source:
        return None

    cached: dict = {
        "provider": str(source.get("provider") or item.provider),
    }
    asset_id = source.get("asset_id")
    source_page = _safe_public_url(source.get("source_page"))
    if asset_id not in (None, ""):
        cached["asset_id"] = str(asset_id)
    if source_page:
        cached["source_page"] = source_page

    raw_creator = source.get("creator")
    if isinstance(raw_creator, dict):
        creator = {}
        creator_id = raw_creator.get("id")
        creator_name = raw_creator.get("name")
        creator_page = _safe_public_url(raw_creator.get("profile_page"))
        if creator_id not in (None, ""):
            creator["id"] = str(creator_id)
        if creator_name not in (None, ""):
            creator["name"] = str(creator_name)
        if creator_page:
            creator["profile_page"] = creator_page
        if creator:
            cached["creator"] = creator

    raw_rendition = source.get("rendition")
    if isinstance(raw_rendition, dict):
        rendition = {}
        for field in ("id", "width", "height"):
            value = raw_rendition.get(field)
            if value not in (None, ""):
                rendition[field] = str(value) if field == "id" else value
        if rendition:
            cached["rendition"] = rendition
    return cached


def _cache_dir() -> Path:
    """
    Возвращает каталог кэша поиска материалов, общий для всех точек запуска.

    Кэш обязан лежать под ``storage``, а не в сессии WebUI и не в памяти
    процесса: только тогда WebUI, API, CLI и задачи после перезапуска Docker
    переиспользуют один и тот же результат.
    """
    return Path(utils.storage_dir("cache_material_search", create=True))


def _cache_key(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> str:
    """
    Формирует устойчивое имя файла по тем бизнес-параметрам, которые влияют на
    результат поиска.

    API-ключ отвечает только за аутентификацию и на публичные результаты поиска
    не влияет, поэтому в ключ и содержимое кэша он попадать не должен. SHA-256 не
    даёт ключевым словам оказаться прямо в имени файла и удерживает длину пути
    постоянной.
    """
    aspect_value = getattr(video_aspect, "value", video_aspect)
    cache_key = json.dumps(
        {
            "provider": str(provider).strip().lower(),
            "search_term": str(search_term).strip(),
            "minimum_duration": int(minimum_duration),
            "video_aspect": str(aspect_value),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()


def _cache_path(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> Path:
    digest = _cache_key(
        provider=provider,
        search_term=search_term,
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
    )
    return _cache_dir() / f"{digest}.json"


def get_material_search_cache_lock(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
) -> threading.Lock:
    """Возвращает внутрипроцессный шард лока, соответствующий текущим условиям поиска."""
    digest = _cache_key(
        provider=provider,
        search_term=search_term,
        minimum_duration=minimum_duration,
        video_aspect=video_aspect,
    )
    return _CACHE_LOCKS[int(digest[:8], 16) % len(_CACHE_LOCKS)]


def _remove_invalid_cache(cache_path: Path) -> None:
    """Удаляет один просроченный или неразбираемый файл кэша; неудача не влияет на основной поток поиска материалов."""
    try:
        cache_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            f"failed to remove invalid material search cache: "
            f"file={cache_path.name}, error={exc}"
        )


def load_material_search_cache(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
    *,
    now: float | None = None,
) -> list[MaterialInfo] | None:
    """
    Читает результаты поиска материалов, если они ещё укладываются в 24-часовой
    срок годности.

    ``None`` означает промах кэша и необходимость запроса к удалённому API.
    Пустой список валидным кэшем не считается: иначе сетевая ошибка или сбой на
    стороне поставщика закэшировались бы и надолго блокировали последующие
    задачи.
    """
    if str(provider).strip().lower() == "coverr":
        # Адрес скачивания у Coverr содержит подписанный JWT, привязанный к API-ключу.
        # Он нужен только текущему запросу и в дисковый кэш попадать не должен; заодно при том же запросе удаляем кэш, который могла оставить прежняя версия.
        try:
            _remove_invalid_cache(
                _cache_path(
                    provider=provider,
                    search_term=search_term,
                    minimum_duration=minimum_duration,
                    video_aspect=video_aspect,
                )
            )
        except Exception as exc:
            logger.warning(
                "failed to remove disabled Coverr material search cache: "
                f"error={type(exc).__name__}, detail={exc}"
            )
        return None

    try:
        cache_path = _cache_path(
            provider=provider,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
    except Exception as exc:
        # Ошибки создания каталога кэша, разбора пути и подобные не должны прерывать
        # удалённый поиск материалов. Сохраняем полный тип и текст исключения для разбора проблем с правами или монтированием и продолжаем основной поток как при промахе кэша.
        logger.warning(
            "failed to prepare material search cache: "
            f"operation=read, error={type(exc).__name__}, detail={exc}"
        )
        return None
    try:
        stat_result = cache_path.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning(
            f"failed to inspect material search cache: "
            f"file={cache_path.name}, error={exc}"
        )
        return None

    current_time = time.time() if now is None else now
    cache_age = current_time - stat_result.st_mtime
    # После перевода системных часов назад или копирования файла с другой машины
    # mtime может оказаться в будущем. Считать такой кэш свежим надолго нельзя — надёжнее сразу признать его недействительным и запросить удалённый сервис заново.
    if cache_age < 0 or cache_age >= MATERIAL_SEARCH_CACHE_TTL_SECONDS:
        _remove_invalid_cache(cache_path)
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)

        if (
            not isinstance(payload, dict)
            or payload.get("version") != _CACHE_FORMAT_VERSION
            or not isinstance(payload.get("items"), list)
            or not payload["items"]
        ):
            raise ValueError("invalid cache payload")

        items = []
        for raw_item in payload["items"]:
            if not isinstance(raw_item, dict):
                raise ValueError("invalid material item")
            item_provider = raw_item.get("provider")
            item_url = raw_item.get("url")
            item_duration = raw_item.get("duration")
            source_info = raw_item.get("source_info")
            if (
                not isinstance(item_provider, str)
                or not item_provider
                or not isinstance(item_url, str)
                or not item_url
                or isinstance(item_duration, bool)
                or not isinstance(item_duration, (int, float))
                or item_duration <= 0
                or not isinstance(source_info, dict)
                or not source_info
            ):
                raise ValueError("invalid material fields")
            source_info = dict(source_info)
            source_info["search_term"] = search_term
            items.append(
                MaterialInfo(
                    provider=item_provider,
                    url=item_url,
                    duration=int(item_duration),
                    source_info=source_info,
                )
            )
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(
            f"failed to load material search cache: file={cache_path.name}, error={exc}"
        )
        _remove_invalid_cache(cache_path)
        return None

    logger.info(
        f"material search cache hit: provider={provider}, "
        f"term={search_term!r}, items={len(items)}"
    )
    return items


def save_material_search_cache(
    provider: str,
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect | str,
    items: Iterable[MaterialInfo],
) -> bool:
    """
    Атомарно сохраняет один успешный непустой результат поиска материалов.

    Несколько задач могут искать по одинаковым ключевым словам одновременно.
    Сначала пишется уникальный временный файл в том же каталоге, затем
    ``os.replace`` публикует его — читающий процесс увидит либо целиком старый
    файл, либо целиком новый. Даже если два пишущих процесса завершатся
    одновременно, итоговым содержимым останется корректный результат для того же
    ключа кэша.
    """
    if str(provider).strip().lower() == "coverr":
        return False

    temp_path = None
    try:
        serialized_items = []
        for item in items:
            source_info = _cached_source_info(item)
            if not item.url or item.duration <= 0 or not source_info:
                continue
            serialized_items.append(
                {
                    "provider": item.provider,
                    "url": item.url,
                    "duration": int(item.duration),
                    "source_info": source_info,
                }
            )
        if not serialized_items:
            return False

        cache_path = _cache_path(
            provider=provider,
            search_term=search_term,
            minimum_duration=minimum_duration,
            video_aspect=video_aspect,
        )
        cleanup_expired_material_search_cache()
        payload = {
            "version": _CACHE_FORMAT_VERSION,
            "items": serialized_items,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, cache_path)
        return True
    except Exception as exc:
        logger.warning(
            "failed to save material search cache: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def cleanup_expired_material_search_cache(
    *,
    now: float | None = None,
    force: bool = False,
) -> int:
    """
    Нечастая уборка просроченного кэша поиска, к которому больше не обращались.

    Обычный путь записи сканирует каталог не чаще раза в час, чтобы каждый поиск
    не вызывал линейный обход. ``force`` предназначен только для тестов и явного
    обслуживания. Удаляются лишь JSON-файлы с именами вида SHA-256 — другие
    файлы, положенные пользователем в каталог, не затрагиваются.
    """
    global _last_cleanup_monotonic

    monotonic_now = time.monotonic()
    with _cleanup_state_lock:
        if (
            not force
            and _last_cleanup_monotonic is not None
            and monotonic_now - _last_cleanup_monotonic
            < _CACHE_CLEANUP_INTERVAL_SECONDS
        ):
            return 0
        _last_cleanup_monotonic = monotonic_now

    try:
        cache_dir = _cache_dir()
        entries = os.scandir(cache_dir)
    except Exception as exc:
        logger.warning(
            "failed to scan material search cache: "
            f"error={type(exc).__name__}, detail={exc}"
        )
        return 0

    current_time = time.time() if now is None else now
    deleted_count = 0
    failed_count = 0
    with entries:
        for entry in entries:
            if not _CACHE_FILE_PATTERN.fullmatch(entry.name):
                continue
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                cache_age = current_time - entry.stat(follow_symlinks=False).st_mtime
                if 0 <= cache_age < MATERIAL_SEARCH_CACHE_TTL_SECONDS:
                    continue
                os.unlink(entry.path)
                deleted_count += 1
            except OSError as exc:
                failed_count += 1
                logger.warning(
                    "failed to delete material search cache file: "
                    f"file={entry.name}, error={exc}"
                )

    if deleted_count or failed_count:
        logger.info(
            "finished cleaning material search cache: "
            f"deleted={deleted_count}, failed={failed_count}"
        )
    return deleted_count
