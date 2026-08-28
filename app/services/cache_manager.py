"""Сервис подсчёта, предпросмотра и очистки кэша видеоматериалов."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from app.utils import utils


# Онлайн-материалы получают устойчивое имя файла из MD5 их URL. Управление кэшем
# принимает только этот формат имени, чтобы не удалить как кэш видео, заметки
# или другие рабочие файлы, случайно положенные пользователем в тот же каталог.
_VIDEO_CACHE_FILE_PATTERN = re.compile(r"^vid-[0-9a-f]{32}\.mp4$")
_SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class VideoCacheStats:
    """Лёгкая статистика по каталогу кэша: только метаданные файловой системы."""

    file_count: int = 0
    total_size: int = 0
    oldest_mtime: float | None = None
    newest_mtime: float | None = None


@dataclass(frozen=True)
class VideoCacheCleanupResult:
    """Результат одной очистки; часть файлов может не удалиться."""

    deleted_count: int = 0
    deleted_size: int = 0
    failed_count: int = 0


@dataclass(frozen=True)
class _VideoCacheEntry:
    """Минимальные сведения о файле, сохранённые при сканировании, чтобы при очистке не открывать и не разбирать видео."""

    path: str
    name: str
    size: int
    mtime: float


def video_cache_dir() -> str:
    """Возвращает каталог кэша видео по умолчанию, которым управляет проект."""

    return os.path.realpath(utils.storage_dir("cache_videos"))


def _iter_video_cache_entries() -> Iterator[_VideoCacheEntry]:
    """
    Последовательно сканирует первый уровень каталога кэша по умолчанию.

    ``os.scandir`` выбран, чтобы при десятках тысяч файлов переиспользовать
    метаданные, которые возвращает обход каталога, и не запрашивать тип файла
    повторно после ``Path.iterdir``. Рекурсии нет, видео не открываются, FFmpeg
    не вызывается, поэтому время работы линейно зависит от числа файлов, а не от
    суммарного объёма видео.
    """

    cache_dir = video_cache_dir()
    try:
        entries = os.scandir(cache_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            f"failed to scan video cache directory: path={cache_dir}, error={exc}"
        )
        return

    with entries:
        for entry in entries:
            if not _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name):
                continue

            try:
                # Не идём по симлинкам: логика очистки не должна выходить за границы каталога кэша по умолчанию.
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat_result = entry.stat(follow_symlinks=False)
            except OSError as exc:
                logger.warning(
                    f"failed to inspect video cache file: file={entry.name}, error={exc}"
                )
                continue

            yield _VideoCacheEntry(
                path=entry.path,
                name=entry.name,
                size=stat_result.st_size,
                mtime=stat_result.st_mtime,
            )


def _is_cleanup_candidate(
    entry: _VideoCacheEntry,
    max_age_days: int | None,
    now: float,
) -> bool:
    if max_age_days is None:
        return True
    return entry.mtime < now - max_age_days * _SECONDS_PER_DAY


def _validate_max_age_days(max_age_days: int | None) -> None:
    """Даже при пустом каталоге кэша некорректные параметры очистки отклоняются предсказуемо."""
    if max_age_days is None:
        return
    if (
        isinstance(max_age_days, bool)
        or not isinstance(max_age_days, int)
        or max_age_days <= 0
    ):
        raise ValueError("max_age_days must be a positive integer or None")


def get_video_cache_stats(max_age_days: int | None = None) -> VideoCacheStats:
    """
    Считает весь кэш или показывает предпросмотр записей старше указанного числа
    дней.

    ``max_age_days=None`` означает весь кэш. При подсчёте читаются только размер
    и время изменения элементов каталога, содержимое видео не открывается,
    поэтому даже большой кэш не даёт I/O, пропорционального объёму.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    file_count = 0
    total_size = 0
    oldest_mtime = None
    newest_mtime = None

    for entry in _iter_video_cache_entries():
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        file_count += 1
        total_size += entry.size
        oldest_mtime = (
            entry.mtime if oldest_mtime is None else min(oldest_mtime, entry.mtime)
        )
        newest_mtime = (
            entry.mtime if newest_mtime is None else max(newest_mtime, entry.mtime)
        )

    return VideoCacheStats(
        file_count=file_count,
        total_size=total_size,
        oldest_mtime=oldest_mtime,
        newest_mtime=newest_mtime,
    )


def clean_video_cache(max_age_days: int | None = None) -> VideoCacheCleanupResult:
    """
    Очищает кэш видео по умолчанию и возвращает сводку, пригодную для показа
    пользователю.

    Между предпросмотром на странице и реальным нажатием «Очистить» может пройти
    много времени, поэтому при выполнении список кандидатов пересобирается
    заново, а старый не переиспользуется. Удаление устойчиво к ошибкам по
    каждому файлу: если файл занят или не хватает прав, пишется предупреждение и
    работа продолжается — один проблемный файл из сотен не должен срывать всю
    очистку.
    """

    _validate_max_age_days(max_age_days)
    now = time.time()
    logger.info(
        f"start cleaning video cache: max_age_days={max_age_days}"
    )

    candidate_count = 0
    candidate_size = 0
    deleted_count = 0
    deleted_size = 0
    failed_count = 0
    cache_dir = video_cache_dir()

    # Удаляем по ходу сканирования, не держа в памяти полный список кандидатов. Даже
    # при сотнях тысяч файлов дополнительная память остаётся константной. Используем
    # единое значение now, чтобы за время долгой очистки граница отсечения не поехала и набор кандидатов не стал непредсказуемым.
    for entry in _iter_video_cache_entries():
        if not _is_cleanup_candidate(entry, max_age_days, now):
            continue
        candidate_count += 1
        candidate_size += entry.size
        try:
            # entry.path приходит из scandir по первому уровню каталога по умолчанию. Перед
            # удалением ещё раз проверяем родительский каталог и имя файла, чтобы будущая правка логики сканирования случайно не расширила круг удаляемого.
            if (
                os.path.realpath(os.path.dirname(entry.path)) != cache_dir
                or not _VIDEO_CACHE_FILE_PATTERN.fullmatch(entry.name)
                or os.path.islink(entry.path)
            ):
                raise ValueError("cache file is outside the managed directory")
            os.unlink(entry.path)
            deleted_count += 1
            deleted_size += entry.size
        except (OSError, ValueError) as exc:
            failed_count += 1
            logger.warning(
                f"failed to delete video cache file: file={entry.name}, error={exc}"
            )

    logger.info(
        "finished cleaning video cache: "
        f"candidates={candidate_count}, candidate_bytes={candidate_size}, "
        f"deleted={deleted_count}, deleted_bytes={deleted_size}, failed={failed_count}"
    )
    return VideoCacheCleanupResult(
        deleted_count=deleted_count,
        deleted_size=deleted_size,
        failed_count=failed_count,
    )
