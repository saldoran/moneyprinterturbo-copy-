"""Проверяет, доступна ли новая стабильная версия MoneyPrinterTurbo."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import requests
from loguru import logger
from packaging.version import InvalidVersion, Version


LATEST_RELEASE_API_URL: Final = (
    "https://api.github.com/repos/harry0703/MoneyPrinterTurbo/releases/latest"
)
LATEST_RELEASE_PAGE_URL: Final = (
    "https://github.com/harry0703/MoneyPrinterTurbo/releases/latest"
)
# Проверка обновлений — вспомогательная функция, и сетевой сбой не должен заметно
# тормозить локальный WebUI. Таймауты подключения и чтения ограничены раздельно:
# GitHub успевает ответить на обычной сети, а офлайн-окружение не ждёт подолгу.
RELEASE_CHECK_TIMEOUT: Final = (1.0, 2.0)
RELEASE_CHECK_HEADERS: Final = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "MoneyPrinterTurbo-Version-Checker",
}
UPDATE_CHECK_CACHE_TTL_SECONDS: Final = 12 * 60 * 60


def _parse_version(value: str) -> Version:
    """Понимает привычные для GitHub теги вида ``v1.2.3`` и переводит их в сравнимую версию."""
    normalized = str(value or "").strip()
    if normalized.lower().startswith("v"):
        normalized = normalized[1:]
    return Version(normalized)


def get_available_update(current_version: str) -> str | None:
    """
    Возвращает последнюю стабильную версию выше текущей; если обновления нет или
    проверка не удалась, возвращает ``None``.

    Эндпоинт GitHub ``releases/latest`` сам исключает черновики и предрелизы,
    поэтому фильтрация по статусу публикации здесь не дублируется. WebUI вызывает
    эту функцию в фоне через ``AsyncUpdateChecker``. При проблемах с сетью,
    форматом ответа или тегом версии пишется лог, а поведение мягко деградирует
    до «не показывать уведомление» — на генерацию видео и другие ключевые
    функции это не влияет.
    """
    try:
        installed_version = _parse_version(current_version)
    except InvalidVersion:
        logger.warning(
            f"skip update check because current version is invalid: {current_version!r}"
        )
        return None

    try:
        response = requests.get(
            LATEST_RELEASE_API_URL,
            headers=RELEASE_CHECK_HEADERS,
            timeout=RELEASE_CHECK_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # Неудачная проверка обновлений — восстановимое некритичное исключение. Тип и
        # текст ошибки сохраняем, чтобы найти проблемы с прокси, DNS, лимитами GitHub или битым ответом, но обычного пользователя в WebUI не беспокоим.
        logger.debug(
            "GitHub release check failed: "
            f"error_type={type(exc).__name__}, error={exc}"
        )
        return None

    if not isinstance(payload, dict):
        logger.debug(
            "GitHub release check returned an invalid payload: "
            f"payload_type={type(payload).__name__}"
        )
        return None

    tag_name = payload.get("tag_name", "")
    try:
        latest_version = _parse_version(tag_name)
    except InvalidVersion:
        logger.warning(
            f"skip update notification because release tag is invalid: {tag_name!r}"
        )
        return None

    if latest_version <= installed_version:
        return None

    normalized_latest_version = str(latest_version)
    logger.info(
        "MoneyPrinterTurbo update available: "
        f"current={installed_version}, latest={normalized_latest_version}"
    )
    return normalized_latest_version


@dataclass(frozen=True)
class UpdateCheckSnapshot:
    """Моментальный статус фоновой проверки версии для неблокирующего чтения из WebUI."""

    complete: bool
    available_version: str | None = None


class AsyncUpdateChecker:
    """
    Выполняет проверку версии в фоновом потоке и кэширует последний результат.

    Streamlit перезапускает скрипт страницы с начала после любого взаимодействия
    с виджетом. Обращение к GitHub прямо в области заголовка блокировало бы всю
    страницу при первом открытии или после сброса кэша. Поэтому сетевой запрос
    вынесен в демон-поток, а страница лишь читает текущий снимок; по завершении
    проверки WebUI один раз обновляет результат короткоживущим fragment.

    Кэшируется любой исход — и «есть обновление», и «обновлений нет или сеть
    недоступна», — чтобы при недоступном GitHub запрос не повторялся на каждом
    rerun. Лок защищает только состояние в памяти и не оборачивает сетевой
    запрос, поэтому чтение статуса другими сессиями не блокируется.
    """

    def __init__(
        self,
        check: Callable[[str], str | None] = get_available_update,
        ttl_seconds: float = UPDATE_CHECK_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._check = check
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._current_version: str | None = None
        self._available_version: str | None = None
        self._completed_at: float | None = None
        self._checking = False

    def poll(self, current_version: str) -> UpdateCheckSnapshot:
        """Немедленно возвращает снимок проверки; при устаревшем кэше запускает в фоне новую."""
        normalized_current_version = str(current_version or "").strip()
        now = self._clock()

        with self._lock:
            cache_is_fresh = (
                self._current_version == normalized_current_version
                and self._completed_at is not None
                and now - self._completed_at < self._ttl_seconds
            )
            if cache_is_fresh:
                return UpdateCheckSnapshot(
                    complete=True,
                    available_version=self._available_version,
                )

            if (
                self._checking
                and self._current_version == normalized_current_version
            ):
                return UpdateCheckSnapshot(complete=False)

            # При смене версии или истечении кэша старый результат показывать нельзя.
            # Сначала сбрасываем состояние, потом стартуем новый поток — так вызывающая сторона на время проверки видит явный снимок pending.
            self._current_version = normalized_current_version
            self._available_version = None
            self._completed_at = None
            self._checking = True

            worker = threading.Thread(
                target=self._run_check,
                args=(normalized_current_version,),
                name="mpt-version-check",
                daemon=True,
            )
            worker.start()

        return UpdateCheckSnapshot(complete=False)

    def _run_check(self, current_version: str) -> None:
        try:
            available_version = self._check(current_version)
        except Exception:
            # get_available_update уже обрабатывает ожидаемые сетевые ошибки и проблемы с
            # данными. Здесь последний защитный рубеж фонового потока: пишем полный стек, иначе неожиданное исключение молча оборвёт поток и статус навсегда останется pending.
            logger.exception(
                "unexpected error while checking for a MoneyPrinterTurbo update"
            )
            available_version = None

        with self._lock:
            # В редких случаях версия может смениться прямо во время работы. Старый поток не вправе перезаписать статус новой версии.
            if self._current_version != current_version:
                return
            self._available_version = available_version
            self._completed_at = self._clock()
            self._checking = False


_ASYNC_UPDATE_CHECKER = AsyncUpdateChecker()


def poll_available_update(current_version: str) -> UpdateCheckSnapshot:
    """Читает состояние глобального фонового чекера, чтобы разные сессии Streamlit не дёргали GitHub повторно."""
    return _ASYNC_UPDATE_CHECKER.poll(current_version)
