import secrets
from typing import Annotated
from uuid import uuid4

from fastapi import Header, Request

from app.config import config
from app.models.exception import HttpException

MAX_TASK_ID_LENGTH = 128


def normalize_task_id(value: object) -> str:
    """Return a log-safe request ID, replacing invalid client input with a UUID."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TASK_ID_LENGTH
        or not value.isprintable()
    ):
        return str(uuid4())
    return value


def get_task_id(request: Request) -> str:
    return normalize_task_id(request.headers.get("x-task-id"))


def get_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    return api_key


def get_api_key_values(request: Request) -> list[str]:
    """Возвращает все заголовки API-ключа из запроса, сохраняя дубликаты для проверки безопасности."""

    # У Starlette Headers есть getlist(), который различает дубликаты заголовка от
    # прокси или клиента. Лёгкий дублёр Request в юнит-тестах использует обычный dict, поэтому оставлен совместимый откат.
    get_list = getattr(request.headers, "getlist", None)
    if callable(get_list):
        return [value for value in get_list("x-api-key") if isinstance(value, str)]

    api_key = get_api_key(request)
    return [api_key] if isinstance(api_key, str) else []


def verify_token(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
):
    """Решает по конфигурации, проверять ли API-ключ.

    Пустой ключ сохраняет текущий локальный режим без аутентификации. После того
    как администратор явно задал непустой ключ, и маршруты API, и скачивание
    артефактов задач требуют от клиента то же значение в заголовке ``x-api-key``.
    Объявление параметра заодно показывает этот заголовок в Swagger, что упрощает
    отладку в защищённом окружении.
    """

    configured_key = config.app.get("api_key", "")
    if configured_key in (None, ""):
        return None

    # Параметр конфигурации обязан быть строкой. Отклоняем списки, числа и прочие
    # неверные типы: неявное приведение к строке дало бы трудноуловимое поведение аутентификации. Текст ошибки сам ключ не содержит.
    if not isinstance(configured_key, str):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=500,
            message="API authentication is misconfigured",
        )

    # Параметр FastAPI нужен, чтобы объявить x-api-key в OpenAPI; реальная проверка
    # всегда читает Request — только так видно повторную отправку одноимённого
    # заголовка. Обычные клиенты и обратные прокси могут брать дубликаты в разном
    # порядке, поэтому их нужно отклонять, а не молча использовать первое или последнее значение.
    token_values = get_api_key_values(request)
    if not token_values and isinstance(x_api_key, str):
        token_values = [x_api_key]

    if len(token_values) != 1:
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    # compare_digest для str работает только с ASCII. Заголовок запроса — недоверенный
    # ввод, и атакующий может прислать символы Latin-1 и вызвать TypeError. Единое
    # кодирование в UTF-8 bytes сохраняет сравнение за постоянное время и поддерживает допустимый в TOML Unicode-ключ.
    token = token_values[0]
    if not secrets.compare_digest(
        token.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HttpException(
            task_id=get_task_id(request),
            status_code=401,
            message="invalid API key",
        )

    return None
