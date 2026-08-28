from fastapi import APIRouter


def new_router(dependencies=None):
    router = APIRouter()
    router.tags = ["V1"]
    router.prefix = "/api/v1"
    # Применяем зависимость аутентификации ко всем маршрутам
    if dependencies:
        router.dependencies = dependencies
    return router
