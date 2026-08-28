import runpy
from pathlib import Path
from unittest.mock import patch

from app.config import config


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_main_starts_uvicorn_with_runtime_config():
    """
    Точка входа сервиса только передаёт конфигурацию запуска в Uvicorn. Реальный
    старт сервера замокан: тест не занимает порт и подтверждает, что адрес, порт
    и настройка горячей перезагрузки не теряются на уровне точки входа.
    """
    with (
        patch.object(config, "listen_host", "127.0.0.1"),
        patch.object(config, "listen_port", 8765),
        patch.object(config, "reload_debug", True),
        patch("uvicorn.run") as run_server,
    ):
        runpy.run_path(str(ROOT_DIR / "main.py"), run_name="__main__")

    run_server.assert_called_once_with(
        app="app.asgi:app",
        host="127.0.0.1",
        port=8765,
        reload=True,
        log_level="warning",
    )
