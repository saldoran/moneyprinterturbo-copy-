import errno
import threading
import time
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import toml

from app.config import config
from app.models.llm_provider import LLM_PROVIDER_REGISTRY, get_llm_provider


class TestConfigPersistence:
    @staticmethod
    def _wait_for_deferred_flush(timeout=1):
        """Дожидается завершения потока обновления конфигурации, чтобы параллельные тесты не делили фоновое состояние."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with config._pending_config_lock:
                if not config._pending_config_flush_scheduled:
                    return
            time.sleep(0.005)
        raise AssertionError("deferred config flush did not finish")

    @staticmethod
    def _load_example_config():
        config_path = Path(__file__).resolve().parents[2] / "config.example.toml"
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    def test_example_config_documents_runtime_settings(self):
        """Пример конфигурации должен показывать сервисы, материалы и продвинутые параметры запуска, которые пользователь ведёт вручную."""
        example_config = self._load_example_config()
        app_config = example_config["app"]

        assert example_config["listen_host"] == "0.0.0.0"
        assert example_config["listen_port"] == 8080
        assert example_config["log_level"] == "DEBUG"
        assert app_config["video_source"] in {
            "pexels",
            "pixabay",
            "coverr",
            "volcengine_seedance",
            "loomloom",
            "local",
        }
        assert "match_materials_to_script" in app_config
        assert app_config["script_generation_backend"] == "local"
        assert app_config["loomloom_api_token"] == ""
        assert app_config["loomloom_video_run_timeout_seconds"] == 1800
        assert app_config["volcengine_seedance_api_key"] == ""
        assert app_config["volcengine_seedance_base_url"].startswith("https://")
        assert app_config["volcengine_seedance_model"]
        assert "loomloom_market_listing_id" not in app_config
        assert "loomloom_video_market_listing_id" not in app_config
        assert app_config["shengsuanyun_api_key"] == ""
        assert example_config["whisper"]["device"] == "cpu"

    def test_example_config_covers_llm_provider_registry(self):
        """Настраиваемые поля провайдеров из Registry обязаны находиться в файле-примере."""
        app_config = self._load_example_config()["app"]

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.show_api_key:
                assert provider.config_key("api_key") in app_config
            if provider.show_base_url:
                assert provider.config_key("base_url") in app_config
            if provider.requires_model_name:
                assert provider.config_key("model_name") in app_config
            for field in provider.extra_fields:
                assert provider.config_key(field.config_suffix) in app_config

    def test_load_config_accepts_repeated_utf8_bom_without_rewriting_file(self):
        """Повторяющийся BOM не должен мешать запуску в Windows и не должен переписывать существующую конфигурацию."""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            original_content = b"\xef\xbb\xbf\xef\xbb\xbf[app]\nvideo_source = \"pexels\"\n"
            config_path.write_bytes(original_content)

            with patch.object(config, "config_file", str(config_path)):
                loaded_config = config.load_config()

            assert loaded_config["app"]["video_source"] == "pexels"
            assert config_path.read_bytes() == original_content

    def test_load_config_still_rejects_invalid_toml_after_bom_normalization(self):
        """Совместимость с BOM не должна маскировать настоящую синтаксическую ошибку: при сбое остаётся внятный диагностический лог."""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("[app\nvideo_source = \"pexels\"\n", encoding="utf-8")

            with (
                patch.object(config, "config_file", str(config_path)),
                patch.object(config.logger, "error") as error_mock,
            ):
                try:
                    config.load_config()
                except toml.TomlDecodeError:
                    pass
                else:
                    raise AssertionError("expected invalid TOML to be rejected")

            error_message = str(error_mock.call_args.args[0])
            assert str(config_path) in error_message
            assert "TomlDecodeError" in error_message

    def test_kimi_uses_current_default_model(self):
        """Если для Kimi модель не переопределена, используется модель по умолчанию из текущего релиза."""
        provider = get_llm_provider("moonshot")

        assert provider is not None
        assert provider.resolve_model_name("") == "kimi-k3"

    def test_upload_post_settings_belong_to_app_section(self):
        """Настройки публикации обязаны лежать в секции app, чтобы путь чтения в примере и в рантайме совпадал."""
        example_config = self._load_example_config()
        upload_post_keys = {
            "upload_post_enabled",
            "upload_post_api_key",
            "upload_post_username",
            "upload_post_platforms",
            "upload_post_auto_upload",
            "upload_post_youtube_privacy_status",
            "upload_post_max_pending_tasks",
        }

        assert upload_post_keys <= example_config["app"].keys()
        assert upload_post_keys.isdisjoint(example_config.get("ui", {}).keys())

    def test_save_config_uses_parseable_atomic_output(self):
        """
        Сохранение конфигурации сначала пишет временный файл, затем атомарно
        подменяет целевой. Тест заодно подтверждает, что вывод остаётся
        корректным TOML и после успешного сохранения в каталоге конфигурации не
        остаётся временных файлов.
        """
        original_cfg = dict(config._cfg)
        original_app = dict(config.app)
        try:
            with TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.toml"
                config.app["atomic_save_test"] = "ok"
                with (
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                ):
                    config.save_config()

                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                assert saved_config["app"]["atomic_save_test"] == "ok"
                assert list(Path(temp_dir).glob(".config-*.toml.tmp")) == []
        finally:
            config.app.clear()
            config.app.update(original_app)
            config._cfg.clear()
            config._cfg.update(original_cfg)

    def test_save_config_falls_back_for_bind_mounted_file(self):
        """
        Точку монтирования одиночного файла в Docker Desktop нельзя заменить
        через os.replace. При EBUSY нужно перезаписать файл на месте под локом и
        добиться, чтобы итоговое содержимое было полным, разбираемым и без
        оставшихся временных файлов.
        """
        original_cfg = dict(config._cfg)
        original_app = dict(config.app)
        try:
            with TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.toml"
                config_path.write_text("[app]\nold_value = true\n", encoding="utf-8")
                config.app["bind_mount_save_test"] = "ok"

                with (
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                    patch.object(
                        config.os,
                        "replace",
                        side_effect=OSError(
                            errno.EBUSY,
                            "Device or resource busy",
                        ),
                    ),
                    patch.object(config.logger, "warning") as warning_mock,
                ):
                    config.save_config()

                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                assert saved_config["app"]["bind_mount_save_test"] == "ok"
                assert list(Path(temp_dir).glob(".config-*.toml.tmp")) == []
                warning_mock.assert_called_once()
        finally:
            config.app.clear()
            config.app.update(original_app)
            config._cfg.clear()
            config._cfg.update(original_cfg)

    def test_save_config_does_not_hide_other_replace_errors(self):
        """Ошибки, отличные от EBUSY, пробрасываются дальше: сбой прав или диска нельзя выдавать за успешное сохранение."""
        original_cfg = dict(config._cfg)
        original_app = dict(config.app)
        try:
            with TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.toml"
                config_path.write_text("[app]\nold_value = true\n", encoding="utf-8")
                config.app["replace_error_test"] = "not-saved"

                with (
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                    patch.object(
                        config.os,
                        "replace",
                        side_effect=OSError(errno.EACCES, "Permission denied"),
                    ),
                ):
                    try:
                        config.save_config()
                    except OSError as exc:
                        assert exc.errno == errno.EACCES
                    else:
                        raise AssertionError("expected config save to fail")

                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                assert saved_config["app"]["old_value"] is True
                assert list(Path(temp_dir).glob(".config-*.toml.tmp")) == []
        finally:
            config.app.clear()
            config.app.update(original_app)
            config._cfg.clear()
            config._cfg.update(original_cfg)

    def test_runtime_config_lock_blocks_concurrent_config_writes(self):
        """Пока длинная задача держит рабочий лок, другие сессии не могут переписать глобальную конфигурацию посреди её выполнения."""
        write_started = threading.Event()
        write_finished = threading.Event()

        def update_config():
            write_started.set()
            config.app["runtime_lock_test"] = "updated"
            write_finished.set()

        config.app.pop("runtime_lock_test", None)
        with config.runtime_config_lock():
            worker = threading.Thread(target=update_config)
            worker.start()
            assert write_started.wait(timeout=1)
            assert not write_finished.wait(timeout=0.05)

        worker.join(timeout=1)
        assert write_finished.is_set()
        config.app.pop("runtime_lock_test", None)

    def test_runtime_config_lock_allows_idempotent_page_writes(self):
        """При обновлении страницы во время генерации обратная запись того же значения виджета не должна блокировать отрисовку всей страницы."""
        key = "runtime_lock_idempotent_test"
        config.app[key] = "unchanged"
        write_finished = threading.Event()

        def write_same_value():
            config.app[key] = "unchanged"
            assert config.app.setdefault(key, "other") == "unchanged"
            config.app.update({key: "unchanged"})
            assert config.app.pop("runtime_lock_missing_key", None) is None
            write_finished.set()

        with config.runtime_config_lock():
            worker = threading.Thread(target=write_same_value)
            worker.start()
            assert write_finished.wait(timeout=0.2)

        worker.join(timeout=1)
        assert config.app[key] == "unchanged"
        config.app.pop(key, None)

    def test_try_runtime_config_lock_returns_immediately_when_busy(self):
        """Лок прослушивания не ждёт, пока длинная задача отпустит глобальную конфигурацию: если занято, UI сразу предлагает повторить."""
        attempted = threading.Event()
        result = []

        def try_lock():
            with config.try_runtime_config_lock() as acquired:
                result.append(acquired)
            attempted.set()

        with config.runtime_config_lock():
            worker = threading.Thread(target=try_lock)
            worker.start()
            assert attempted.wait(timeout=0.2)

        worker.join(timeout=1)
        assert result == [False]

        with config.try_runtime_config_lock() as acquired:
            assert acquired is True

    def test_nonblocking_update_is_applied_after_runtime_task_finishes(self):
        """Правки в WebUI не ждут длинную задачу, а после её завершения последнее значение обязательно применяется и сохраняется."""
        key = "nonblocking_runtime_update_test"
        original_value = config.app.get(key, config._MISSING)
        update_finished = threading.Event()
        update_result = []

        def update_config():
            update_result.append(
                config.update_config_nonblocking(config.app, key, "updated")
            )
            update_finished.set()

        try:
            with patch.object(config, "save_config") as save_config:
                with config.runtime_config_lock():
                    worker = threading.Thread(target=update_config)
                    worker.start()
                    assert update_finished.wait(timeout=0.2)
                    assert update_result == [False]
                    assert config.app.get(key) != "updated"

                worker.join(timeout=1)
                assert config.app[key] == "updated"
                save_config.assert_called_once()
        finally:
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_nonblocking_update_keeps_only_latest_value(self):
        """Если один виджет меняют многократно за время задачи, применяется только последний выбор."""
        key = "nonblocking_latest_value_test"
        original_value = config.app.get(key, config._MISSING)
        updates_finished = threading.Event()

        def update_config():
            assert not config.update_config_nonblocking(config.app, key, "first")
            assert not config.update_config_nonblocking(config.app, key, "latest")
            updates_finished.set()

        try:
            with patch.object(config, "save_config"):
                with config.runtime_config_lock():
                    worker = threading.Thread(target=update_config)
                    worker.start()
                    assert updates_finished.wait(timeout=0.2)

                worker.join(timeout=1)
                assert config.app[key] == "latest"
        finally:
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_nonblocking_delete_is_applied_after_runtime_task_finishes(self):
        """При возврате к варианту по умолчанию удаление настройки тоже не должно блокировать выполняющуюся задачу видео."""
        key = "nonblocking_runtime_delete_test"
        config.app[key] = "custom"
        delete_finished = threading.Event()
        delete_result = []

        def delete_config():
            delete_result.append(config.delete_config_nonblocking(config.app, key))
            delete_finished.set()

        try:
            with patch.object(config, "save_config") as save_config:
                with config.runtime_config_lock():
                    worker = threading.Thread(target=delete_config)
                    worker.start()
                    assert delete_finished.wait(timeout=0.2)
                    assert delete_result == [False]
                    assert config.app[key] == "custom"

                worker.join(timeout=1)
                assert key not in config.app
                save_config.assert_called_once()
        finally:
            config.app.pop(key, None)

    def test_try_save_config_returns_immediately_while_runtime_task_is_active(self):
        """Запрос сохранения при rerun страницы не ждёт, пока задача видео отпустит лок конфигурации."""
        save_finished = threading.Event()
        save_result = []

        def save_config():
            save_result.append(config.try_save_config())
            save_finished.set()

        with patch.object(config, "save_config") as blocking_save:
            with config.runtime_config_lock():
                worker = threading.Thread(target=save_config)
                worker.start()
                assert save_finished.wait(timeout=0.2)
                assert save_result == [False]

            worker.join(timeout=1)
            blocking_save.assert_called_once()

        self._wait_for_deferred_flush()

    def test_try_runtime_lock_flushes_updates_queued_during_operation(self):
        """Когда короткая операция отпускает лок конфигурации, пришедшие за это время правки страницы тоже применяются и сохраняются."""
        key = "try_runtime_queued_update_test"
        original_value = config.app.get(key, config._MISSING)
        update_finished = threading.Event()

        def queue_update():
            assert not config.update_config_nonblocking(config.app, key, "updated")
            update_finished.set()

        try:
            with patch.object(config, "save_config") as save_config:
                with config.try_runtime_config_lock() as acquired:
                    assert acquired is True
                    worker = threading.Thread(target=queue_update)
                    worker.start()
                    assert update_finished.wait(timeout=0.2)
                    assert config.app.get(key) != "updated"

                worker.join(timeout=1)
                assert config.app[key] == "updated"
                save_config.assert_called_once()

            self._wait_for_deferred_flush()
        finally:
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_update_queued_during_save_is_flushed_after_lock_release(self):
        """Правки, пришедшие во время финального сохранения, не должны застрять в очереди и не должны затираться более ранним значением."""
        key = "late_runtime_update_test"
        original_value = config.app.get(key, config._MISSING)
        runtime_entered = threading.Event()
        release_runtime = threading.Event()
        first_save_started = threading.Event()
        release_first_save = threading.Event()
        second_save_finished = threading.Event()
        save_count = 0
        save_count_lock = threading.Lock()

        def blocking_save():
            nonlocal save_count
            with save_count_lock:
                save_count += 1
                current_save = save_count
            if current_save == 1:
                first_save_started.set()
                assert release_first_save.wait(timeout=1)
            elif current_save == 2:
                second_save_finished.set()

        def hold_runtime_lock():
            with config.runtime_config_lock():
                runtime_entered.set()
                assert release_runtime.wait(timeout=1)

        try:
            with patch.object(config, "save_config", side_effect=blocking_save):
                runtime_worker = threading.Thread(target=hold_runtime_lock)
                runtime_worker.start()
                assert runtime_entered.wait(timeout=1)

                assert not config.update_config_nonblocking(config.app, key, "first")
                release_runtime.set()
                assert first_save_started.wait(timeout=1)

                # Первый круг сохранения уже снял снимок конфигурации, поэтому пришедшее сейчас
                # значение фоновый поток обновления обязан применить и сохранить повторно после освобождения лока — итог определяется именно им.
                assert not config.update_config_nonblocking(config.app, key, "latest")
                release_first_save.set()

                runtime_worker.join(timeout=1)
                assert not runtime_worker.is_alive()
                assert second_save_finished.wait(timeout=1)
                assert config.app[key] == "latest"
                assert save_count == 2

            self._wait_for_deferred_flush()
        finally:
            release_runtime.set()
            release_first_save.set()
            if original_value is config._MISSING:
                config.app.pop(key, None)
            else:
                config.app[key] = original_value

    def test_config_snapshot_includes_pending_updates(self):
        """Пока генерация видео держит лок, новый запрос к LLM должен видеть свежий выбор в интерфейсе, а не старую конфигурацию."""
        keys = {
            "llm_provider": "pending-provider",
            "pending-provider_api_key": "pending-key",
            "pending-provider_model_name": "pending-model",
        }
        original_values = {key: config.app.get(key, config._MISSING) for key in keys}
        updates_finished = threading.Event()

        def queue_updates():
            for key, value in keys.items():
                assert not config.update_config_nonblocking(config.app, key, value)
            updates_finished.set()

        try:
            with patch.object(config, "save_config"):
                with config.runtime_config_lock():
                    worker = threading.Thread(target=queue_updates)
                    worker.start()
                    assert updates_finished.wait(timeout=0.2)

                    snapshot = config.snapshot_config_with_pending(config.app)
                    assert all(snapshot[key] == value for key, value in keys.items())
                    assert config.app.get("llm_provider") != "pending-provider"

                worker.join(timeout=1)

            self._wait_for_deferred_flush()
        finally:
            for key, original_value in original_values.items():
                if original_value is config._MISSING:
                    config.app.pop(key, None)
                else:
                    config.app[key] = original_value
