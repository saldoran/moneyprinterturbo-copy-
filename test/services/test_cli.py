import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cli
from app.config import config as app_config


class TestCli(unittest.TestCase):
    def setUp(self):
        # ``build_video_params`` берёт значения по умолчанию из секции [ui] локального
        # config.toml. Без изоляции эти тесты зависели бы от состояния машины
        # разработчика: стоит удалить сохранённый в конфигурации шрифт, как сработает
        # проверка шрифтов в ``prepare_cli_files``, и посторонние тесты упадут с вводящей в заблуждение ошибкой.
        ui_patch = patch.dict(app_config.ui, {}, clear=True)
        ui_patch.start()
        self.addCleanup(ui_patch.stop)

    def test_default_voice_is_valid_edge_tts_voice(self):
        args = cli.parse_args(["--video-subject", "测试主题"])
        params = cli.build_video_params(args)

        self.assertEqual(params.voice_name, "zh-CN-XiaoxiaoNeural-Female")

    def test_complete_script_can_replace_video_subject(self):
        args = cli.parse_args(["--video-script", "完整的视频文案"])
        params = cli.build_video_params(args)

        self.assertEqual(params.video_subject, "")
        self.assertEqual(params.video_script, "完整的视频文案")

    def test_subject_or_script_is_required(self):
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args([])

        self.assertEqual(cm.exception.code, 2)

    def test_build_video_params_with_local_materials(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "测试主题",
                "--video-source",
                "local",
                "--video-materials",
                "a.mp4, ,b.jpg,",
                "--video-terms",
                "foo, bar",
            ]
        )

        params = cli.build_video_params(args)
        materials = params.video_materials

        self.assertEqual(params.video_subject, "测试主题")
        self.assertEqual(params.video_source, "local")
        self.assertEqual([m.url for m in materials], ["a.mp4", "b.jpg"])
        self.assertTrue(all(m.provider == "local" for m in materials))
        self.assertEqual(params.video_terms, ["foo", "bar"])

    def test_run_cli_dispatches_task_start(self):
        with patch("app.services.task.start", return_value={"script": "ok"}) as start, patch(
            "app.utils.utils.get_uuid", return_value="task-123"
        ), patch("builtins.print") as print_mock:
            code = cli.run_cli(["--video-subject", "命令行测试", "--stop-at", "script"])

        self.assertEqual(code, 0)
        self.assertTrue(start.called)
        kwargs = start.call_args.kwargs
        self.assertEqual(kwargs["task_id"], "task-123")
        self.assertEqual(kwargs["stop_at"], "script")
        self.assertEqual(kwargs["params"].video_subject, "命令行测试")
        self.assertIs(kwargs["allow_server_file_input"], True)
        print_mock.assert_called_once()

    def test_run_cli_returns_error_when_task_fails(self):
        with patch("app.services.task.start", return_value=None), patch(
            "app.utils.utils.get_uuid", return_value="task-456"
        ), patch.object(cli.logger, "error") as log_error:
            code = cli.run_cli(["--video-subject", "失败场景"])

        self.assertEqual(code, 1)
        log_error.assert_called_once()

    def test_run_cli_returns_error_for_structured_task_failure(self):
        """Когда сервис задач возвращает структурированное сообщение об ошибке, CLI всё равно обязан выйти с ненулевым кодом."""
        failure = {
            "task_id": "task-structured-failure",
            "state": -1,
            "progress": 30,
            "failed_stage": "audio",
            "error": "TTS request timed out",
        }

        with patch("app.services.task.start", return_value=failure), patch(
            "app.utils.utils.get_uuid", return_value="task-structured-failure"
        ), patch.object(cli.logger, "error") as log_error, patch(
            "builtins.print"
        ) as print_mock:
            code = cli.run_cli(["--video-subject", "失败场景"])

        self.assertEqual(code, 1)
        print_mock.assert_not_called()
        self.assertIn("stage=audio", log_error.call_args.args[0])
        self.assertIn("TTS request timed out", log_error.call_args.args[0])

    def test_subtitle_enabled_by_default(self):
        args = cli.parse_args(["--video-subject", "test"])
        params = cli.build_video_params(args)
        self.assertTrue(params.subtitle_enabled)

    def test_subtitle_disabled_with_no_flag(self):
        args = cli.parse_args(["--video-subject", "test", "--no-subtitle-enabled"])
        params = cli.build_video_params(args)
        self.assertFalse(params.subtitle_enabled)

    def test_coverr_video_source_accepted(self):
        args = cli.parse_args(["--video-subject", "test", "--video-source", "coverr"])
        params = cli.build_video_params(args)
        self.assertEqual(params.video_source, "coverr")

    def test_seedance_video_source_requires_explicit_charge_confirmation(self):
        with self.assertRaises(SystemExit) as raised:
            cli.parse_args(
                [
                    "--video-subject",
                    "test",
                    "--video-source",
                    "volcengine_seedance",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--video-source",
                "volcengine_seedance",
                "--confirm-seedance-charge",
            ]
        )
        self.assertEqual(
            cli.build_video_params(args).video_source, "volcengine_seedance"
        )

    def test_seedance_confirmation_is_not_required_before_material_stage(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--video-source",
                "volcengine_seedance",
                "--stop-at",
                "script",
            ]
        )
        self.assertEqual(args.video_source, "volcengine_seedance")

    def test_batch_seedance_source_uses_global_charge_confirmation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": "Seedance batch task",
                            "video_source": "volcengine_seedance",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch("app.services.task.start") as start:
                rejected = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "materials"]
                )
            self.assertEqual(rejected, 2)
            start.assert_not_called()

            with (
                patch(
                    "app.services.task.start",
                    return_value={"state": 1, "materials": ["ok"]},
                ) as start,
                patch("app.utils.utils.get_uuid", return_value="task-seedance"),
                redirect_stdout(io.StringIO()),
            ):
                accepted = cli.run_cli(
                    [
                        "--batch-file",
                        str(manifest),
                        "--stop-at",
                        "materials",
                        "--confirm-seedance-charge",
                    ]
                )

            self.assertEqual(accepted, 0)
            start.assert_called_once()

    def test_build_video_params_with_script_video_and_audio_options(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--video-language",
                "en",
                "--paragraph-number",
                "3",
                "--video-script-prompt",
                "use a lighter tone",
                "--custom-system-prompt",
                "write concise short-form scripts",
                "--video-concat-mode",
                "sequential",
                "--video-transition-mode",
                "fade-in",
                "--video-clip-duration",
                "4",
                "--match-materials-to-script",
                "--voice-volume",
                "1.2",
                "--voice-rate",
                "1.1",
                "--bgm-type",
                "custom",
                "--bgm-file",
                "output001.mp3",
                "--bgm-volume",
                "0.3",
                "--n-threads",
                "4",
            ]
        )

        params = cli.build_video_params(args)

        self.assertEqual(params.video_language, "en")
        self.assertEqual(params.paragraph_number, 3)
        self.assertEqual(params.video_script_prompt, "use a lighter tone")
        self.assertEqual(params.custom_system_prompt, "write concise short-form scripts")
        self.assertEqual(params.video_concat_mode, "sequential")
        self.assertEqual(params.video_transition_mode, "FadeIn")
        self.assertEqual(params.video_clip_duration, 4)
        self.assertTrue(params.match_materials_to_script)
        self.assertEqual(params.voice_volume, 1.2)
        self.assertEqual(params.voice_rate, 1.1)
        self.assertEqual(params.bgm_type, "custom")
        self.assertEqual(params.bgm_file, "output001.mp3")
        self.assertEqual(params.bgm_volume, 0.3)
        self.assertEqual(params.n_threads, 4)

    def test_custom_audio_file_maps_to_video_params(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--custom-audio-file",
                "voiceover.mp3",
            ]
        )
        params = cli.build_video_params(args)

        self.assertEqual(params.custom_audio_file, "voiceover.mp3")

    def test_build_video_params_with_subtitle_style_options(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--font-name",
                "MicrosoftYaHeiBold.ttc",
                "--subtitle-position",
                "custom",
                "--custom-position",
                "42.5",
                "--text-fore-color",
                "#AABBCC",
                "--font-size",
                "72",
                "--stroke-color",
                "#112233",
                "--stroke-width",
                "2.5",
                "--subtitle-background-color",
                "#000001",
                "--rounded-subtitle-background",
            ]
        )

        params = cli.build_video_params(args)

        self.assertEqual(params.font_name, "MicrosoftYaHeiBold.ttc")
        self.assertEqual(params.subtitle_position, "custom")
        self.assertEqual(params.custom_position, 42.5)
        self.assertEqual(params.text_fore_color, "#AABBCC")
        self.assertEqual(params.font_size, 72)
        self.assertEqual(params.stroke_color, "#112233")
        self.assertEqual(params.stroke_width, 2.5)
        self.assertEqual(params.text_background_color, "#000001")
        self.assertTrue(params.rounded_subtitle_background)

    def test_disabled_subtitle_background_rejects_rounding(self):
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(
                [
                    "--video-subject",
                    "test",
                    "--no-subtitle-background-enabled",
                    "--rounded-subtitle-background",
                ]
            )

        self.assertEqual(cm.exception.code, 2)

    def test_bgm_type_none_maps_to_disabled_background_music(self):
        args = cli.parse_args(["--video-subject", "test", "--bgm-type", "none"])
        params = cli.build_video_params(args)
        self.assertEqual(params.bgm_type, "")

    def test_sonilo_prompt_implies_sonilo_bgm_mode(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--sonilo-bgm-prompt",
                "warm acoustic",
            ]
        )
        params = cli.build_video_params(args)
        self.assertEqual(params.bgm_type, "sonilo")
        self.assertEqual(params.sonilo_bgm_prompt, "warm acoustic")

    def test_local_material_filename_resolved_to_absolute_path(self):
        """After preprocess_video, material.url should be an absolute path, not a bare filename."""
        import os
        from app.utils import utils
        from app.services import video as vd
        from app.models.schema import MaterialInfo

        local_videos_dir = utils.storage_dir("local_videos", create=True)
        # Create a minimal valid video file for testing
        test_filename = "_cli_test_resolve.mp4"
        test_filepath = os.path.join(local_videos_dir, test_filename)
        # We need a real video file; use a tiny one via moviepy
        try:
            from moviepy import ColorClip
            clip = ColorClip(size=(640, 640), color=(0, 0, 0), duration=1)
            clip.write_videofile(test_filepath, fps=1, logger=None)
            clip.close()
        except Exception:
            self.skipTest("moviepy not available for creating test video")

        try:
            materials = [MaterialInfo(provider="local", url=test_filename, duration=0)]
            result = vd.preprocess_video(materials=materials, clip_duration=4)
            self.assertTrue(len(result) > 0, "preprocess_video should return valid materials")
            self.assertTrue(
                os.path.isabs(result[0].url),
                f"material url should be absolute path, got: {result[0].url}",
            )
            self.assertEqual(result[0].url, test_filepath)
        finally:
            if os.path.exists(test_filepath):
                os.remove(test_filepath)


    def test_local_source_requires_video_materials(self):
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(["--video-subject", "test", "--video-source", "local"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_local_source_does_not_require_materials_before_material_stage(self):
        for stop_at in ("script", "audio", "subtitle"):
            with self.subTest(stop_at=stop_at):
                args = cli.parse_args(
                    [
                        "--video-subject",
                        "test",
                        "--video-source",
                        "local",
                        "--stop-at",
                        stop_at,
                    ]
                )
                self.assertEqual(args.stop_at, stop_at)

    def test_local_source_stop_at_terms_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args([
                "--video-subject", "test",
                "--video-source", "local",
                "--video-materials", "a.mp4",
                "--stop-at", "terms",
            ])
        self.assertNotEqual(cm.exception.code, 0)

    def test_video_materials_rejected_for_online_source(self):
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(
                [
                    "--video-subject",
                    "test",
                    "--video-source",
                    "pexels",
                    "--video-materials",
                    "a.mp4",
                ]
            )

        self.assertEqual(cm.exception.code, 2)

    def test_positive_volume_custom_bgm_requires_file_before_task_start(self):
        """При включённом пользовательском BGM отсутствие файла должно сообщаться до запуска задачи."""
        with (
            patch("app.services.task.start") as start,
            patch.object(cli.logger, "error") as log_error,
        ):
            code = cli.run_cli(
                [
                    "--video-subject",
                    "test",
                    "--bgm-type",
                    "custom",
                    "--stop-at",
                    "script",
                ]
            )

        self.assertEqual(code, 2)
        start.assert_not_called()
        self.assertIn(
            "--bgm-file is required",
            str(log_error.call_args),
        )

    def test_bgm_file_implies_custom_mode(self):
        args = cli.parse_args(
            ["--video-subject", "test", "--bgm-file", "output001.mp3"]
        )
        self.assertEqual(args.bgm_type, "custom")

    def test_zero_volume_custom_bgm_skips_file_requirement_and_resolution(self):
        """При нулевой громкости отсутствующий или некорректный файл игнорируется — как в WebUI и сервисе видео."""
        file_arguments = [[], ["--bgm-file", "missing-background.mp3"]]
        for extra_arguments in file_arguments:
            with self.subTest(extra_arguments=extra_arguments):
                args = cli.parse_args(
                    [
                        "--video-subject",
                        "test",
                        "--bgm-type",
                        "custom",
                        "--bgm-volume",
                        "0",
                        *extra_arguments,
                    ]
                )
                params = cli.build_video_params(args)
                with patch(
                    "app.services.bgm.resolve_bgm_file",
                    side_effect=AssertionError(
                        "zero-volume BGM must not resolve a file"
                    ),
                ) as resolver:
                    cli.prepare_cli_files(params, stop_at="script")

                resolver.assert_not_called()
                self.assertEqual(params.bgm_file, "")

    def test_custom_bgm_reuses_service_formats_and_managed_path_resolution(self):
        """CLI следует белому списку форматов сервиса BGM и больше не ограничивается MP3 самостоятельно."""
        from app.services import bgm as bgm_service

        for extension in bgm_service.SUPPORTED_BGM_EXTENSIONS:
            with self.subTest(extension=extension):
                filename = f"uploaded{extension}"
                resolved_path = f"/managed/storage/bgm/{filename}"
                args = cli.parse_args(
                    [
                        "--video-subject",
                        "test",
                        "--bgm-type",
                        "custom",
                        "--bgm-file",
                        filename,
                    ]
                )
                params = cli.build_video_params(args)
                with patch.object(
                    bgm_service,
                    "resolve_bgm_file",
                    return_value=resolved_path,
                ) as resolver:
                    cli.prepare_cli_files(params, stop_at="script")

                resolver.assert_called_once_with(filename)
                self.assertEqual(params.bgm_file, resolved_path)

    def test_custom_bgm_reports_service_resolution_failure_before_task_start(self):
        """Недопустимый формат или путь за границей превращаются в ошибку CLI с единым перечнем допустимых форматов."""
        from app.services import bgm as bgm_service

        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--bgm-type",
                "custom",
                "--bgm-file",
                "unsafe.exe",
            ]
        )
        params = cli.build_video_params(args)
        with (
            patch.object(
                bgm_service,
                "resolve_bgm_file",
                side_effect=ValueError("unsupported background music path"),
            ),
            self.assertRaisesRegex(ValueError, "storage/bgm or resource/songs"),
        ):
            cli.prepare_cli_files(params, stop_at="script")

    def test_invalid_aspect_and_non_finite_numbers_are_argument_errors(self):
        invalid_argvs = [
            ["--video-subject", "test", "--video-aspect", "invalid"],
            ["--video-subject", "test", "--custom-position", "nan"],
            ["--video-subject", "test", "--voice-rate", "inf"],
        ]
        for argv in invalid_argvs:
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as cm:
                cli.parse_args(argv)
            self.assertEqual(cm.exception.code, 2)

    def test_custom_position_requires_custom_subtitle_position(self):
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(
                ["--video-subject", "test", "--custom-position", "50"]
            )

        self.assertEqual(cm.exception.code, 2)

    def test_task_id_must_be_uuid(self):
        task_id = str(uuid4())
        args = cli.parse_args(
            ["--video-subject", "test", "--task-id", task_id]
        )
        self.assertEqual(args.task_id, task_id)

        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(
                ["--video-subject", "test", "--task-id", "../../escape"]
            )
        self.assertEqual(cm.exception.code, 2)

    def test_prepare_cli_files_accepts_relative_and_absolute_materials(self):
        with (
            tempfile.TemporaryDirectory() as source_dir,
            tempfile.TemporaryDirectory() as managed_dir,
        ):
            relative_file = Path(source_dir) / "relative.mp4"
            absolute_file = Path(source_dir) / "absolute.jpg"
            relative_file.write_bytes(b"relative-video")
            absolute_file.write_bytes(b"absolute-image")
            old_cwd = os.getcwd()
            try:
                os.chdir(source_dir)
                args = cli.parse_args(
                    [
                        "--video-subject",
                        "test",
                        "--video-source",
                        "local",
                        "--video-materials",
                        f"relative.mp4,{absolute_file}",
                    ]
                )
                params = cli.build_video_params(args)
                with patch("app.utils.utils.storage_dir", return_value=managed_dir):
                    cli.prepare_cli_files(params, stop_at="video")
            finally:
                os.chdir(old_cwd)

            prepared_paths = [Path(item.url) for item in params.video_materials]
            self.assertTrue(all(path.parent == Path(managed_dir) for path in prepared_paths))
            self.assertEqual(
                {path.read_bytes() for path in prepared_paths},
                {b"relative-video", b"absolute-image"},
            )

    def test_prepare_cli_files_rejects_missing_material_before_task_start(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--video-source",
                "local",
                "--video-materials",
                "missing.mp4",
            ]
        )
        params = cli.build_video_params(args)

        with tempfile.TemporaryDirectory() as managed_dir, patch(
            "app.utils.utils.storage_dir", return_value=managed_dir
        ):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                cli.prepare_cli_files(params, stop_at="video")

    def test_run_cli_rejects_missing_material_before_starting_task(self):
        with patch("app.services.task.start") as start:
            code = cli.run_cli(
                [
                    "--video-subject",
                    "test",
                    "--video-source",
                    "local",
                    "--video-materials",
                    "missing.mp4",
                ]
            )

        self.assertEqual(code, 2)
        start.assert_not_called()

    def test_prepare_cli_files_resolves_relative_custom_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_file = Path(temp_dir) / "voiceover.mp3"
            audio_file.write_bytes(b"audio")
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                args = cli.parse_args(
                    [
                        "--video-subject",
                        "test",
                        "--custom-audio-file",
                        "voiceover.mp3",
                        "--stop-at",
                        "audio",
                    ]
                )
                params = cli.build_video_params(args)
                cli.prepare_cli_files(params, stop_at="audio")
            finally:
                os.chdir(old_cwd)

            self.assertEqual(params.custom_audio_file, str(audio_file.resolve()))

    def test_batch_file_does_not_require_global_subject_and_conflicts_with_task_id(self):
        args = cli.parse_args(["--batch-file", "tasks.jsonl"])
        self.assertEqual(args.batch_file, "tasks.jsonl")

        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(
                [
                    "--batch-file",
                    "tasks.jsonl",
                    "--task-id",
                    str(uuid4()),
                ]
            )

        self.assertEqual(cm.exception.code, 2)

    def test_batch_json_array_merges_cli_defaults_and_prints_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"video_subject": "first subject"},
                        {"video_script": "prepared second script"},
                    ]
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch(
                    "app.services.task.start",
                    side_effect=[
                        {"state": 1, "script": "first"},
                        {"state": 1, "script": "second"},
                    ],
                ) as start,
                patch(
                    "app.utils.utils.get_uuid",
                    side_effect=["task-one", "task-two"],
                ),
                redirect_stdout(output),
            ):
                code = cli.run_cli(
                    [
                        "--batch-file",
                        str(manifest),
                        "--stop-at",
                        "script",
                        "--voice-name",
                        "global-voice",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(start.call_count, 2)
        first_call, second_call = start.call_args_list
        self.assertEqual(first_call.kwargs["task_id"], "task-one")
        self.assertEqual(second_call.kwargs["task_id"], "task-two")
        self.assertEqual(first_call.kwargs["params"].video_subject, "first subject")
        self.assertEqual(
            second_call.kwargs["params"].video_script,
            "prepared second script",
        )
        self.assertEqual(first_call.kwargs["params"].voice_name, "global-voice")
        self.assertEqual(second_call.kwargs["params"].voice_name, "global-voice")
        self.assertIs(first_call.kwargs["allow_server_file_input"], True)
        self.assertIs(second_call.kwargs["allow_server_file_input"], True)
        summary = json.loads(output.getvalue())
        self.assertEqual(
            {key: summary[key] for key in ("total", "succeeded", "failed")},
            {"total": 2, "succeeded": 2, "failed": 0},
        )
        self.assertEqual(
            [task["status"] for task in summary["tasks"]],
            ["succeeded", "succeeded"],
        )
        self.assertEqual(
            set(summary["tasks"][0]),
            {"index", "task_id", "status", "result", "failed_stage", "error"},
        )

    def test_batch_jsonl_continues_after_runtime_and_structured_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "tasks.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps({"video_subject": subject})
                    for subject in ("one", "two", "three")
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch(
                    "app.services.task.start",
                    side_effect=[
                        {"state": 1, "script": "ok"},
                        RuntimeError("provider unavailable"),
                        {
                            "state": -1,
                            "failed_stage": "audio",
                            "error": "TTS failed",
                        },
                    ],
                ) as start,
                patch(
                    "app.utils.utils.get_uuid",
                    side_effect=["task-one", "task-two", "task-three"],
                ),
                patch.object(cli.logger, "exception"),
                patch.object(cli.logger, "error"),
                redirect_stdout(output),
            ):
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "script"]
                )

        self.assertEqual(code, 1)
        self.assertEqual(start.call_count, 3)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(
            [task["status"] for task in summary["tasks"]],
            ["succeeded", "failed", "failed"],
        )
        self.assertEqual(summary["tasks"][1]["failed_stage"], "runtime")
        self.assertEqual(summary["tasks"][1]["error"], "provider unavailable")
        self.assertEqual(summary["tasks"][2]["failed_stage"], "audio")

    def test_invalid_later_batch_task_prevents_every_task_from_starting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"video_subject": "valid"},
                        {"video_subject": "invalid", "unknown_option": True},
                    ]
                ),
                encoding="utf-8",
            )
            with (
                patch("app.services.task.start") as start,
                patch.object(cli.logger, "error") as log_error,
            ):
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "script"]
                )

        self.assertEqual(code, 2)
        start.assert_not_called()
        self.assertIn("unknown VideoParams fields", str(log_error.call_args))

    def test_missing_file_in_later_batch_task_prevents_every_task_from_starting(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.TemporaryDirectory() as managed_dir,
        ):
            source_file = Path(temp_dir) / "valid.mp4"
            source_file.write_bytes(b"valid video")
            manifest = Path(temp_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": "valid local material",
                            "video_source": "local",
                            "video_materials": [
                                {"provider": "local", "url": "valid.mp4"}
                            ],
                        },
                        {
                            "video_subject": "missing local material",
                            "video_source": "local",
                            "video_materials": [
                                {"provider": "local", "url": "missing.mp4"}
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            with (
                patch("app.services.task.start") as start,
                patch("app.utils.utils.storage_dir", return_value=managed_dir),
            ):
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "materials"]
                )

            self.assertEqual(code, 2)
            start.assert_not_called()
            self.assertEqual(os.listdir(managed_dir), [])

    def test_batch_reuses_one_managed_copy_for_repeated_local_material(self):
        with (
            tempfile.TemporaryDirectory() as manifest_dir,
            tempfile.TemporaryDirectory() as managed_dir,
        ):
            source_file = Path(manifest_dir) / "shared.mp4"
            source_file.write_bytes(b"shared video")
            manifest = Path(manifest_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": subject,
                            "video_source": "local",
                            "video_materials": [
                                {"provider": "local", "url": "shared.mp4"}
                            ],
                        }
                        for subject in ("first", "second")
                    ]
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "app.services.task.start",
                    side_effect=[
                        {"state": 1, "materials": ["first"]},
                        {"state": 1, "materials": ["second"]},
                    ],
                ) as start,
                patch(
                    "app.utils.utils.get_uuid",
                    side_effect=["task-one", "task-two"],
                ),
                patch("app.utils.utils.storage_dir", return_value=managed_dir),
                redirect_stdout(io.StringIO()),
            ):
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "materials"]
                )

            first_path = start.call_args_list[0].kwargs["params"].video_materials[0].url
            second_path = start.call_args_list[1].kwargs["params"].video_materials[0].url
            managed_files = os.listdir(managed_dir)

        self.assertEqual(code, 0)
        self.assertEqual(first_path, second_path)
        self.assertEqual(len(managed_files), 1)
        self.assertTrue(managed_files[0].startswith("cli-material-"))

    def test_batch_copy_failure_removes_all_managed_materials(self):
        with (
            tempfile.TemporaryDirectory() as manifest_dir,
            tempfile.TemporaryDirectory() as managed_dir,
        ):
            first_source = Path(manifest_dir) / "first.mp4"
            second_source = Path(manifest_dir) / "second.mp4"
            first_source.write_bytes(b"first video")
            second_source.write_bytes(b"second video")
            manifest = Path(manifest_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": name,
                            "video_source": "local",
                            "video_materials": [
                                {"provider": "local", "url": f"{name}.mp4"}
                            ],
                        }
                        for name in ("first", "second")
                    ]
                ),
                encoding="utf-8",
            )
            real_copy = cli.shutil.copy2

            def fail_second_copy(source, target):
                if os.path.basename(source) == "second.mp4":
                    Path(target).write_bytes(b"partial copy")
                    raise OSError("simulated copy failure")
                return real_copy(source, target)

            with (
                patch("app.services.task.start") as start,
                patch("app.utils.utils.storage_dir", return_value=managed_dir),
                patch.object(cli.shutil, "copy2", side_effect=fail_second_copy),
            ):
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "materials"]
                )

            self.assertEqual(code, 2)
            start.assert_not_called()
            self.assertEqual(os.listdir(managed_dir), [])

    def test_batch_rejects_non_object_and_unknown_material_fields(self):
        invalid_manifests = (
            ["not an object"],
            [
                {
                    "video_subject": "invalid material",
                    "video_source": "local",
                    "video_materials": [
                        {"provider": "local", "url": "clip.mp4", "secret": "x"}
                    ],
                }
            ],
        )
        for payload in invalid_manifests:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp_dir:
                manifest = Path(temp_dir) / "tasks.json"
                manifest.write_text(json.dumps(payload), encoding="utf-8")
                with patch("app.services.task.start") as start:
                    code = cli.run_cli(
                        ["--batch-file", str(manifest), "--stop-at", "script"]
                    )

                self.assertEqual(code, 2)
                start.assert_not_called()

    def test_batch_validates_every_task_has_subject_or_script_before_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "tasks.json"
            manifest.write_text(
                json.dumps([{"video_subject": "valid"}, {}]),
                encoding="utf-8",
            )
            with patch("app.services.task.start") as start:
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "script"]
                )

        self.assertEqual(code, 2)
        start.assert_not_called()

    def test_batch_rejects_invalid_subtitle_color_before_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": "invalid color",
                            "text_fore_color": "white",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch("app.services.task.start") as start:
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "script"]
                )

        self.assertEqual(code, 2)
        start.assert_not_called()

    def test_batch_rejects_invalid_video_clip_speed_before_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": "invalid speed",
                            "video_clip_speed": -1,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch("app.services.task.start") as start:
                code = cli.run_cli(
                    ["--batch-file", str(manifest), "--stop-at", "script"]
                )

        self.assertEqual(code, 2)
        start.assert_not_called()

    def test_later_null_runtime_field_prevents_every_batch_task_from_starting(self):
        for field_name in ("video_aspect", "video_concat_mode"):
            with self.subTest(field_name=field_name), tempfile.TemporaryDirectory() as temp_dir:
                manifest = Path(temp_dir) / "tasks.json"
                manifest.write_text(
                    json.dumps(
                        [
                            {"video_subject": "valid first task"},
                            {
                                "video_subject": "invalid later task",
                                field_name: None,
                            },
                        ]
                    ),
                    encoding="utf-8",
                )
                with patch("app.services.task.start") as start:
                    code = cli.run_cli(
                        [
                            "--batch-file",
                            str(manifest),
                            "--stop-at",
                            "video",
                            "--no-subtitle-enabled",
                        ]
                    )

                self.assertEqual(code, 2)
                start.assert_not_called()

    def test_batch_manifest_limits_size_and_task_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            oversized = Path(temp_dir) / "oversized.jsonl"
            oversized.write_bytes(b"x" * (cli._BATCH_FILE_MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "1 MiB limit"):
                cli._load_batch_manifest(str(oversized))

            too_many = Path(temp_dir) / "too-many.json"
            too_many.write_text(
                json.dumps(
                    [
                        {"video_subject": f"task {index}"}
                        for index in range(cli._BATCH_TASK_MAX_COUNT + 1)
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "the limit is"):
                cli._load_batch_manifest(str(too_many))

    def test_batch_jsonl_error_reports_source_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "invalid.jsonl"
            manifest.write_text(
                '{"video_subject": "valid"}\n{invalid json}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 2"):
                cli._load_batch_manifest(str(manifest))

    def test_manifest_relative_custom_audio_uses_manifest_directory(self):
        with (
            tempfile.TemporaryDirectory() as manifest_dir,
            tempfile.TemporaryDirectory() as working_dir,
        ):
            audio_file = Path(manifest_dir) / "voice.mp3"
            audio_file.write_bytes(b"audio")
            manifest = Path(manifest_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": "manifest audio",
                            "custom_audio_file": "voice.mp3",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            try:
                os.chdir(working_dir)
                with (
                    patch(
                        "app.services.task.start",
                        return_value={"state": 1, "audio_file": "ok"},
                    ) as start,
                    patch("app.utils.utils.get_uuid", return_value="task-one"),
                    redirect_stdout(io.StringIO()),
                ):
                    code = cli.run_cli(
                        ["--batch-file", str(manifest), "--stop-at", "audio"]
                    )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(code, 0)
        self.assertEqual(
            start.call_args.kwargs["params"].custom_audio_file,
            str(audio_file.resolve()),
        )

    def test_manifest_relative_local_material_uses_manifest_directory(self):
        with (
            tempfile.TemporaryDirectory() as manifest_dir,
            tempfile.TemporaryDirectory() as working_dir,
            tempfile.TemporaryDirectory() as managed_dir,
        ):
            source_file = Path(manifest_dir) / "clip.mp4"
            source_file.write_bytes(b"manifest video")
            manifest = Path(manifest_dir) / "tasks.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "video_subject": "manifest material",
                            "video_source": "local",
                            "video_materials": [
                                {
                                    "provider": "local",
                                    "url": "clip.mp4",
                                    "duration": 0,
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            try:
                os.chdir(working_dir)
                with (
                    patch(
                        "app.services.task.start",
                        return_value={"state": 1, "materials": ["ok"]},
                    ) as start,
                    patch("app.utils.utils.get_uuid", return_value="task-one"),
                    patch("app.utils.utils.storage_dir", return_value=managed_dir),
                    redirect_stdout(io.StringIO()),
                ):
                    code = cli.run_cli(
                        ["--batch-file", str(manifest), "--stop-at", "materials"]
                    )
            finally:
                os.chdir(old_cwd)

            prepared_file = Path(
                start.call_args.kwargs["params"].video_materials[0].url
            )
            self.assertEqual(code, 0)
            self.assertEqual(prepared_file.parent, Path(managed_dir))
            self.assertEqual(prepared_file.read_bytes(), b"manifest video")

    def test_global_relative_custom_audio_keeps_current_working_directory(self):
        with (
            tempfile.TemporaryDirectory() as manifest_dir,
            tempfile.TemporaryDirectory() as working_dir,
        ):
            audio_file = Path(working_dir) / "voice.mp3"
            audio_file.write_bytes(b"audio")
            manifest = Path(manifest_dir) / "tasks.json"
            manifest.write_text(
                json.dumps([{"video_subject": "global audio"}]),
                encoding="utf-8",
            )
            old_cwd = os.getcwd()
            try:
                os.chdir(working_dir)
                with (
                    patch(
                        "app.services.task.start",
                        return_value={"state": 1, "audio_file": "ok"},
                    ) as start,
                    patch("app.utils.utils.get_uuid", return_value="task-one"),
                    redirect_stdout(io.StringIO()),
                ):
                    code = cli.run_cli(
                        [
                            "--batch-file",
                            str(manifest),
                            "--custom-audio-file",
                            "voice.mp3",
                            "--stop-at",
                            "audio",
                        ]
                    )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(code, 0)
        self.assertEqual(
            start.call_args.kwargs["params"].custom_audio_file,
            str(audio_file.resolve()),
        )

    def test_help_documents_defaults_paths_stages_and_exit_codes(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            cli.parse_args(["--help"])

        self.assertEqual(cm.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("zh-CN-XiaoxiaoNeural-Female", help_text)
        self.assertIn("current working directory", help_text)
        self.assertIn("Pipeline stages:", help_text)
        self.assertIn("Batch manifests:", help_text)
        self.assertIn("JSONL", help_text)
        self.assertIn("exit with 2", help_text)

    def test_help_does_not_initialize_application_or_write_logs(self):
        """Команда справки не зависит от загрузки бизнес-конфигурации — её удобно смотреть вручную и собирать скриптами."""
        project_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            [sys.executable, str(project_root / "cli.py"), "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Generate MoneyPrinterTurbo videos", result.stdout)
        self.assertEqual(result.stderr, "")


class TestCliUiDefaults(unittest.TestCase):
    """
    Значения по умолчанию в CLI следуют за WebUI: сначала явные аргументы
    командной строки, затем сохранённые значения из [ui] в config.toml и только
    потом встроенные умолчания.
    """

    UI_CONFIG = {
        "font_name": "MicrosoftYaHeiBold.ttc",
        "text_fore_color": "#123456",
        "font_size": 48,
        "subtitle_background_enabled": True,
        "subtitle_background_color": "#654321",
        "rounded_subtitle_background": True,
        "voice_name": "gemini:Puck-Male",
    }

    def test_ui_config_supplies_subtitle_defaults(self):
        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, self.UI_CONFIG, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.font_name, "MicrosoftYaHeiBold.ttc")
        self.assertEqual(params.text_fore_color, "#123456")
        self.assertEqual(params.font_size, 48)
        self.assertEqual(params.text_background_color, "#654321")
        self.assertTrue(params.rounded_subtitle_background)

    def test_ui_config_supplies_voice_name(self):
        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, self.UI_CONFIG, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_name, "gemini:Puck-Male")

    def test_cli_flags_take_precedence_over_ui_config(self):
        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--font-name",
                "STHeitiLight.ttc",
                "--text-fore-color",
                "#AABBCC",
                "--font-size",
                "72",
                "--voice-name",
                "no-voice",
                "--no-rounded-subtitle-background",
            ]
        )

        with patch.dict(app_config.ui, self.UI_CONFIG, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.font_name, "STHeitiLight.ttc")
        self.assertEqual(params.text_fore_color, "#AABBCC")
        self.assertEqual(params.font_size, 72)
        self.assertEqual(params.voice_name, "no-voice")
        self.assertFalse(params.rounded_subtitle_background)

    def test_builtin_defaults_apply_when_ui_config_is_empty(self):
        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, {}, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.font_name, "STHeitiMedium.ttc")
        self.assertEqual(params.text_fore_color, "#FFFFFF")
        self.assertEqual(params.font_size, 60)
        self.assertFalse(params.text_background_color)
        self.assertFalse(params.rounded_subtitle_background)

    def test_ui_config_can_disable_subtitle_background(self):
        """
        Противоречивая конфигурация [ui] не должна прерывать CLI:
        `--no-subtitle-background-enabled` вместе с цветом — ошибка в аргументах
        командной строки, но как сохранённая настройка это лишь означает, что фон
        отключён.
        """
        ui_config = dict(self.UI_CONFIG)
        ui_config["subtitle_background_enabled"] = False

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertFalse(params.text_background_color)
        self.assertFalse(params.rounded_subtitle_background)

    def test_unusable_ui_config_values_fall_back_to_builtin_defaults(self):
        """Повреждённый config.toml не должен приводить к traceback."""
        ui_config = {
            "font_size": "sechzig",
            "text_fore_color": 42,
            "font_name": "",
            "voice_name": None,
        }

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.font_size, 60)
        self.assertEqual(params.text_fore_color, "#FFFFFF")
        self.assertEqual(params.font_name, "STHeitiMedium.ttc")
        self.assertEqual(params.voice_name, "zh-CN-XiaoxiaoNeural-Female")

    def test_saved_no_voice_mode_disables_tts(self):
        """
        WebUI сохраняет «без озвучки» как отдельный voice_mode, попутно оставляя
        последний реально выбранный пользователем голос — чтобы можно было
        вернуться к автоозвучке. CLI обязан следовать этой модели, иначе он снова
        включит TTS и может вызвать запрос к платному поставщику.
        """
        ui_config = {"voice_mode": "none", "voice_name": "gemini:Puck-Male"}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_name, "no-voice")

    def test_explicit_voice_name_overrides_saved_no_voice_mode(self):
        """Явно указанный в командной строке голос имеет наивысший приоритет, и сохранённый режим «без озвучки» его не перекрывает."""
        ui_config = {"voice_mode": "none", "voice_name": "gemini:Puck-Male"}

        args = cli.parse_args(
            ["--video-subject", "test", "--voice-name", "zh-CN-XiaoxiaoNeural-Female"]
        )

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_name, "zh-CN-XiaoxiaoNeural-Female")

    def test_saved_tts_mode_keeps_saved_voice(self):
        """Когда voice_mode — автоозвучка, сохранённый голос по-прежнему применяется."""
        ui_config = {"voice_mode": "tts", "voice_name": "gemini:Puck-Male"}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_name, "gemini:Puck-Male")

    def test_enabling_background_without_color_keeps_saved_color(self):
        """
        При передаче одного лишь --subtitle-background-enabled пользователь не
        переопределял цвет, поэтому нужно взять цвет, сохранённый в WebUI, а не
        откатываться к чёрному фону.
        """
        ui_config = {"subtitle_background_color": "#654321"}

        args = cli.parse_args(
            ["--video-subject", "test", "--subtitle-background-enabled"]
        )

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.text_background_color, "#654321")

    def test_enabling_background_falls_back_to_default_without_saved_color(self):
        """Если пригодного сохранённого цвета нет, простое включение фона откатывается к фону по умолчанию."""
        args = cli.parse_args(
            ["--video-subject", "test", "--subtitle-background-enabled"]
        )

        with patch.dict(app_config.ui, {}, clear=True):
            params = cli.build_video_params(args)

        self.assertIs(params.text_background_color, True)

    def test_explicit_background_color_overrides_saved_color(self):
        """Явно указанный в командной строке цвет фона приоритетнее сохранённого значения."""
        ui_config = {"subtitle_background_color": "#654321"}

        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--subtitle-background-enabled",
                "--subtitle-background-color",
                "#ABCDEF",
            ]
        )

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.text_background_color, "#ABCDEF")

    def test_saved_upload_mode_disables_tts(self):
        """
        Режим загрузки собственного аудио тоже означает «не нужна автоозвучка», а
        путь к файлу в [ui] не сохраняется, и CLI не может воспроизвести эту
        загрузку. Использование сохранённого голоса здесь молча вызвало бы
        платный запрос к TTS, поэтому режим, как и «без озвучки», отображается в
        no-voice; если озвучка нужна, её задают явным --voice-name.
        """
        ui_config = {"voice_mode": "upload", "voice_name": "gemini:Puck-Male"}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_name, "no-voice")

    def test_explicit_voice_name_overrides_saved_upload_mode(self):
        """Явно указанный в командной строке голос приоритетнее и сохранённого режима загрузки."""
        ui_config = {"voice_mode": "upload", "voice_name": "gemini:Puck-Male"}

        args = cli.parse_args(
            ["--video-subject", "test", "--voice-name", "mimo:Female"]
        )

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_name, "mimo:Female")

    def test_saved_color_alone_enables_background(self):
        """
        WebUI всегда пишет переключатель и цвет вместе, поэтому сохранённый только
        цвет — признак ручной правки конфигурации. Сам факт сохранённого цвета
        говорит, что фон пользователю нужен, и обрабатывается как включённый.
        """
        ui_config = {"subtitle_background_color": "#654321"}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.text_background_color, "#654321")

    def test_ui_config_supplies_voice_and_stroke_defaults(self):
        """Громкость озвучки, скорость речи, обводка и переключатель субтитров тоже хранятся в [ui] и должны наследоваться вместе с остальным."""
        ui_config = {
            "voice_volume": 0.5,
            "voice_rate": 1.3,
            "stroke_color": "#112233",
            "stroke_width": 2.5,
            "subtitle_enabled": False,
        }

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_volume, 0.5)
        self.assertEqual(params.voice_rate, 1.3)
        self.assertEqual(params.stroke_color, "#112233")
        self.assertEqual(params.stroke_width, 2.5)
        self.assertFalse(params.subtitle_enabled)

    def test_cli_flags_take_precedence_over_saved_voice_and_stroke(self):
        """Явно переданные в командной строке значения приоритетнее этих сохранённых."""
        ui_config = {
            "voice_volume": 0.5,
            "voice_rate": 1.3,
            "stroke_color": "#112233",
            "stroke_width": 2.5,
            "subtitle_enabled": False,
        }

        args = cli.parse_args(
            [
                "--video-subject",
                "test",
                "--voice-volume",
                "0.9",
                "--voice-rate",
                "1.1",
                "--stroke-color",
                "#AABBCC",
                "--stroke-width",
                "3.5",
                "--subtitle-enabled",
            ]
        )

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_volume, 0.9)
        self.assertEqual(params.voice_rate, 1.1)
        self.assertEqual(params.stroke_color, "#AABBCC")
        self.assertEqual(params.stroke_width, 3.5)
        self.assertTrue(params.subtitle_enabled)

    def test_saved_integers_are_accepted_for_float_fields(self):
        """Целое число в TOML — тоже допустимые громкость и скорость: его нужно преобразовать и использовать, а не отбрасывать."""
        ui_config = {"voice_volume": 1, "voice_rate": 2, "stroke_width": 3}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_volume, 1.0)
        self.assertEqual(params.voice_rate, 2.0)
        self.assertEqual(params.stroke_width, 3.0)

    def test_saved_values_out_of_range_fall_back_to_builtin_defaults(self):
        """
        Сохранённые значения проверяются по тем же правилам, что и аргументы
        командной строки: громкость не может быть отрицательной, скорость речи
        обязана быть положительной, цвет — в формате #RRGGBB. Недопустимые
        значения откатываются к встроенным умолчаниям.
        """
        ui_config = {
            "voice_volume": -1.0,
            "voice_rate": 0,
            "stroke_color": "notacolor",
            "stroke_width": -2.0,
            "font_size": 0,
        }

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.voice_volume, 1.0)
        self.assertEqual(params.voice_rate, 1.0)
        self.assertEqual(params.stroke_color, "#000000")
        self.assertEqual(params.stroke_width, 1.5)
        self.assertEqual(params.font_size, 60)

    def test_stop_at_subtitle_overrides_saved_subtitle_disabled(self):
        """
        `--stop-at subtitle` прямо требует сгенерировать субтитры. Сохранённое
        выключенное состояние не должно превращать этот этап в пустую операцию и
        не должно давать ошибку аргументов, как явный --no-subtitle-enabled.
        """
        ui_config = {"subtitle_enabled": False}

        args = cli.parse_args(
            ["--video-subject", "test", "--stop-at", "subtitle"]
        )

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertTrue(params.subtitle_enabled)

    def test_explicit_no_subtitle_still_rejects_stop_at_subtitle(self):
        """Явное отключение субтитров вместе с `--stop-at subtitle` по-прежнему остаётся ошибкой аргументов."""
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(
                [
                    "--video-subject",
                    "test",
                    "--stop-at",
                    "subtitle",
                    "--no-subtitle-enabled",
                ]
            )

        self.assertEqual(cm.exception.code, 2)

    def test_saved_subtitle_position_is_validated_and_applied(self):
        """
        Позиция субтитров раньше зависела только от значения поля VideoParams по
        умолчанию, а оно вычисляется один раз при импорте модуля — его нельзя ни
        проверить, ни подменить в тесте. Теперь она разбирается явно, как и
        остальные поля.
        """
        ui_config = {"subtitle_position": "custom", "custom_position": 42.5}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.subtitle_position, "custom")
        self.assertEqual(params.custom_position, 42.5)

    def test_unusable_saved_subtitle_position_falls_back(self):
        """Сохранённая позиция за пределами допустимого диапазона откатывается к встроенному умолчанию."""
        ui_config = {"subtitle_position": "diagonal", "custom_position": 150.0}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertEqual(params.subtitle_position, "bottom")
        self.assertEqual(params.custom_position, 70.0)

    def test_invalid_saved_background_color_with_saved_enable_flag(self):
        """
        Сохранённый цвет фона тоже проверяется по формату #RRGGBB. Оставшись в
        VideoParams, недопустимое значение при отрисовке станет чёрным, а проверка
        совпадения цветов будет сравнивать исходную некорректную строку — и чёрный
        текст на чёрном фоне останется незамеченным.
        """
        ui_config = {
            "subtitle_background_enabled": True,
            "subtitle_background_color": "not-a-color",
        }

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertIs(params.text_background_color, True)

    def test_invalid_saved_background_color_with_explicit_enable_flag(self):
        """При явном включении фона недопустимый сохранённый цвет тоже откатывается к фону по умолчанию."""
        ui_config = {"subtitle_background_color": "not-a-color"}

        args = cli.parse_args(
            ["--video-subject", "test", "--subtitle-background-enabled"]
        )

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertIs(params.text_background_color, True)

    def test_invalid_saved_background_color_without_enable_flag(self):
        """
        Если сохранён только недопустимый цвет и нет переключателя, выводить
        отсюда, что фон нужен, нельзя — сохраняется выключенное состояние
        VideoParams по умолчанию.
        """
        ui_config = {"subtitle_background_color": "not-a-color"}

        args = cli.parse_args(["--video-subject", "test"])

        with patch.dict(app_config.ui, ui_config, clear=True):
            params = cli.build_video_params(args)

        self.assertFalse(params.text_background_color)


if __name__ == "__main__":
    unittest.main()
