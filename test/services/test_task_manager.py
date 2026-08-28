import json
import unittest
from unittest.mock import MagicMock, patch

from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import task as task_service


class TestInMemoryTaskManager(unittest.TestCase):
    def test_queue_operations_preserve_task_payload(self):
        """Очередь в памяти сохраняет функцию, позиционные и именованные аргументы, не меняя содержимое задачи."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        task = {"func": len, "args": ([1, 2],), "kwargs": {}}

        manager.enqueue(task)

        self.assertFalse(manager.is_queue_empty())
        self.assertEqual(manager.queue_size(), 1)
        self.assertEqual(manager.dequeue(), task)
        self.assertTrue(manager.is_queue_empty())

    def test_add_task_rejects_only_after_queue_limit(self):
        """Когда слоты параллелизма исчерпаны, задачи встают в очередь до её предела, и только за пределом возвращается явная ошибка."""
        manager = InMemoryTaskManager(max_concurrent_tasks=0, max_queued_tasks=1)

        manager.add_task(len, [1])

        with self.assertRaises(TaskQueueFullError):
            manager.add_task(len, [2])

    def test_add_task_reserves_slot_before_background_thread_runs(self):
        """
        Слот параллелизма обязан заниматься до старта потока: даже если
        замоканный поток ещё не дошёл до run_task, второй запрос должен попасть в
        очередь и не пробить max_concurrent_tasks.
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)

        with patch.object(manager, "execute_task") as execute_task:
            manager.add_task(len, [1])
            manager.add_task(len, [2])

        self.assertEqual(manager.current_tasks, 1)
        execute_task.assert_called_once_with(len, [1])
        self.assertEqual(manager.queue_size(), 1)

    def test_add_task_rolls_back_slot_when_thread_cannot_start(self):
        """Неудачный старт потока не должен навсегда занимать слот параллелизма, а исключение по-прежнему отдаётся вызывающей стороне."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1)

        with patch.object(
            manager,
            "execute_task",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                manager.add_task(len, [1])

        self.assertEqual(manager.current_tasks, 0)

    def test_task_done_starts_next_queued_task(self):
        """По завершении текущей задачи слот освобождается, и следующая задача из очереди планируется немедленно."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=2)
        manager.current_tasks = 1
        manager.enqueue({"func": len, "args": ([1, 2],), "kwargs": {}})

        with patch.object(manager, "execute_task") as execute_task:
            manager.task_done()

        self.assertEqual(manager.current_tasks, 1)
        execute_task.assert_called_once_with(len, [1, 2])
        self.assertTrue(manager.is_queue_empty())

    def test_task_done_requeues_task_when_thread_cannot_start(self):
        """Если после извлечения из очереди поток не стартовал, слот возвращается, а задача кладётся обратно в очередь, чтобы не потеряться."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)
        manager.current_tasks = 1
        queued_task = {"func": len, "args": ([1, 2],), "kwargs": {}}
        manager.enqueue(queued_task)

        with patch.object(
            manager,
            "execute_task",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                manager.task_done()

        self.assertEqual(manager.current_tasks, 0)
        self.assertEqual(manager.dequeue(), queued_task)

    def test_run_task_releases_slot_after_failure(self):
        """Когда функция задачи бросает исключение, finally всё равно обязан освободить слот, иначе очередь заблокируется навсегда."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1)
        manager.current_tasks = 1

        with patch.object(manager, "task_done") as task_done:
            with self.assertRaisesRegex(RuntimeError, "task failed"):
                manager.run_task(MagicMock(side_effect=RuntimeError("task failed")))

        self.assertEqual(manager.current_tasks, 1)
        task_done.assert_called_once_with()

    def test_check_queue_handles_dequeue_returning_none(self):
        """
        dequeue() может вернуть None, внутренне пропустив все задачи в очереди,
        которые больше не проходят текущую валидацию, — даже если перед вызовом
        check_queue функция is_queue_empty() возвращала False. check_queue не
        вправе предполагать, что dequeue обязательно отдаст пригодную задачу:
        иначе он упадёт на task_info["func"].
        """
        manager = InMemoryTaskManager(max_concurrent_tasks=1, max_queued_tasks=1)

        with patch.object(manager, "is_queue_empty", return_value=False), patch.object(
            manager, "dequeue", return_value=None
        ), patch.object(manager, "execute_task") as execute_task:
            manager.check_queue()

        execute_task.assert_not_called()
        self.assertEqual(manager.current_tasks, 0)

    def test_execute_task_starts_background_thread(self):
        """Точка запуска задачи обязана стартовать поток и передать в run_task все аргументы функции целиком."""
        manager = InMemoryTaskManager(max_concurrent_tasks=1)
        fake_thread = MagicMock()

        with patch(
            "app.controllers.manager.base_manager.threading.Thread",
            return_value=fake_thread,
        ) as thread:
            manager.execute_task(len, [1, 2])

        thread.assert_called_once_with(
            target=manager.run_task,
            args=(len, [1, 2]),
            kwargs={},
        )
        fake_thread.start.assert_called_once_with()


class TestRedisTaskManager(unittest.TestCase):
    def setUp(self):
        self.redis_client = MagicMock()
        patcher = patch(
            "app.controllers.manager.redis_manager.redis.Redis.from_url",
            return_value=self.redis_client,
        )
        self.addCleanup(patcher.stop)
        from_url = patcher.start()
        self.manager = RedisTaskManager(
            max_concurrent_tasks=1,
            redis_url="redis://localhost:6379/0",
            max_queued_tasks=3,
        )
        from_url.assert_called_once_with("redis://localhost:6379/0")

    def test_enqueue_serializes_video_params_without_mutating_task(self):
        """
        В Redis можно хранить только JSON, поэтому VideoParams превращается в
        словарь, но у исходной задачи модель должна остаться: побочные эффекты
        сериализации не должны влиять на логи, повторы и последующее чтение
        вызывающей стороной.
        """
        params = VideoParams(video_subject="Coffee")
        task = {
            "func": task_service.start,
            "args": (),
            "kwargs": {"task_id": "task-1", "params": params},
        }

        self.manager.enqueue(task)

        self.assertIs(task["kwargs"]["params"], params)
        queue_name, payload = self.redis_client.rpush.call_args.args
        decoded = json.loads(payload)
        self.assertEqual(queue_name, "task_queue")
        self.assertEqual(decoded["func"], "start")
        self.assertEqual(decoded["kwargs"]["task_id"], "task-1")
        self.assertEqual(decoded["kwargs"]["params"]["video_subject"], "Coffee")

    def test_dequeue_restores_function_and_video_params(self):
        """Извлечённая из Redis задача восстанавливает вызываемую функцию и модель VideoParams."""
        payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-1",
                "params": VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ),
            },
        }
        self.redis_client.lpop.return_value = json.dumps(payload)

        task = self.manager.dequeue()

        self.redis_client.lpop.assert_called_once_with("task_queue")
        self.assertIs(task["func"], task_service.start)
        self.assertIsInstance(task["kwargs"]["params"], VideoParams)
        self.assertEqual(task["kwargs"]["params"].video_subject, "Coffee")

    def test_empty_queue_and_size_use_redis_length(self):
        """Проверка пустоты и длина очереди напрямую отражают текущую длину списка в Redis."""
        self.redis_client.lpop.return_value = None
        self.redis_client.llen.side_effect = [0, 2]

        self.assertIsNone(self.manager.dequeue())
        self.assertTrue(self.manager.is_queue_empty())
        self.assertEqual(self.manager.queue_size(), 2)

    def test_dequeue_skips_task_that_fails_current_validation(self):
        """
        Задача могла попасть в очередь до ужесточения правил валидации (например,
        когда video_count разрешалось равным 0). lpop разрушающая: если
        пересборка VideoParams не удалась, задача уже безвозвратно удалена из
        Redis, и делать вид, что она там, нельзя. dequeue не должен пробрасывать
        ошибку валидации вызывающей стороне — так держатель лока упадёт, задача
        потеряется, а в логах ничего не останется. Вместо этого нужно пропустить
        её и перейти к следующей в очереди, пока не найдётся пригодная задача или
        очередь действительно не опустеет.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        valid_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-valid",
                "params": VideoParams(video_subject="Tea").model_dump(
                    warnings=False
                ),
            },
        }
        self.redis_client.lpop.side_effect = [
            json.dumps(stale_payload),
            json.dumps(valid_payload),
        ]

        task = self.manager.dequeue()

        self.assertEqual(self.redis_client.lpop.call_count, 2)
        self.assertEqual(task["kwargs"]["task_id"], "task-valid")
        self.assertIsInstance(task["kwargs"]["params"], VideoParams)
        self.assertEqual(task["kwargs"]["params"].video_subject, "Tea")

    def test_dequeue_returns_none_when_every_queued_task_is_stale(self):
        """Если все оставшиеся задачи отброшены текущими правилами валидации, возвращается None, а не исключение."""
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": -1},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        self.assertIsNone(self.manager.dequeue())
        self.assertEqual(self.redis_client.lpop.call_count, 2)

    def test_dequeue_marks_stale_task_failed_instead_of_leaving_it_processing(self):
        """
        Запись статуса создаётся до постановки в очередь и по умолчанию равна
        processing. Если внутри dequeue просто пропустить и выбросить элемент
        очереди, не обновив запись статуса, задача навсегда останется
        «выполняющейся» в API и WebUI. Помечать её ошибкой нужно через patch_task,
        а не update_task: тогда при уже удалённой пользователем задаче мы не
        создадим её запись статуса заново.
        """
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-stale",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = True
            result = self.manager.dequeue()

        self.assertIsNone(result)
        state.patch_task.assert_called_once()
        call_args = state.patch_task.call_args
        self.assertEqual(call_args.args[0], "task-stale")
        self.assertEqual(call_args.kwargs["state"], const.TASK_STATE_FAILED)
        self.assertEqual(call_args.kwargs["failed_stage"], "dequeue")
        self.assertIn("video_count", call_args.kwargs["error"])

    def test_dequeue_does_not_recreate_state_for_already_deleted_task(self):
        """Для уже удалённой задачи patch_task возвращает False, и dequeue не должен считать это ошибкой."""
        stale_payload = {
            "func": "start",
            "args": [],
            "kwargs": {
                "task_id": "task-deleted",
                "params": {**VideoParams(video_subject="Coffee").model_dump(
                    warnings=False
                ), "video_count": 0},
            },
        }
        self.redis_client.lpop.side_effect = [json.dumps(stale_payload), None]

        with patch("app.controllers.manager.redis_manager.sm.state") as state:
            state.patch_task.return_value = False
            result = self.manager.dequeue()

        self.assertIsNone(result)
        state.patch_task.assert_called_once()
        state.update_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
