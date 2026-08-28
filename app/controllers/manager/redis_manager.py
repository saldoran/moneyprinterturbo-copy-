import json
from typing import Dict

import redis
from loguru import logger
from pydantic import ValidationError

from app.controllers.manager.base_manager import TaskManager
from app.models import const
from app.models.schema import VideoParams
from app.services import state as sm
from app.services import task as tm

FUNC_MAP = {
    "start": tm.start,
    # 'start_test': tm.start_test
}


class RedisTaskManager(TaskManager):
    def __init__(
        self,
        max_concurrent_tasks: int,
        redis_url: str,
        max_queued_tasks: int = 100,
    ):
        self.redis_client = redis.Redis.from_url(redis_url)
        super().__init__(max_concurrent_tasks, max_queued_tasks=max_queued_tasks)

    def create_queue(self):
        return "task_queue"

    def enqueue(self, task: Dict):
        task_with_serializable_params = task.copy()
        # task.copy() копирует только верхний словарь. Правка вложенного kwargs на месте
        # заменила бы VideoParams, которым владеет вызывающая сторона, на dict. Логи
        # и повторы могут ещё читать исходную задачу, поэтому kwargs копируется отдельно — сериализация не должна давать неожиданных побочных эффектов.
        task_kwargs = task.get("kwargs", {})
        task_with_serializable_params["kwargs"] = task_kwargs.copy()

        if "params" in task_kwargs and isinstance(task_kwargs["params"], VideoParams):
            task_with_serializable_params["kwargs"]["params"] = task_kwargs[
                "params"
            ].model_dump(warnings=False)

        # Преобразуем объект функции в её имя
        task_with_serializable_params["func"] = task["func"].__name__
        self.redis_client.rpush(self.queue, json.dumps(task_with_serializable_params))

    def dequeue(self):
        # Цикл, а не однократное извлечение: задача могла удовлетворять правилам
        # валидации VideoParams на момент постановки в очередь, а между двумя
        # развёртываниями правила ужесточились (например, добавилось ge=1). lpop —
        # разрушающая операция: извлечённое обратно не вернуть. Если ошибка валидации
        # обнаружится только при пересборке VideoParams, задача уже навсегда удалена из
        # очереди, и делать вид, что она там, нельзя. Вместо того чтобы пробрасывать
        # исключение наверх и ронять держателя лока вместе с потерянной задачей,
        # отбрасываем её здесь и переходим к следующей, сохраняя контракт
        # «вернули пригодную задачу либо очередь действительно пуста».
        while True:
            task_json = self.redis_client.lpop(self.queue)
            if not task_json:
                return None

            task_info = json.loads(task_json)
            # Преобразуем имя функции обратно в объект функции
            task_info["func"] = FUNC_MAP[task_info["func"]]

            if "params" in task_info["kwargs"] and isinstance(
                task_info["kwargs"]["params"], dict
            ):
                try:
                    task_info["kwargs"]["params"] = VideoParams(
                        **task_info["kwargs"]["params"]
                    )
                except ValidationError as e:
                    logger.error(
                        "dropping queued task with params that fail current "
                        f"VideoParams validation (queued under an older, more "
                        f"permissive schema, or corrupted): {e}"
                    )
                    # Запись статуса создаётся до постановки в очередь и по умолчанию равна
                    # processing. Если просто выбросить элемент очереди, не тронув запись статуса,
                    # API и WebUI будут вечно показывать задачу выполняющейся и никогда — упавшей.
                    # Используем patch_task, а не update_task: если пользователь уже удалил задачу, мы не создадим её заново.
                    task_id = task_info["kwargs"].get("task_id")
                    if task_id:
                        sm.state.patch_task(
                            task_id,
                            state=const.TASK_STATE_FAILED,
                            failed_stage="dequeue",
                            error=f"discarded stale queued task: {e}",
                        )
                    continue

            return task_info

    def is_queue_empty(self):
        return self.redis_client.llen(self.queue) == 0

    def queue_size(self):
        return self.redis_client.llen(self.queue)
