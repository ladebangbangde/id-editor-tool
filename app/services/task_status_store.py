from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock

from app.core.exceptions import InvalidArgumentError
from app.schemas.task_status import TaskStatusData
from app.utils.file_naming import build_task_id


@dataclass(frozen=True)
class StageDefinition:
    code: str
    progress: int


STAGE_CHECKING = StageDefinition(code='checking', progress=20)
STAGE_ADJUSTING = StageDefinition(code='adjusting', progress=45)
STAGE_GENERATING = StageDefinition(code='generating', progress=75)
STAGE_FINALIZING = StageDefinition(code='finalizing', progress=92)
STAGE_SUCCESS = StageDefinition(code='success', progress=100)
STAGE_FAILED = StageDefinition(code='failed', progress=100)


class TaskStatusStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: dict[str, TaskStatusData] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def create_task(self, task_id: str | None = None, message: str = '任务已创建，等待开始') -> TaskStatusData:
        real_task_id = (task_id or '').strip() or build_task_id('task')
        now = self._now()
        status = TaskStatusData(
            taskId=real_task_id,
            stageCode='created',
            progress=0,
            startedAt=now,
            updatedAt=now,
            message=message,
            errorCode=None,
            errorMessage=None,
        )
        with self._lock:
            self._tasks[real_task_id] = status
        return status

    def get_status(self, task_id: str) -> TaskStatusData:
        with self._lock:
            status = self._tasks.get(task_id)
        if status is None:
            raise InvalidArgumentError(f'未找到任务状态: {task_id}')
        return status

    def update_stage(
        self,
        task_id: str,
        stage: StageDefinition,
        message: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> TaskStatusData:
        with self._lock:
            current = self._tasks.get(task_id)
            if current is None:
                now = self._now()
                current = TaskStatusData(
                    taskId=task_id,
                    stageCode='created',
                    progress=0,
                    startedAt=now,
                    updatedAt=now,
                    message='任务已创建，等待开始',
                    errorCode=None,
                    errorMessage=None,
                )
            updated = TaskStatusData(
                taskId=task_id,
                stageCode=stage.code,
                progress=stage.progress,
                startedAt=current.startedAt,
                updatedAt=self._now(),
                message=message,
                errorCode=error_code,
                errorMessage=error_message,
            )
            self._tasks[task_id] = updated
            return updated


@lru_cache
def get_task_status_store() -> TaskStatusStore:
    return TaskStatusStore()
