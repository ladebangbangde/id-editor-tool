from threading import Lock

from app.schemas.task_status import StageCode, TaskStatusData, utc_now


class TaskStatusService:
    """In-memory task status store. Can be replaced by Redis/DB in production."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[str, TaskStatusData] = {}

    def update(
        self,
        *,
        task_id: str,
        stage_code: StageCode,
        progress: int,
        message: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> TaskStatusData:
        status = TaskStatusData(
            taskId=task_id,
            stageCode=stage_code,
            progress=progress,
            message=message,
            updatedAt=utc_now(),
            errorCode=error_code,
            errorMessage=error_message,
        )
        with self._lock:
            self._store[task_id] = status
        return status

    def get(self, task_id: str) -> TaskStatusData | None:
        with self._lock:
            return self._store.get(task_id)
