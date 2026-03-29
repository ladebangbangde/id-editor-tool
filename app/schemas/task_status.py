from datetime import datetime

from pydantic import BaseModel


class TaskStatusData(BaseModel):
    taskId: str
    stageCode: str
    progress: int
    startedAt: datetime
    updatedAt: datetime
    message: str
    errorCode: str | None = None
    errorMessage: str | None = None
