from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class StageCode(str, Enum):
    CHECKING = 'checking'
    ADJUSTING = 'adjusting'
    ENHANCING = 'enhancing'
    FINALIZING = 'finalizing'
    SUCCESS = 'success'
    FAILED = 'failed'


class TaskStatusData(BaseModel):
    taskId: str
    stageCode: StageCode
    progress: int = Field(ge=0, le=100)
    message: str | None = None
    updatedAt: datetime
    errorCode: str | None = None
    errorMessage: str | None = None


class TaskStatusUpdate(BaseModel):
    taskId: str
    stageCode: StageCode
    progress: int
    message: str | None = None
    errorCode: str | None = None
    errorMessage: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
