from typing import List, Optional

from pydantic import BaseModel

from app.schemas.common import FileInfo, SizeInfo
from app.schemas.detect import DetectData


class GenerateData(BaseModel):
    taskId: str
    previewPath: str
    previewUrl: str
    hdPath: str
    hdUrl: str
    backgroundColor: str
    size: SizeInfo
    width: int
    height: int
    warnings: List[str]
    detect: DetectData
    intermediateFiles: Optional[dict[str, FileInfo]] = None
