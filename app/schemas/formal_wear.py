from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.detect import DetectData


class FormalWearData(BaseModel):
    taskId: str
    previewUrl: str
    hdUrl: str
    gender: Optional[str] = None
    style: Optional[str] = None
    color: str
    warnings: List[str]
    previewPath: str = ''
    hdPath: str = ''
    previewWidth: int = 0
    previewHeight: int = 0
    previewFormat: str = 'JPEG'
    previewQuality: int = 75
    hdWidth: int = 0
    hdHeight: int = 0
    hdFormat: str = 'PNG'
    hdQuality: int = 100
    primaryIssue: Optional[str] = None
    primaryMessage: Optional[str] = None
    secondaryWarnings: List[str] = Field(default_factory=list)
    qualityStatus: str = 'PASS'
    qualityMessage: str = '照片质量良好，可直接处理'
    detectSummary: Optional[DetectData] = None
