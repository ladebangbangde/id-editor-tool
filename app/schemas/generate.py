from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import FileInfo, SizeInfo
from app.schemas.detect import DetectData, DetectIssue


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
    detectSummary: DetectData
    primaryIssue: Optional[str] = None
    primaryMessage: Optional[str] = None
    secondaryWarnings: List[str] = Field(default_factory=list)
    qualityStatus: str = 'PASS'
    qualityMessage: str = '照片质量良好，可直接处理'
    previewWidth: int = 0
    previewHeight: int = 0
    previewFormat: str = 'JPEG'
    previewQuality: int = 75
    hdWidth: int = 0
    hdHeight: int = 0
    hdFormat: str = 'PNG'
    hdQuality: int = 100
    intermediateFiles: Optional[dict[str, FileInfo]] = None
    processStatus: str = 'generated'
    processMessage: str = '图片已生成'
    complianceStatus: str = 'passed'
    complianceMessage: str = '满足证件照合规要求'
    complianceDetails: List[DetectIssue] = Field(default_factory=list)
    safeToSubmit: bool = True
