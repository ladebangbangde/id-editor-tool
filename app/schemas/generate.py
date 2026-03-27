from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import FileInfo, SizeInfo
from app.schemas.detect import DetectData, DetectIssue


class GenerateCandidate(BaseModel):
    candidateId: str
    engineKey: str
    label: str
    imagePath: str
    imageUrl: str
    width: int
    height: int
    format: str = 'PNG'
    previewPath: str = ''
    previewUrl: str = ''
    qualityStatus: str = 'PASS'
    qualityMessage: str = '照片质量良好，可直接处理'
    outputQualityStatus: str = 'PASS'
    outputQualityMessage: str = '输出成片质量正常'
    outputReasonCodes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    primaryIssue: Optional[str] = None
    primaryMessage: Optional[str] = None
    safeToSubmit: bool = True
    debugInfo: Optional[dict[str, Any]] = None


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
    warnings: List[str] = Field(default_factory=list, description='用户可见提示，仅包含自然中文文案，禁止展示内部调试信息')
    detect: DetectData
    detectSummary: DetectData
    candidates: List[GenerateCandidate] = Field(default_factory=list)
    selectedCandidateId: Optional[str] = None
    requireUserSelection: bool = True
    primaryIssue: Optional[str] = None
    primaryMessage: Optional[str] = Field(default=None, description='用户可见主提示，必须是自然中文')
    secondaryWarnings: List[str] = Field(default_factory=list, description='用户可见补充提示，必须是自然中文')
    qualityStatus: str = 'PASS'
    qualityMessage: str = Field(default='照片质量良好，可直接处理', description='用户可见质量提示，必须是自然中文')
    outputQualityStatus: str = 'PASS'
    outputQualityMessage: str = Field(default='输出成片质量正常', description='用户可见输出质量提示，必须是自然中文')
    outputReasonCodes: List[str] = Field(default_factory=list)
    allowPreviewSave: bool = True
    allowHdSave: bool = True
    previewWidth: int = 0
    previewHeight: int = 0
    previewFormat: str = 'JPEG'
    previewQuality: int = 75
    hdWidth: int = 0
    hdHeight: int = 0
    hdFormat: str = 'PNG'
    hdQuality: int = 100
    intermediateFiles: Optional[dict[str, FileInfo]] = None
    debugInfo: Optional[dict[str, Any]] = Field(default=None, description='内部调试字段，前端主展示区默认不渲染')
    processStatus: str = 'generated'
    processMessage: str = '图片已生成'
    complianceStatus: str = 'passed'
    complianceMessage: str = '满足证件照合规要求'
    complianceDetails: List[DetectIssue] = Field(default_factory=list)
    safeToSubmit: bool = True
    outputQualityMetrics: dict[str, float] = Field(default_factory=dict)


class GenerateSelectionRequest(BaseModel):
    taskId: str
    candidateId: str


class GenerateSelectionData(BaseModel):
    taskId: str
    candidateId: str
    imagePath: str
    imageUrl: str
    previewPath: str = ''
    previewUrl: str = ''
    status: str = 'selected'
    message: str = '已确认所选图片'
