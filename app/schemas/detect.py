from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import FaceBox


class DetectIssue(BaseModel):
    code: str
    message: str
    severity: Literal['WARNING', 'FAIL']


class DetectReason(BaseModel):
    code: str
    title: str
    detail: str


class DetectData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hasFace: bool
    faceCount: int
    width: int
    height: int
    pass_: bool = Field(alias='pass')
    recommended: bool
    status: Literal['PASS', 'WARNING', 'FAIL']
    resultLevel: Literal['PASS', 'WARNING', 'FAIL']
    canGenerate: bool
    reasons: List[DetectReason]
    suggestions: List[str] = Field(default_factory=list)
    reasonCodes: List[str]
    warnings: List[str] = Field(default_factory=list, description='用户可见提示，仅包含自然中文文案，禁止展示内部调试信息')
    warningCodes: List[str]
    issues: List[DetectIssue]
    faceBoxes: List[FaceBox]
    warning: Optional[str] = None
    blurScore: Optional[float] = None
    occlusionDetected: bool = False
    occlusionAreas: List[str] = Field(default_factory=list)
    poseAccepted: bool = True
    landmarkStable: bool = True
    compositionAccepted: bool = True
    metrics: Dict[str, float] = Field(default_factory=dict)
    primaryIssue: Optional[str] = None
    primaryMessage: Optional[str] = Field(default=None, description='用户可见主提示，必须是自然中文')
    secondaryWarnings: List[str] = Field(default_factory=list, description='用户可见补充提示，必须是自然中文')
    qualityStatus: Literal['PASS', 'WARNING', 'FAIL'] = 'PASS'
    qualityMessage: str = Field(default='照片质量良好，可直接处理', description='用户可见质量提示，必须是自然中文')
    processStatus: Literal['success'] = 'success'
    processMessage: str = '图片检测流程已完成'
    complianceStatus: Literal['passed', 'warning', 'failed'] = 'passed'
    complianceMessage: str = '满足证件照合规要求'
    complianceDetails: List[DetectIssue] = Field(default_factory=list)
