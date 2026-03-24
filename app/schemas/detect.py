from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import FaceBox


class DetectIssue(BaseModel):
    code: str
    message: str
    severity: Literal['WARNING', 'FAILED']


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
    status: Literal['PASSED', 'WARNING', 'FAILED']
    resultLevel: Literal['PASSED', 'WARNING', 'FAILED']
    canGenerate: bool
    reasons: List[DetectReason]
    suggestions: List[str] = Field(default_factory=list)
    reasonCodes: List[str]
    warnings: List[str]
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
