from typing import Optional

from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    errorCode: Optional[str] = None


class DetectResult(BaseModel):
    imageId: str
    hasFace: bool
    faceCount: int
    pass_: bool = Field(alias='pass')
    reasons: list[str]
    blurScore: Optional[float] = None
    poseValid: bool
    occlusionDetected: bool
    message: str
    imageWidth: int
    imageHeight: int
    primaryFaceBox: Optional[dict] = None


class GenerateIdPhotoResult(BaseModel):
    imageId: str
    previewUrl: str
    hdUrl: str
    printUrl: Optional[str]
    backgroundColor: str
    widthMm: int
    heightMm: int
    pixelWidth: int
    pixelHeight: int
    qualityStatus: str


class GeneratePrintLayoutResult(BaseModel):
    imageId: str
    layoutType: str
    printUrl: str
