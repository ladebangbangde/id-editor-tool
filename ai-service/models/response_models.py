from typing import Optional

from pydantic import BaseModel


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


class DetectResult(BaseModel):
    imageId: str
    hasFace: bool
    faceCount: int
    blurScore: float
    poseValid: bool
    occlusionDetected: bool
    message: str


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
