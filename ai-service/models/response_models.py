from __future__ import annotations

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    service: str
    status: str
    version: str
    docs: str
    openapi: str


class HealthResponse(BaseModel):
    service: str
    status: str


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    errorCode: str | None = None
    data: dict | None = None


class BackgroundColorModel(BaseModel):
    key: str
    nameZh: str
    nameEn: str
    rgb: list[int]
    hex: str
    description: str


class BackgroundColorsResponse(BaseModel):
    success: bool = True
    message: str
    data: list[BackgroundColorModel]


class PhotoSizeModel(BaseModel):
    sceneKey: str
    sceneName: str
    sceneNameEn: str
    widthMm: int
    heightMm: int
    pixelWidth: int
    pixelHeight: int
    unit: str
    description: str


class PhotoSizesResponse(BaseModel):
    success: bool = True
    message: str
    data: list[PhotoSizeModel]


class ImageMetadataModel(BaseModel):
    filename: str
    contentType: str
    fileSize: int
    width: int
    height: int
    format: str
    mode: str


class QualityResultModel(BaseModel):
    status: str
    message: str


class FaceBoxModel(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectResultModel(BaseModel):
    imageId: str
    hasFace: bool
    faceCount: int
    blurScore: float
    poseValid: bool
    occlusionDetected: bool
    isProcessable: bool
    qualityStatus: str
    qualityMessage: str
    imageWidth: int
    imageHeight: int
    imageFormat: str
    imageMode: str
    validationPassed: bool
    reasons: list[str]
    message: str
    primaryFaceBox: FaceBoxModel | None = None


class DetectResponse(BaseModel):
    success: bool = True
    message: str
    data: DetectResultModel


class ValidateResultModel(BaseModel):
    passed: bool
    message: str
    reasons: list[str]
    metadata: ImageMetadataModel
    qualityStatus: str
    qualityMessage: str


class ValidateResponse(BaseModel):
    success: bool = True
    message: str
    data: ValidateResultModel


class ChangeBackgroundResultModel(BaseModel):
    accepted: bool
    processed: bool
    backgroundColor: str
    backgroundColorHex: str
    message: str
    metadata: ImageMetadataModel
    note: str | None = None


class ChangeBackgroundResponse(BaseModel):
    success: bool = True
    message: str
    data: ChangeBackgroundResultModel
