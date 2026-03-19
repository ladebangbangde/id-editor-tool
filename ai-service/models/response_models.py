from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel


T = TypeVar('T')


class RootResponse(BaseModel):
    service: str
    status: str
    version: str
    docs: str
    openapi: str


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = 'OK'
    errorCode: str | None = None
    data: T | None = None


class ErrorResponse(ApiResponse[dict]):
    success: bool = False


class HealthPayload(BaseModel):
    service: str
    status: str
    host: str
    port: int
    uploadRoot: str
    staticMountPath: str


class FaceBoxModel(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectResultModel(BaseModel):
    imageId: str
    faceDetected: bool
    faceCount: int
    faceBoxes: list[FaceBoxModel]
    primaryFaceBox: FaceBoxModel | None = None
    imageWidth: int
    imageHeight: int
    imageFormat: str
    imageMode: str
    blurScore: float
    poseValid: bool
    occlusionDetected: bool
    isProcessable: bool
    validationPassed: bool
    reasons: list[str]
    qualityStatus: str
    qualityMessage: str
    suggestion: str
    message: str
    originalImagePath: str | None = None
    originalImageUrl: str | None = None


class GenerateResultModel(BaseModel):
    imageId: str
    originalImagePath: str
    originalImageUrl: str
    previewPath: str
    previewUrl: str
    hdPath: str
    hdUrl: str
    printPath: str | None = None
    printUrl: str | None = None
    backgroundColor: str
    method: str
    widthMm: int
    heightMm: int
    pixelWidth: int
    pixelHeight: int
    qualityStatus: str
    qualityMessage: str
    cropBox: dict[str, int]
    targetWidth: int
    targetHeight: int
    headRatio: float
    appliedOperations: list[str]
    processNotes: list[str]
    layoutType: str | None = None
    paperType: str | None = None
    photoCount: int | None = None


class HealthResponse(ApiResponse[HealthPayload]):
    pass


class DetectResponse(ApiResponse[DetectResultModel]):
    pass


class GenericDataResponse(ApiResponse[dict[str, Any]]):
    pass
