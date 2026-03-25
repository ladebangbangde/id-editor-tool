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


class BackgroundColorPayload(BaseModel):
    key: str
    nameZh: str
    nameEn: str
    rgb: list[int]
    hex: str
    description: str


class PhotoSizePayload(BaseModel):
    sceneKey: str
    sceneName: str
    sceneNameEn: str
    widthMm: int
    heightMm: int
    pixelWidth: int
    pixelHeight: int
    unit: str
    description: str


class FaceBoxModel(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectResultModel(BaseModel):
    imageId: str
    originalImagePath: str | None = None
    originalImageUrl: str | None = None
    faceDetected: bool
    faceCount: int
    primaryFaceBox: FaceBoxModel | None = None
    faceBoxes: list[FaceBoxModel]
    imageWidth: int | None = None
    imageHeight: int | None = None
    imageFormat: str | None = None
    imageMode: str | None = None
    blurScore: float | None = None
    poseValid: bool | None = None
    occlusionDetected: bool | None = None
    isProcessable: bool | None = None
    validationPassed: bool
    reasons: list[str]
    qualityStatus: str | None = None
    qualityMessage: str | None = None
    auditResult: dict[str, Any] | None = None
    keypointConfidences: dict[str, float] | None = None
    suggestion: str
    message: str


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
    sourceResolutionTooLow: bool | None = None
    outputSizeIsStandard: bool | None = None
    likelyUpscaled: bool | None = None
    cropBox: dict[str, int]
    targetWidth: int
    targetHeight: int
    headRatio: float
    appliedOperations: list[str]
    processNotes: list[str]
    whetherFallbackUsed: bool
    segmentationSucceeded: bool
    finalOutputType: str
    canDirectlyUseForRegistration: bool
    layoutType: str | None = None
    paperType: str | None = None
    photoCount: int | None = None


class PrintLayoutResultModel(BaseModel):
    imageId: str
    originalImagePath: str | None = None
    originalImageUrl: str | None = None
    previewPath: str | None = None
    previewUrl: str | None = None
    hdPath: str
    hdUrl: str
    printPath: str
    printUrl: str
    layoutType: str
    paperType: str
    photoCount: int


class HealthResponse(ApiResponse[HealthPayload]):
    pass


class DetectResponse(ApiResponse[DetectResultModel]):
    pass


class GenerateResponse(ApiResponse[GenerateResultModel]):
    pass


class PrintLayoutResponse(ApiResponse[PrintLayoutResultModel]):
    pass


class ColorListResponse(ApiResponse[list[BackgroundColorPayload]]):
    pass


class PhotoSizeListResponse(ApiResponse[list[PhotoSizePayload]]):
    pass


class GenericDataResponse(ApiResponse[dict[str, Any]]):
    pass
