from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from core.config import get_settings
from models.request_models import GenerateIdPhotoRequest
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from services.detect_service import DetectService
from services.storage_service import StorageService
from utils.response_utils import success_response

router = APIRouter(tags=['upload-debug'])
settings = get_settings()
storage_service = StorageService()
detect_service = DetectService()
generate_pipeline = GenerateIdPhotoPipeline()


@router.post('/ai/detect-upload')
async def detect_upload(
    image: UploadFile = File(...),
    imageId: str | None = Form(default=None),
):
    stored = storage_service.save_upload(image, imageId)
    result = detect_service.detect(stored['imageId'], stored['originalImagePath']).to_dict()
    result['originalImagePath'] = stored['originalImagePath']
    result['originalImageUrl'] = stored['originalImageUrl']
    return success_response(result, message='OK')


@router.post('/ai/generate-id-photo-upload')
async def generate_id_photo_upload(
    image: UploadFile = File(...),
    imageId: str | None = Form(default=None),
    sourceType: str = Form(default='scene'),
    sceneKey: str | None = Form(default='passport'),
    customWidthMm: int | None = Form(default=None),
    customHeightMm: int | None = Form(default=None),
    backgroundColor: str = Form(default=settings.default_bg_color),
    beautyEnabled: bool = Form(default=False),
    printLayoutType: str | None = Form(default=None),
):
    stored = storage_service.save_upload(image, imageId)
    payload = GenerateIdPhotoRequest(
        imageId=stored['imageId'],
        sourceType=sourceType,
        sceneKey=sceneKey,
        customWidthMm=customWidthMm,
        customHeightMm=customHeightMm,
        backgroundColor=backgroundColor,
        beautyEnabled=beautyEnabled,
        printLayoutType=printLayoutType,
        originalImagePath=stored['originalImagePath'],
    )
    result = generate_pipeline.run(payload.model_dump())
    result['originalImagePath'] = stored['originalImagePath']
    result['originalImageUrl'] = stored['originalImageUrl']
    return success_response(result, message='Generate success')


@router.post('/ai/generate-print-layout-upload')
async def generate_print_layout_upload(
    image: UploadFile = File(...),
    imageId: str | None = Form(default=None),
    layoutType: str = Form(default=settings.default_layout_type),
    sourceType: str = Form(default='scene'),
    sceneKey: str | None = Form(default='passport'),
    customWidthMm: int | None = Form(default=None),
    customHeightMm: int | None = Form(default=None),
    backgroundColor: str = Form(default=settings.default_bg_color),
    beautyEnabled: bool = Form(default=False),
):
    stored = storage_service.save_upload(image, imageId)
    payload = GenerateIdPhotoRequest(
        imageId=stored['imageId'],
        sourceType=sourceType,
        sceneKey=sceneKey,
        customWidthMm=customWidthMm,
        customHeightMm=customHeightMm,
        backgroundColor=backgroundColor,
        beautyEnabled=beautyEnabled,
        printLayoutType=layoutType,
        originalImagePath=stored['originalImagePath'],
    )
    generated = generate_pipeline.run(payload.model_dump())
    result = {
        'imageId': generated['imageId'],
        'layoutType': layoutType,
        'printUrl': generated['printUrl'],
        'previewUrl': generated['previewUrl'],
        'hdUrl': generated['hdUrl'],
        'originalImagePath': stored['originalImagePath'],
        'originalImageUrl': stored['originalImageUrl'],
    }
    return success_response(result, message='Generate print layout success')
