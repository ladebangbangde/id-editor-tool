from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from models.request_models import GenerateIdPhotoRequest
from models.response_models import DetectResponse, GenerateResponse, PrintLayoutResponse
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from pipeline.generate_print_layout import GeneratePrintLayoutPipeline
from services.detect_service import DetectService
from services.storage_service import StorageService
from utils.config import get_settings
from utils.file_utils import resolve_input_path
from utils.logger import get_logger
from utils.response_utils import success_response

router = APIRouter(tags=['upload-debug'])
settings = get_settings()
storage_service = StorageService()
detect_service = DetectService()
generate_pipeline = GenerateIdPhotoPipeline()
print_pipeline = GeneratePrintLayoutPipeline()
logger = get_logger(component='upload_debug_api')


@router.post('/ai/detect-upload', response_model=DetectResponse, summary='Debug only: upload file and reuse detect pipeline')
async def detect_upload(
    file: UploadFile = File(...),
    imageId: str | None = Form(default=None),
):
    api_logger = logger.bind(endpoint='/ai/detect-upload', image_id=imageId, filename=file.filename)
    api_logger.info('detect upload request accepted')
    stored = storage_service.save_upload(file, imageId)
    api_logger = api_logger.bind(stored_image_id=stored['imageId'], stored_path=stored['originalImagePath'])
    result = detect_service.detect(stored['imageId'], stored['originalImagePath']).to_dict()
    result['originalImagePath'] = stored['originalImageStoragePath']
    result['originalImageUrl'] = stored['originalImageUrl']
    api_logger.bind(face_detected=result['faceDetected'], face_count=result['faceCount']).info('detect upload request completed')
    return success_response(result, message='OK')


@router.post('/ai/generate-id-photo-upload', response_model=GenerateResponse, summary='Debug only: upload file and generate ID photo')
async def generate_id_photo_upload(
    file: UploadFile = File(...),
    imageId: str | None = Form(default=None),
    sourceType: str = Form(default='scene'),
    sceneKey: str | None = Form(default='passport'),
    sizeName: str | None = Form(default=None),
    customWidthMm: int | None = Form(default=None),
    customHeightMm: int | None = Form(default=None),
    backgroundColor: str = Form(default=settings.default_bg_color),
    beautyEnabled: bool = Form(default=False),
    printLayoutType: str | None = Form(default=None),
):
    api_logger = logger.bind(
        endpoint='/ai/generate-id-photo-upload',
        image_id=imageId,
        filename=file.filename,
        source_type=sourceType,
        scene_key=(sizeName or sceneKey),
        background_color=backgroundColor,
        beauty_enabled=beautyEnabled,
        print_layout=printLayoutType,
    )
    api_logger.info('generate id photo upload request accepted')
    stored = storage_service.save_upload(file, imageId)
    payload = GenerateIdPhotoRequest(
        imageId=stored['imageId'],
        sourceType=sourceType,
        sceneKey=sizeName or sceneKey,
        customWidthMm=customWidthMm,
        customHeightMm=customHeightMm,
        backgroundColor=backgroundColor,
        beautyEnabled=beautyEnabled,
        printLayoutType=printLayoutType,
        originalImagePath=stored['originalImagePath'],
    )
    result = generate_pipeline.run(payload.model_dump())
    api_logger.bind(stored_image_id=stored['imageId'], method=result['method'], fallback_used=result['whetherFallbackUsed']).info(
        'generate id photo upload request completed'
    )
    return success_response(result, message='Generate success')


@router.post('/ai/generate-print-layout-upload', response_model=PrintLayoutResponse, summary='Debug only: upload file, generate HD photo, then print layout')
async def generate_print_layout_upload(
    file: UploadFile = File(...),
    imageId: str | None = Form(default=None),
    layoutType: str = Form(default=settings.default_layout_type),
    sourceType: str = Form(default='scene'),
    sceneKey: str | None = Form(default='passport'),
    sizeName: str | None = Form(default=None),
    customWidthMm: int | None = Form(default=None),
    customHeightMm: int | None = Form(default=None),
    backgroundColor: str = Form(default=settings.default_bg_color),
    beautyEnabled: bool = Form(default=False),
):
    api_logger = logger.bind(
        endpoint='/ai/generate-print-layout-upload',
        image_id=imageId,
        filename=file.filename,
        layout_type=layoutType,
        source_type=sourceType,
        scene_key=(sizeName or sceneKey),
        background_color=backgroundColor,
        beauty_enabled=beautyEnabled,
    )
    api_logger.info('generate print layout upload request accepted')
    stored = storage_service.save_upload(file, imageId)
    payload = GenerateIdPhotoRequest(
        imageId=stored['imageId'],
        sourceType=sourceType,
        sceneKey=sizeName or sceneKey,
        customWidthMm=customWidthMm,
        customHeightMm=customHeightMm,
        backgroundColor=backgroundColor,
        beautyEnabled=beautyEnabled,
        printLayoutType=None,
        originalImagePath=stored['originalImagePath'],
    )
    generated = generate_pipeline.run(payload.model_dump())
    result = print_pipeline.run(stored['imageId'], resolve_input_path(generated['hdPath']), layoutType)
    result.update(
        {
            'originalImagePath': generated['originalImagePath'],
            'originalImageUrl': generated['originalImageUrl'],
            'previewPath': generated['previewPath'],
            'previewUrl': generated['previewUrl'],
            'hdPath': generated['hdPath'],
            'hdUrl': generated['hdUrl'],
        }
    )
    api_logger.bind(stored_image_id=stored['imageId'], print_path=result['printPath'], photo_count=result['photoCount']).info(
        'generate print layout upload request completed'
    )
    return success_response(result, message='Generate print layout success')
