from __future__ import annotations

import secrets

from fastapi import APIRouter, File, Form, UploadFile

from constants.colors import ALLOWED_BACKGROUND_COLORS, list_background_colors
from constants.photo_sizes import PHOTO_SIZE_TEMPLATES, list_photo_sizes
from core.exceptions import AppException, ERROR_INVALID_ARGUMENT
from models.request_models import GenerateIdPhotoRequest
from models.response_models import ToolProcessResponse, ToolSpecsResponse
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from services.storage_service import StorageService
from utils.logger import get_logger
from utils.response_utils import tool_success_response

router = APIRouter(prefix='/api/v1', tags=['tool-v1'])
storage_service = StorageService()
generate_pipeline = GenerateIdPhotoPipeline()
logger = get_logger(component='tool_api')
SUPPORTED_OUTPUT_FORMATS = ('jpg',)


@router.get('/specs', response_model=ToolSpecsResponse, summary='List server-facing tool specs')
def get_specs():
    payload = {
        'backgroundColors': list(ALLOWED_BACKGROUND_COLORS),
        'backgroundColorOptions': list_background_colors(),
        'sizeCodes': list(PHOTO_SIZE_TEMPLATES.keys()),
        'photoSizes': list_photo_sizes(),
        'outputFormats': list(SUPPORTED_OUTPUT_FORMATS),
        'legacyEndpoints': [
            '/ai/health',
            '/ai/colors',
            '/ai/photo-sizes',
            '/ai/detect',
            '/ai/generate-id-photo',
            '/ai/generate-print-layout',
        ],
        'stableEndpoints': ['/health', '/api/v1/specs', '/api/v1/process'],
    }
    logger.bind(endpoint='/api/v1/specs', size_code_count=len(payload['sizeCodes'])).info('tool specs requested')
    return tool_success_response(payload)


@router.post('/process', response_model=ToolProcessResponse, summary='Process uploaded image for server integration')
async def process_image(
    file: UploadFile = File(...),
    imageId: str | None = Form(default=None),
    sizeCode: str = Form(default='passport'),
    backgroundColor: str = Form(default='white'),
    outputFormat: str = Form(default='jpg'),
    beautyEnabled: bool = Form(default=False),
    printLayoutType: str | None = Form(default=None),
):
    normalized_format = (outputFormat or 'jpg').lower()
    generated_image_id = imageId or f'process_{secrets.token_hex(4)}'
    api_logger = logger.bind(
        endpoint='/api/v1/process',
        image_id=generated_image_id,
        filename=file.filename,
        size_code=sizeCode,
        background_color=backgroundColor,
        output_format=normalized_format,
        print_layout=printLayoutType,
    )
    api_logger.info('tool process request accepted')

    if sizeCode not in PHOTO_SIZE_TEMPLATES:
        raise AppException(f'Unknown sizeCode: {sizeCode}', ERROR_INVALID_ARGUMENT, 400)
    if backgroundColor not in ALLOWED_BACKGROUND_COLORS:
        raise AppException(f'Unknown backgroundColor: {backgroundColor}', ERROR_INVALID_ARGUMENT, 400)
    if normalized_format not in SUPPORTED_OUTPUT_FORMATS:
        raise AppException(
            f'Unsupported outputFormat: {outputFormat}. Supported formats: {", ".join(SUPPORTED_OUTPUT_FORMATS)}',
            ERROR_INVALID_ARGUMENT,
            400,
        )

    stored = storage_service.save_upload(file, generated_image_id)
    payload = GenerateIdPhotoRequest(
        imageId=stored['imageId'],
        sourceType='scene',
        sceneKey=sizeCode,
        backgroundColor=backgroundColor,
        beautyEnabled=beautyEnabled,
        printLayoutType=printLayoutType,
        originalImagePath=stored['originalImagePath'],
    )
    result = generate_pipeline.run(payload.model_dump())
    response_data = {
        'imageId': result['imageId'],
        'resultPath': result['hdPath'],
        'resultUrl': result['hdUrl'],
        'previewPath': result['previewPath'],
        'previewUrl': result['previewUrl'],
        'printPath': result['printPath'],
        'printUrl': result['printUrl'],
        'sizeCode': sizeCode,
        'backgroundColor': result['backgroundColor'],
        'outputFormat': normalized_format,
        'width': result['pixelWidth'],
        'height': result['pixelHeight'],
        'qualityStatus': result['qualityStatus'],
        'qualityMessage': result['qualityMessage'],
        'canDirectlyUseForRegistration': result['canDirectlyUseForRegistration'],
        'whetherFallbackUsed': result['whetherFallbackUsed'],
        'segmentationSucceeded': result['segmentationSucceeded'],
        'processNotes': result['processNotes'],
    }
    api_logger.bind(result_path=response_data['resultPath'], fallback_used=response_data['whetherFallbackUsed']).info(
        'tool process request completed'
    )
    return tool_success_response(response_data)
