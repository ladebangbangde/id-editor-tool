from __future__ import annotations

import secrets

from fastapi import APIRouter, File, Form, UploadFile

from constants.colors import BACKGROUND_COLORS, ALLOWED_BACKGROUND_COLORS, list_background_colors
from constants.photo_sizes import list_photo_sizes
from core.exceptions import AppException, ERROR_INVALID_ARGUMENT
from models.response_models import (
    BackgroundColorsResponse,
    ChangeBackgroundResponse,
    DetectResponse,
    PhotoSizesResponse,
    ValidateResponse,
)
from services.detect_service import DetectService
from services.validation_service import ValidationService
from utils.logger import get_logger

router = APIRouter(prefix='/api/v1', tags=['id-photo'])
logger = get_logger()
validation_service = ValidationService()
detect_service = DetectService()


async def _load_uploaded_image(image: UploadFile):
    file_bytes = await image.read()
    return validation_service.load_image(
        file_bytes=file_bytes,
        filename=image.filename or 'upload.jpg',
        content_type=image.content_type,
    )


@router.post('/detect', response_model=DetectResponse, summary='Detect whether the uploaded photo is processable')
async def detect_face(
    image: UploadFile = File(..., description='待检测图片文件'),
    image_id: str | None = Form(default=None, description='可选图片标识'),
) -> DetectResponse:
    loaded_image = await _load_uploaded_image(image)
    result = detect_service.detect_from_loaded_image(
        image_id=image_id or secrets.token_hex(8),
        loaded_image=loaded_image,
    )
    logger.info('detect completed: filename={}, processable={}', loaded_image.filename, result.isProcessable)
    return DetectResponse(message='检测完成', data=result.to_dict())


@router.post('/validate', response_model=ValidateResponse, summary='Validate uploaded photo basics')
async def validate_image(image: UploadFile = File(..., description='待校验图片文件')) -> ValidateResponse:
    loaded_image = await _load_uploaded_image(image)
    result = validation_service.validate_upload_image(loaded_image)
    logger.info('validate completed: filename={}', loaded_image.filename)
    return ValidateResponse(message='校验完成', data=result)


@router.post('/change-background', response_model=ChangeBackgroundResponse, summary='Change photo background color')
async def change_background(
    image: UploadFile = File(..., description='待处理图片文件'),
    background_color: str = Form(default='white', description='目标背景色 key，例如 white/blue/red'),
) -> ChangeBackgroundResponse:
    if background_color not in ALLOWED_BACKGROUND_COLORS:
        raise AppException(
            f'不支持的背景色：{background_color}，可选值为 {", ".join(ALLOWED_BACKGROUND_COLORS)}',
            ERROR_INVALID_ARGUMENT,
            400,
        )

    loaded_image = await _load_uploaded_image(image)
    logger.warning('change-background fallback mode: filename={}, color={}', loaded_image.filename, background_color)
    return ChangeBackgroundResponse(
        message='当前版本已接收换底请求，但未启用高级抠图，仅返回校验结果与目标底色信息',
        data={
            'accepted': True,
            'processed': False,
            'backgroundColor': background_color,
            'backgroundColorHex': '#%02X%02X%02X' % BACKGROUND_COLORS[background_color],
            'message': '高级抠图能力尚未启用，服务保持可运行并预留扩展接口',
            'metadata': loaded_image.metadata(),
            'note': '如需真实换底，可在后续启用 segmentation_enabled 并接入稳定分割模型。',
        },
    )


@router.get('/photo-sizes', response_model=PhotoSizesResponse, summary='Get supported ID photo sizes')
def get_photo_sizes() -> PhotoSizesResponse:
    return PhotoSizesResponse(message='获取常用证件照尺寸成功', data=list_photo_sizes())


@router.get('/colors', response_model=BackgroundColorsResponse, summary='Get supported background colors')
def get_colors() -> BackgroundColorsResponse:
    return BackgroundColorsResponse(message='获取背景色列表成功', data=list_background_colors())


@router.post('/quality', include_in_schema=False)
async def deprecated_quality_endpoint() -> None:
    raise AppException('quality 接口已并入 /api/v1/validate 和 /api/v1/detect', ERROR_INVALID_ARGUMENT, 410)
