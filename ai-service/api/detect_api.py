from __future__ import annotations

from fastapi import APIRouter

from models.request_models import DetectRequest
from models.response_models import DetectResponse
from services.detect_service import DetectService
from utils.file_utils import public_url_for_path, resolve_input_path, to_url_like_path
from utils.logger import get_logger
from utils.response_utils import success_response

router = APIRouter(tags=['detect'])
detect_service = DetectService()
logger = get_logger(component='detect_api')


@router.post('/ai/detect', response_model=DetectResponse, summary='Detect face and validate source image')
def detect(payload: DetectRequest):
    api_logger = logger.bind(endpoint='/ai/detect', image_id=payload.imageId)
    api_logger.info('detect request accepted')
    resolved_path = resolve_input_path(payload.originalImagePath)
    api_logger.bind(image_path=resolved_path).debug('resolved detect image path')
    result = detect_service.detect(payload.imageId, resolved_path).to_dict()
    result['originalImagePath'] = to_url_like_path(resolved_path)
    result['originalImageUrl'] = public_url_for_path(resolved_path)
    api_logger.bind(face_detected=result['faceDetected'], face_count=result['faceCount']).info('detect request completed')
    return success_response(result, message='OK')
