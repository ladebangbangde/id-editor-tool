from __future__ import annotations

from fastapi import APIRouter

from models.request_models import DetectRequest
from models.response_models import DetectResponse
from services.detect_service import DetectService
from utils.file_utils import public_url_for_path, resolve_input_path, to_url_like_path
from utils.response_utils import success_response

router = APIRouter(tags=['detect'])
detect_service = DetectService()


@router.post('/ai/detect', response_model=DetectResponse, summary='Detect face and validate source image')
def detect(payload: DetectRequest):
    resolved_path = resolve_input_path(payload.originalImagePath)
    result = detect_service.detect(payload.imageId, resolved_path).to_dict()
    result['originalImagePath'] = to_url_like_path(resolved_path)
    result['originalImageUrl'] = public_url_for_path(resolved_path)
    return success_response(result, message='OK')
