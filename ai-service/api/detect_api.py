from fastapi import APIRouter

from models.request_models import DetectRequest
from services.detect_service import DetectService
from utils.response_utils import success_response

router = APIRouter(tags=['detect'])
service = DetectService()


@router.post('/ai/detect')
def detect_face(payload: DetectRequest):
    result = service.detect(image_id=payload.imageId, image_path=payload.originalImagePath)
    return success_response(result.to_dict(), message='OK')
