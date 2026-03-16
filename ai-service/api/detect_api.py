from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models.request_models import DetectRequest
from services.detect_service import DetectService
from utils.response_utils import error_response, success_response

router = APIRouter(tags=["detect"])
service = DetectService()


@router.post("/ai/detect")
def detect_face(payload: DetectRequest):
    try:
        result = service.detect(image_id=payload.imageId, image_path=payload.originalImagePath)
        return success_response(result.to_dict(), message="OK")
    except Exception as exc:
        return JSONResponse(status_code=400, content=error_response(str(exc), data=None))
