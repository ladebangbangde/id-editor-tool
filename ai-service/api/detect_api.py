from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models.request_models import DetectRequest
from services.detect_service import DetectService
from utils.file_utils import resolve_input_path
from utils.response_utils import error_response, success_response

router = APIRouter(tags=["detect"])
service = DetectService()


@router.post("/ai/detect")
def detect_face(payload: DetectRequest):
    try:
        image_path = resolve_input_path(payload.originalImagePath)
        result = service.detect(image_id=payload.imageId, image_path=image_path)
        return success_response(result.to_dict(), message="OK")
    except Exception as exc:
        return JSONResponse(status_code=400, content=error_response(str(exc), data=None))
