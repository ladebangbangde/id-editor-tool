from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models.request_models import GenerateIdPhotoRequest
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from utils.response_utils import error_response, success_response

router = APIRouter(tags=["generate"])
pipeline = GenerateIdPhotoPipeline()


@router.post("/ai/generate-id-photo")
def generate_id_photo(payload: GenerateIdPhotoRequest):
    try:
        result = pipeline.run(payload.model_dump())
        return success_response(result, message="Generate success")
    except Exception as exc:
        return JSONResponse(status_code=400, content=error_response(str(exc), data=None))
