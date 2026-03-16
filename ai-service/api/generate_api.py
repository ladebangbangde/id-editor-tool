from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models.request_models import GenerateIdPhotoRequest
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from utils.file_utils import resolve_input_path
from utils.response_utils import error_response, success_response

router = APIRouter(tags=["generate"])
pipeline = GenerateIdPhotoPipeline()


@router.post("/ai/generate-id-photo")
def generate_id_photo(payload: GenerateIdPhotoRequest):
    try:
        request_data = payload.model_dump()
        request_data["originalImagePath"] = resolve_input_path(payload.originalImagePath)
        result = pipeline.run(request_data)
        return success_response(result, message="Generate success")
    except Exception as exc:
        return JSONResponse(status_code=400, content=error_response(str(exc), data=None))
