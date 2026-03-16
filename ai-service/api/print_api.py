from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models.request_models import GeneratePrintLayoutRequest
from pipeline.generate_print_layout import GeneratePrintLayoutPipeline
from utils.response_utils import error_response, success_response

router = APIRouter(tags=["print"])
pipeline = GeneratePrintLayoutPipeline()


@router.post("/ai/generate-print-layout")
def generate_print_layout(payload: GeneratePrintLayoutRequest):
    try:
        result = pipeline.run(payload.imageId, payload.hdImagePath, payload.layoutType)
        return success_response(result, message="Generate print layout success")
    except Exception as exc:
        return JSONResponse(status_code=400, content=error_response(str(exc), data=None))
