from fastapi import APIRouter
from fastapi.responses import JSONResponse

from models.request_models import GeneratePrintLayoutRequest
from pipeline.generate_print_layout import GeneratePrintLayoutPipeline
from utils.file_utils import resolve_input_path
from utils.response_utils import error_response, success_response

router = APIRouter(tags=["print"])
pipeline = GeneratePrintLayoutPipeline()


@router.post("/ai/generate-print-layout")
def generate_print_layout(payload: GeneratePrintLayoutRequest):
    try:
        hd_path = resolve_input_path(payload.hdImagePath)
        result = pipeline.run(payload.imageId, hd_path, payload.layoutType)
        return success_response(result, message="Generate print layout success")
    except Exception as exc:
        return JSONResponse(status_code=400, content=error_response(str(exc), data=None))
