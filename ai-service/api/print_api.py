from fastapi import APIRouter

from models.request_models import GeneratePrintLayoutRequest
from pipeline.generate_print_layout import GeneratePrintLayoutPipeline
from utils.response_utils import success_response

router = APIRouter(tags=['print'])
pipeline = GeneratePrintLayoutPipeline()


@router.post('/ai/generate-print-layout')
def generate_print_layout(payload: GeneratePrintLayoutRequest):
    result = pipeline.run(payload.imageId, payload.hdImagePath, payload.layoutType)
    return success_response(result, message='Generate print layout success')
