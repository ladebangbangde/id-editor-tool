from __future__ import annotations

from fastapi import APIRouter

from models.request_models import GeneratePrintLayoutRequest
from models.response_models import PrintLayoutResponse
from pipeline.generate_print_layout import GeneratePrintLayoutPipeline
from utils.file_utils import resolve_input_path
from utils.response_utils import success_response

router = APIRouter(tags=['print'])
pipeline = GeneratePrintLayoutPipeline()


@router.post('/ai/generate-print-layout', response_model=PrintLayoutResponse, summary='Generate print layout from stored HD image path')
def generate_print_layout(payload: GeneratePrintLayoutRequest):
    hd_path = resolve_input_path(payload.hdImagePath)
    result = pipeline.run(payload.imageId, hd_path, payload.layoutType)
    return success_response(result, message='Generate print layout success')
