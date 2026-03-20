from __future__ import annotations

from fastapi import APIRouter

from models.request_models import GeneratePrintLayoutRequest
from models.response_models import PrintLayoutResponse
from pipeline.generate_print_layout import GeneratePrintLayoutPipeline
from utils.file_utils import resolve_input_path
from utils.logger import get_logger
from utils.response_utils import success_response

router = APIRouter(tags=['print'])
pipeline = GeneratePrintLayoutPipeline()
logger = get_logger(component='print_api')


@router.post('/ai/generate-print-layout', response_model=PrintLayoutResponse, summary='Generate print layout from stored HD image path')
def generate_print_layout(payload: GeneratePrintLayoutRequest):
    api_logger = logger.bind(endpoint='/ai/generate-print-layout', image_id=payload.imageId, layout_type=payload.layoutType)
    api_logger.info('generate print layout request accepted')
    hd_path = resolve_input_path(payload.hdImagePath)
    api_logger.bind(hd_path=hd_path).debug('resolved hd image path')
    result = pipeline.run(payload.imageId, hd_path, payload.layoutType)
    api_logger.bind(print_path=result['printPath'], photo_count=result['photoCount']).info('generate print layout request completed')
    return success_response(result, message='Generate print layout success')
