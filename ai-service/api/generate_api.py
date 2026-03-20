from __future__ import annotations

from fastapi import APIRouter

from models.request_models import GenerateIdPhotoRequest
from models.response_models import GenerateResponse
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from utils.file_utils import resolve_input_path
from utils.logger import get_logger
from utils.response_utils import success_response

router = APIRouter(tags=['generate'])
pipeline = GenerateIdPhotoPipeline()
logger = get_logger(component='generate_api')


@router.post('/ai/generate-id-photo', response_model=GenerateResponse, summary='Generate ID photo from stored image path')
def generate_id_photo(payload: GenerateIdPhotoRequest):
    api_logger = logger.bind(
        endpoint='/ai/generate-id-photo',
        image_id=payload.imageId,
        source_type=payload.sourceType,
        scene_key=payload.sceneKey,
        background_color=payload.backgroundColor,
        print_layout=payload.printLayoutType,
    )
    api_logger.info('generate id photo request accepted')
    request_data = payload.model_dump()
    request_data['originalImagePath'] = resolve_input_path(payload.originalImagePath)
    api_logger.bind(image_path=request_data['originalImagePath']).debug('resolved original image path')
    result = pipeline.run(request_data)
    api_logger.bind(method=result['method'], fallback_used=result['whetherFallbackUsed']).info('generate id photo request completed')
    return success_response(result, message='Generate success')
