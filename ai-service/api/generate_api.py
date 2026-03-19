from __future__ import annotations

from fastapi import APIRouter

from models.request_models import GenerateIdPhotoRequest
from models.response_models import GenerateResponse
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from utils.file_utils import resolve_input_path
from utils.response_utils import success_response

router = APIRouter(tags=['generate'])
pipeline = GenerateIdPhotoPipeline()


@router.post('/ai/generate-id-photo', response_model=GenerateResponse, summary='Generate ID photo from stored image path')
def generate_id_photo(payload: GenerateIdPhotoRequest):
    request_data = payload.model_dump()
    request_data['originalImagePath'] = resolve_input_path(payload.originalImagePath)
    result = pipeline.run(request_data)
    return success_response(result, message='Generate success')
