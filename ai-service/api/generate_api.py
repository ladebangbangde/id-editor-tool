from fastapi import APIRouter

from models.request_models import GenerateIdPhotoRequest
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from utils.response_utils import success_response

router = APIRouter(tags=['generate'])
pipeline = GenerateIdPhotoPipeline()


@router.post('/ai/generate-id-photo')
def generate_id_photo(payload: GenerateIdPhotoRequest):
    result = pipeline.run(payload.model_dump())
    return success_response(result, message='Generate success')
