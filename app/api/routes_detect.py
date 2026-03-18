from fastapi import APIRouter, File, UploadFile

from app.schemas.common import ApiResponse
from app.schemas.detect import DetectData
from app.services.photo_processor import PhotoProcessor

router = APIRouter(tags=['detect'])
processor = PhotoProcessor()


@router.post('/detect', response_model=ApiResponse[DetectData])
async def detect(file: UploadFile = File(...)) -> ApiResponse[DetectData]:
    _, image = await processor.read_upload(file)
    data = processor.detect(image)
    return ApiResponse(success=True, message='ok', data=data)
