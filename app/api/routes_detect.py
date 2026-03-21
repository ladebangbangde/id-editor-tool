from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse
from app.schemas.detect import DetectData
from app.services.photo_processor import PhotoProcessor

router = APIRouter(tags=['detect'])
processor = PhotoProcessor()


@router.post('/detect', response_model=ApiResponse[DetectData])
async def detect(
    file: UploadFile | None = File(default=None),
    imagePath: str | None = Form(default=None),
) -> ApiResponse[DetectData]:
    if file is None and not imagePath:
        raise InvalidArgumentError('Either file or imagePath must be provided')
    if file is not None:
        _, image = await processor.read_upload(file)
    else:
        _, image = processor.read_image_path(imagePath or '')
    data = processor.detect(image)
    return ApiResponse(success=True, message='ok', data=data)
