from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse, ErrorBody
from app.schemas.detect import DetectData
from app.services.photo_processor import get_photo_processor

router = APIRouter(tags=['detect'])


@router.post('/detect', response_model=ApiResponse[DetectData])
async def detect(
    file: UploadFile | None = File(None),
    imagePath: str | None = Form(None),
) -> ApiResponse[DetectData]:
    processor = get_photo_processor()
    if file is None and not imagePath:
        raise InvalidArgumentError('Either file or imagePath must be provided')
    if file is not None:
        _, image = await processor.read_upload(file)
    else:
        _, image = processor.read_image_path(imagePath or '')
    data = processor.detect(image)
    if data.resultLevel == 'FAILED':
        primary_reason = data.reasons[0] if data.reasons else None
        return ApiResponse(
            success=False,
            message='当前照片不适合作为证件照原图',
            error=ErrorBody(
                code=primary_reason.code if primary_reason else 'INVALID_IMAGE',
                message=primary_reason.title if primary_reason else '当前照片不适合作为证件照原图',
            ),
            data=data,
        )
    return ApiResponse(success=True, message='ok', data=data)
