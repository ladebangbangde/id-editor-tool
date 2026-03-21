from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse
from app.schemas.generate import GenerateData
from app.services.photo_processor import get_photo_processor

router = APIRouter(tags=['generate'])


@router.post('/generate', response_model=ApiResponse[GenerateData])
async def generate(
    file: UploadFile | None = File(None),
    imagePath: str | None = Form(None),
    sceneId: str | None = Form(None),
    sizeKey: str | None = Form(None),
    backgroundColor: str | None = Form(None),
    enhance: bool = Form(False),
    saveOutput: bool = Form(True),
) -> ApiResponse[GenerateData]:
    processor = get_photo_processor()
    if file is None and not imagePath:
        raise InvalidArgumentError('Either file or imagePath must be provided')
    if file is not None:
        data = await processor.generate_from_upload(
            file=file,
            size_key=sizeKey or sceneId,
            background_color=backgroundColor,
            enhance=enhance,
            save_output=saveOutput,
        )
    else:
        data = processor.generate_from_path(
            image_path=imagePath or '',
            size_key=sizeKey or sceneId,
            background_color=backgroundColor,
            enhance=enhance,
            save_output=saveOutput,
        )
    return ApiResponse(success=True, message='ok', data=data)
