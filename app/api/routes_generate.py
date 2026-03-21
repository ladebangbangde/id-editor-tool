from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse
from app.schemas.generate import GenerateData
from app.services.photo_processor import PhotoProcessor

router = APIRouter(tags=['generate'])
processor = PhotoProcessor()


@router.post('/generate', response_model=ApiResponse[GenerateData])
async def generate(
    file: UploadFile | None = File(default=None),
    imagePath: str | None = Form(default=None),
    sceneId: str | None = Form(default=None),
    sizeKey: str | None = Form(default=None),
    backgroundColor: str | None = Form(default=None),
    enhance: bool = Form(default=False),
    saveOutput: bool = Form(default=True),
) -> ApiResponse[GenerateData]:
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
