from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse
from app.schemas.generate import GenerateData
from app.services.photo_processor import PhotoProcessor

router = APIRouter(tags=['generate'])
processor = PhotoProcessor()


@router.post('/generate', response_model=ApiResponse[GenerateData])
async def generate(
    file: Annotated[UploadFile | None, File(default=None)] = None,
    imagePath: Annotated[str | None, Form(default=None)] = None,
    sceneId: Annotated[str | None, Form(default=None)] = None,
    sizeKey: Annotated[str | None, Form(default=None)] = None,
    backgroundColor: Annotated[str | None, Form(default=None)] = None,
    enhance: Annotated[bool, Form(default=False)] = False,
    saveOutput: Annotated[bool, Form(default=True)] = True,
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
