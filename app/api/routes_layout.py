from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.common import ApiResponse
from app.schemas.layout import LayoutData
from app.services.photo_processor import PhotoProcessor

router = APIRouter(tags=['layout'])
processor = PhotoProcessor()


@router.post('/layout', response_model=ApiResponse[LayoutData])
async def layout(
    idPhoto: Annotated[UploadFile | None, File(default=None)] = None,
    image: Annotated[UploadFile | None, File(default=None)] = None,
    idPhotoPath: Annotated[str | None, Form(default=None)] = None,
    imagePath: Annotated[str | None, Form(default=None)] = None,
    sceneId: Annotated[str | None, Form(default=None)] = None,
    sizeKey: Annotated[str | None, Form(default=None)] = None,
    backgroundColor: Annotated[str | None, Form(default=None)] = None,
    enhance: Annotated[bool, Form(default=False)] = False,
    saveOutput: Annotated[bool, Form(default=True)] = True,
    paper: Annotated[str, Form(default='6inch')] = '6inch',
) -> ApiResponse[LayoutData]:
    data = await processor.layout(
        id_photo=idPhoto,
        image=image,
        id_photo_path=idPhotoPath,
        image_path=imagePath,
        size_key=sizeKey or sceneId,
        background_color=backgroundColor,
        enhance=enhance,
        save_output=saveOutput,
        paper=paper,
    )
    return ApiResponse(success=True, message='ok', data=data)
