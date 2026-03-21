from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.common import ApiResponse
from app.schemas.layout import LayoutData
from app.services.photo_processor import PhotoProcessor

router = APIRouter(tags=['layout'])
processor = PhotoProcessor()


@router.post('/layout', response_model=ApiResponse[LayoutData])
async def layout(
    idPhoto: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    idPhotoPath: str | None = Form(default=None),
    imagePath: str | None = Form(default=None),
    sceneId: str | None = Form(default=None),
    sizeKey: str | None = Form(default=None),
    backgroundColor: str | None = Form(default=None),
    enhance: bool = Form(default=False),
    saveOutput: bool = Form(default=True),
    paper: str = Form(default='6inch'),
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
