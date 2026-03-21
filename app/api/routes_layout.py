from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.common import ApiResponse
from app.schemas.layout import LayoutData
from app.services.photo_processor import get_photo_processor

router = APIRouter(tags=['layout'])


@router.post('/layout', response_model=ApiResponse[LayoutData])
async def layout(
    idPhoto: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    idPhotoPath: str | None = Form(None),
    imagePath: str | None = Form(None),
    sceneId: str | None = Form(None),
    sizeKey: str | None = Form(None),
    backgroundColor: str | None = Form(None),
    enhance: bool = Form(False),
    saveOutput: bool = Form(True),
    paper: str = Form('6inch'),
) -> ApiResponse[LayoutData]:
    processor = get_photo_processor()
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
