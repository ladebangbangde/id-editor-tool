from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.common import ApiResponse
from app.schemas.generate import GenerateData
from app.services.photo_processor import PhotoProcessor

router = APIRouter(tags=['generate'])
processor = PhotoProcessor()


@router.post('/generate', response_model=ApiResponse[GenerateData])
async def generate(
    file: UploadFile = File(...),
    sceneId: str | None = Form(default=None),
    sizeKey: str | None = Form(default=None),
    backgroundColor: str | None = Form(default=None),
    enhance: bool = Form(default=False),
    saveOutput: bool = Form(default=True),
) -> ApiResponse[GenerateData]:
    data = await processor.generate_from_upload(
        file=file,
        size_key=sizeKey or sceneId,
        background_color=backgroundColor,
        enhance=enhance,
        save_output=saveOutput,
    )
    return ApiResponse(success=True, message='ok', data=data)
