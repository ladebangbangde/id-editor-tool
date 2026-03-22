from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.common import ApiResponse
from app.schemas.formal_wear import FormalWearData
from app.services.formal_wear_service import get_formal_wear_service

router = APIRouter(tags=['formal-wear'])


@router.post('/formal-wear', response_model=ApiResponse[FormalWearData])
async def formal_wear(
    file: UploadFile | None = File(None),
    imagePath: str | None = Form(None),
    gender: str | None = Form(None),
    style: str | None = Form(None),
    color: str | None = Form(None),
    enhance: bool = Form(False),
    saveOutput: bool = Form(True),
) -> ApiResponse[FormalWearData]:
    service = get_formal_wear_service()
    data = await service.create(
        file=file,
        image_path=imagePath,
        gender=gender,
        style=style,
        color=color,
        enhance=enhance,
        save_output=saveOutput,
    )
    return ApiResponse(success=True, message='ok', data=data)
