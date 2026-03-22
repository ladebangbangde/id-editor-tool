from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse
from app.schemas.formal_wear import FormalWearColor, FormalWearData, FormalWearGender, FormalWearStyle
from app.services.formal_wear_service import get_formal_wear_service

router = APIRouter(prefix='/formal-wear', tags=['formal-wear'])


@router.post('/generate', response_model=ApiResponse[FormalWearData])
async def generate_formal_wear(
    file: UploadFile | None = File(None),
    imagePath: str | None = Form(None),
    gender: FormalWearGender = Form(...),
    style: FormalWearStyle = Form('standard'),
    color: FormalWearColor = Form('black'),
    enhance: bool = Form(False),
    saveOutput: bool = Form(True),
) -> ApiResponse[FormalWearData]:
    service = get_formal_wear_service()
    if file is None and not imagePath:
        raise InvalidArgumentError('Either file or imagePath must be provided')
    if file is not None:
        data = await service.generate_from_upload(file, gender, style, color, enhance, saveOutput)
    else:
        data = service.generate_from_path(imagePath or '', gender, style, color, enhance, saveOutput)
    return ApiResponse(success=True, message='ok', data=data)
