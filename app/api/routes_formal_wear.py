from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.common import ApiResponse, ErrorBody

router = APIRouter(tags=['formal-wear'])


@router.post('/formal-wear', response_model=ApiResponse[dict])
async def formal_wear(
    file: UploadFile | None = File(None),
    imagePath: str | None = Form(None),
    gender: str | None = Form(None),
    style: str | None = Form(None),
    color: str | None = Form(None),
    enhance: bool = Form(False),
    saveOutput: bool = Form(True),
) -> ApiResponse[dict]:
    _ = (file, imagePath, gender, style, color, enhance, saveOutput)
    return ApiResponse(
        success=False,
        message='换装功能已下线',
        error=ErrorBody(
            code='FORMAL_WEAR_OFFLINE',
            message='换装功能已下线',
            details={'deprecated': True},
        ),
        data={
            'status': 'offline',
            'message': '换装功能已下线',
        },
    )
