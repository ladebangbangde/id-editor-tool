from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse, ErrorBody
from app.schemas.detect import DetectData
from app.schemas.generate import GenerateData
from app.schemas.specs import PhotoSpecsData
from app.services.photo_processor import get_photo_processor
from app.services.specs import list_photo_specs, supported_size_keys

router = APIRouter(prefix='/photo', tags=['photo'])


@router.get('/specs', response_model=ApiResponse[PhotoSpecsData])
async def photo_specs() -> ApiResponse[PhotoSpecsData]:
    data = PhotoSpecsData(
        supportedSizeKeys=supported_size_keys(),
        specs=list_photo_specs(),
        customSizeSupported=False,
        customSizeHint='当前不支持 widthMm/heightMm/pixelWidth/pixelHeight 自定义尺寸输入。',
    )
    return ApiResponse(success=True, message='ok', data=data)


@router.post('/precheck', response_model=ApiResponse[DetectData])
async def precheck(
    file: UploadFile | None = File(None),
    imagePath: str | None = Form(None),
) -> ApiResponse[DetectData]:
    processor = get_photo_processor()
    if file is None and not imagePath:
        raise InvalidArgumentError('Either file or imagePath must be provided')
    if file is not None:
        _, image = await processor.read_upload(file)
    else:
        _, image = processor.read_image_path(imagePath or '')

    data = processor.detect(image)
    if data.resultLevel == 'FAIL':
        primary_reason = data.reasons[0] if data.reasons else None
        return ApiResponse(
            success=False,
            message='当前照片暂不适合进入证件照处理流程',
            error=ErrorBody(
                code=primary_reason.code if primary_reason else 'INVALID_IMAGE',
                message=primary_reason.title if primary_reason else '当前照片暂不适合进入证件照处理流程',
            ),
            data=data,
        )
    return ApiResponse(success=True, message='ok', data=data)


@router.post('/process', response_model=ApiResponse[GenerateData])
async def process_photo(
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
