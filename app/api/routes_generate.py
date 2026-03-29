from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import InvalidArgumentError
from app.schemas.common import ApiResponse
from app.schemas.generate import GenerateData, GenerateSelectionData
from app.schemas.task_status import TaskStatusData
from app.services.photo_processor import get_photo_processor
from app.services.task_status_store import get_task_status_store

router = APIRouter(tags=['generate'])


@router.post('/generate', response_model=ApiResponse[GenerateData])
async def generate(
    file: UploadFile | None = File(None),
    imagePath: str | None = Form(None),
    sceneId: str | None = Form(None),
    sizeKey: str | None = Form(None),
    backgroundColor: str | None = Form(None),
    enhance: bool = Form(False),
    saveOutput: bool = Form(True),
    taskId: str | None = Form(None),
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
            task_id=taskId,
        )
    else:
        data = processor.generate_from_path(
            image_path=imagePath or '',
            size_key=sizeKey or sceneId,
            background_color=backgroundColor,
            enhance=enhance,
            save_output=saveOutput,
            task_id=taskId,
        )
    return ApiResponse(success=True, message='ok', data=data)


@router.post('/tasks/create', response_model=ApiResponse[TaskStatusData])
async def create_task(taskId: str | None = Form(None)) -> ApiResponse[TaskStatusData]:
    store = get_task_status_store()
    data = store.create_task(task_id=taskId)
    return ApiResponse(success=True, message='ok', data=data)


@router.get('/tasks/{task_id}/status', response_model=ApiResponse[TaskStatusData])
async def get_task_status(task_id: str) -> ApiResponse[TaskStatusData]:
    store = get_task_status_store()
    data = store.get_status(task_id)
    return ApiResponse(success=True, message='ok', data=data)


@router.post('/generate/select', response_model=ApiResponse[GenerateSelectionData])
async def select_generated_candidate(
    taskId: str = Form(...),
    candidateId: str = Form(...),
) -> ApiResponse[GenerateSelectionData]:
    processor = get_photo_processor()
    if not candidateId:
        raise InvalidArgumentError('请先选择要保存的图片')
    data = processor.select_candidate(task_id=taskId, candidate_id=candidateId)
    return ApiResponse(success=True, message='ok', data=data)
