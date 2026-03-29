from fastapi import APIRouter, File, Form, UploadFile

from app.core.exceptions import AppError, InvalidArgumentError
from app.schemas.common import ApiResponse
from app.schemas.generate import GenerateData, GenerateSelectionData
from app.schemas.task_status import StageCode, TaskStatusData
from app.services.photo_processor import get_photo_processor
from app.utils.file_naming import build_task_id

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

    current_task_id = taskId or build_task_id('gen')
    processor.update_task_stage(
        task_id=current_task_id,
        stage_code=StageCode.CHECKING,
        progress=10,
        message='photo upload is being validated',
    )

    try:
        if file is not None:
            data = await processor.generate_from_upload(
                file=file,
                size_key=sizeKey or sceneId,
                background_color=backgroundColor,
                enhance=enhance,
                save_output=saveOutput,
                task_id=current_task_id,
            )
        else:
            data = processor.generate_from_path(
                image_path=imagePath or '',
                size_key=sizeKey or sceneId,
                background_color=backgroundColor,
                enhance=enhance,
                save_output=saveOutput,
                task_id=current_task_id,
            )
        return ApiResponse(success=True, message='ok', data=data)
    except AppError as exc:
        processor.update_task_stage(
            task_id=current_task_id,
            stage_code=StageCode.FAILED,
            progress=95,
            message='task failed',
            error_code=exc.code,
            error_message=exc.message,
        )
        raise
    except Exception:
        processor.update_task_stage(
            task_id=current_task_id,
            stage_code=StageCode.FAILED,
            progress=95,
            message='task failed',
            error_code='PROCESS_FAILED',
            error_message='Unexpected server error',
        )
        raise


@router.get('/tasks/{task_id}/status', response_model=ApiResponse[TaskStatusData])
def get_task_status(task_id: str) -> ApiResponse[TaskStatusData]:
    processor = get_photo_processor()
    status = processor.get_task_status(task_id)
    if status is None:
        raise InvalidArgumentError(f'Task status not found: {task_id}')
    return ApiResponse(success=True, message='ok', data=status)


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
