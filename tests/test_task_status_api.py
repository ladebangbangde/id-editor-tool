from fastapi.testclient import TestClient

from app.main import app
from app.schemas.task_status import StageCode
from app.services.photo_processor import get_photo_processor


client = TestClient(app)


def test_get_task_status_returns_latest_stage() -> None:
    processor = get_photo_processor()
    processor.update_task_stage(
        task_id='task_status_demo',
        stage_code=StageCode.ADJUSTING,
        progress=35,
        message='background is being replaced',
    )

    response = client.get('/tasks/task_status_demo/status')

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['data']['taskId'] == 'task_status_demo'
    assert payload['data']['stageCode'] == 'adjusting'
    assert payload['data']['progress'] == 35


def test_get_task_status_missing_task_returns_400() -> None:
    response = client.get('/tasks/task_not_found_demo/status')

    assert response.status_code == 400
    payload = response.json()
    assert payload['success'] is False
    assert payload['error']['code'] == 'INVALID_ARGUMENT'
