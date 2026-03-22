from fastapi.testclient import TestClient

from app.main import app
from app.schemas.formal_wear import FormalWearData
from app.services.formal_wear_service import FormalWearService


class DummyGenerateResult:
    taskId = 'gen_test_001'
    previewUrl = '/uploads/preview/20260322/gen_test_001/id_photo_preview.jpg'
    hdUrl = '/uploads/hd/20260322/gen_test_001/id_photo_hd.png'
    previewPath = '/app/uploads/preview/20260322/gen_test_001/id_photo_preview.jpg'
    hdPath = '/app/uploads/hd/20260322/gen_test_001/id_photo_hd.png'
    warnings = ['face slightly tilted']


def test_formal_wear_service_normalizes_inputs() -> None:
    service = FormalWearService()

    class DummyProcessor:
        def generate_from_path(self, **kwargs):
            assert kwargs['image_path'] == '/app/uploads/original/source.jpg'
            assert kwargs['background_color'] == 'white'
            assert kwargs['size_key'] == service.settings.default_size_key
            assert kwargs['enhance'] is True
            assert kwargs['save_output'] is True
            return DummyGenerateResult()

    service.processor = DummyProcessor()

    result = service.create_from_path(
        image_path='/app/uploads/original/source.jpg',
        gender='女',
        style='business',
        color='navy',
        enhance=True,
        save_output=True,
    )

    assert isinstance(result, FormalWearData)
    assert result.taskId == 'gen_test_001'
    assert result.gender == 'female'
    assert result.style == 'business'
    assert result.color == 'navy'
    assert any('requested clothing color=navy' in warning for warning in result.warnings)
    assert any('style=business' in warning for warning in result.warnings)


def test_formal_wear_service_preserves_black_gray_color_without_using_it_as_background() -> None:
    service = FormalWearService()
    seen_background_colors: list[str] = []

    class DummyProcessor:
        def generate_from_path(self, **kwargs):
            seen_background_colors.append(kwargs['background_color'])
            return DummyGenerateResult()

    service.processor = DummyProcessor()

    for requested_color in ('black', 'gray'):
        result = service.create_from_path(
            image_path='/app/uploads/original/source.jpg',
            gender='male',
            style='formal',
            color=requested_color,
            enhance=False,
            save_output=True,
        )
        assert result.color == requested_color

    assert seen_background_colors == ['white', 'white']


def test_formal_wear_route_supports_image_path(monkeypatch) -> None:
    class DummyService:
        async def create(self, **kwargs):
            assert kwargs['image_path'] == '/app/uploads/original/source.jpg'
            assert kwargs['gender'] == 'male'
            assert kwargs['style'] == 'formal'
            assert kwargs['color'] == 'blue'
            assert kwargs['enhance'] is False
            assert kwargs['save_output'] is True
            return FormalWearData(
                taskId='formal_test_001',
                previewUrl='/uploads/preview/20260322/formal_test_001/id_photo_preview.jpg',
                hdUrl='/uploads/hd/20260322/formal_test_001/id_photo_hd.png',
                gender='male',
                style='formal',
                color='blue',
                warnings=[],
                previewPath='/app/uploads/preview/20260322/formal_test_001/id_photo_preview.jpg',
                hdPath='/app/uploads/hd/20260322/formal_test_001/id_photo_hd.png',
            )

    monkeypatch.setattr('app.api.routes_formal_wear.get_formal_wear_service', lambda: DummyService())

    client = TestClient(app)
    response = client.post(
        '/formal-wear',
        data={
            'imagePath': '/app/uploads/original/source.jpg',
            'gender': 'male',
            'style': 'formal',
            'color': 'blue',
            'saveOutput': 'true',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['message'] == 'ok'
    assert payload['data']['taskId'] == 'formal_test_001'
    assert payload['data']['previewUrl'].endswith('id_photo_preview.jpg')
    assert payload['data']['hdUrl'].endswith('id_photo_hd.png')
