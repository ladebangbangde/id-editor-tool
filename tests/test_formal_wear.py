from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image, ImageChops, ImageDraw

from app.main import app
from app.schemas.common import FileInfo
from app.schemas.formal_wear import FormalWearData
from app.services.formal_wear_renderer import FormalWearRenderer
from app.services.formal_wear_service import FormalWearService


def _build_foreground(size: tuple[int, int] = (295, 413)) -> Image.Image:
    image = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = size
    draw.ellipse((width * 0.32, height * 0.05, width * 0.68, height * 0.42), fill=(223, 187, 160, 255))
    draw.rounded_rectangle((width * 0.38, height * 0.36, width * 0.62, height * 0.82), radius=24, fill=(220, 190, 168, 255))
    return image


def _image_difference_bbox(left: Image.Image, right: Image.Image):
    return ImageChops.difference(left.convert('RGBA'), right.convert('RGBA')).getbbox()


def test_formal_wear_renderer_makes_visible_changes_for_gender_style_and_color() -> None:
    renderer = FormalWearRenderer()
    foreground = _build_foreground()
    face_box = {'x': 96, 'y': 28, 'width': 104, 'height': 134}

    male_business_black, _ = renderer.render(foreground, face_box=face_box, gender='male', style='business', color='black')
    female_simple_gray, _ = renderer.render(foreground, face_box=face_box, gender='female', style='simple', color='gray')

    assert _image_difference_bbox(foreground, male_business_black) is not None
    assert _image_difference_bbox(male_business_black, female_simple_gray) is not None


def test_formal_wear_service_generates_real_overlay_outputs(tmp_path: Path) -> None:
    class DummyStorage:
        def hd_path(self, task_id: str, filename: str) -> Path:
            return tmp_path / 'hd' / task_id / filename

        def preview_path(self, task_id: str, filename: str) -> Path:
            return tmp_path / 'preview' / task_id / filename

        def temp_path(self, task_id: str, filename: str) -> Path:
            return tmp_path / 'temp' / task_id / filename

        def category_task_dir(self, category: str, task_id: str) -> Path:
            path = tmp_path / category / task_id
            path.mkdir(parents=True, exist_ok=True)
            return path

    class DummyBackground:
        def apply(self, foreground_rgba: Image.Image, background_color: str) -> Image.Image:
            assert background_color == 'white'
            background = Image.new('RGBA', foreground_rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, foreground_rgba.convert('RGBA')).convert('RGB')

    class DummyEnhancer:
        def enhance(self, image: Image.Image) -> Image.Image:
            return image

    source = Image.new('RGB', (800, 1000), 'white')
    segment_foreground = Image.new('RGBA', source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(segment_foreground, 'RGBA')
    draw.ellipse((250, 80, 550, 420), fill=(222, 189, 168, 255))
    draw.rounded_rectangle((285, 360, 515, 960), radius=48, fill=(221, 191, 170, 255))

    processor = SimpleNamespace(
        detector=SimpleNamespace(
            detect=lambda _image: SimpleNamespace(
                can_generate=True,
                warnings=['source warning'],
                primary_face={'x': 260, 'y': 90, 'width': 280, 'height': 320},
            )
        ),
        segmenter=SimpleNamespace(remove_background=lambda _image: segment_foreground),
        background=DummyBackground(),
        enhancer=DummyEnhancer(),
        storage=DummyStorage(),
        read_image_path=lambda path: (Path(path), source),
        _raise_detect_failure=lambda _result: (_ for _ in ()).throw(AssertionError('should not fail')),
        _file_info=lambda path: FileInfo(path=str(path), url=f'/uploads/{path.relative_to(tmp_path).as_posix()}'),
    )

    service = FormalWearService(processor=processor)
    result = service.create_from_path(
        image_path=str(tmp_path / 'source.jpg'),
        gender='female',
        style='business',
        color='navy',
        enhance=False,
        save_output=True,
    )

    assert result.gender == 'female'
    assert result.style == 'business'
    assert result.color == 'navy'
    assert result.previewPath.endswith('formal_wear_preview.jpg')
    assert result.hdPath.endswith('formal_wear_hd.png')
    assert Path(result.previewPath).exists()
    assert Path(result.hdPath).exists()
    assert any('lightweight formal-wear overlay gender=female style=business color=navy' in warning for warning in result.warnings)


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
                previewUrl='/uploads/preview/20260322/formal_test_001/formal_wear_preview.jpg',
                hdUrl='/uploads/hd/20260322/formal_test_001/formal_wear_hd.png',
                gender='male',
                style='standard',
                color='black',
                warnings=['Applied lightweight formal-wear overlay gender=male style=standard color=black'],
                previewPath='/app/uploads/preview/20260322/formal_test_001/formal_wear_preview.jpg',
                hdPath='/app/uploads/hd/20260322/formal_test_001/formal_wear_hd.png',
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
    assert payload['data']['previewUrl'].endswith('formal_wear_preview.jpg')
    assert payload['data']['hdUrl'].endswith('formal_wear_hd.png')
