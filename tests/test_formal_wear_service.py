from PIL import Image
import pytest

from app.core.exceptions import ImageTooBlurryError, ShoulderNeckIncompleteError
from app.services.face_detection import FaceDetectionResult
from app.services.formal_wear_geometry import FormalWearGeometry
from app.services.formal_wear_service import FormalWearService


@pytest.fixture
def service() -> FormalWearService:
    return FormalWearService()


def build_detection(*, face_box: dict[str, int], blur_score: float = 0.003) -> FaceDetectionResult:
    return FaceDetectionResult(
        width=1000,
        height=1400,
        face_count=1,
        has_face=True,
        recommended=True,
        can_generate=True,
        status='PASSED',
        result_level='PASSED',
        reasons=[],
        suggestions=[],
        reason_codes=[],
        warnings=[],
        warning_codes=[],
        issues=[],
        face_boxes=[face_box],
        primary_face=face_box,
        blur_score=blur_score,
        occlusion_detected=False,
        occlusion_areas=[],
        pose_accepted=True,
        landmark_stable=True,
        composition_accepted=True,
        metrics={'blurScore': blur_score},
    )


def test_assess_shoulder_neck_rejects_large_face() -> None:
    geometry = FormalWearGeometry()
    detect_result = build_detection(face_box={'x': 240, 'y': 80, 'width': 520, 'height': 860})

    assessment = geometry.assess_shoulder_neck((1000, 1400), detect_result.primary_face, detect_result, 'male', 'standard')

    assert assessment.passed is False
    assert '头像占比过大' in assessment.reasons[0]


def test_generate_rejects_blurry_image_before_render(service: FormalWearService) -> None:
    image = Image.new('RGB', (1000, 1400), 'white')
    service.detector.detect = lambda _image: build_detection(face_box={'x': 300, 'y': 180, 'width': 360, 'height': 480}, blur_score=0.001)  # type: ignore[assignment]

    with pytest.raises(ImageTooBlurryError):
        service.generate(image, 'male', 'standard', 'black', False, False)


def test_generate_returns_warning_and_urls_when_pipeline_runs(service: FormalWearService, tmp_path) -> None:
    image = Image.new('RGB', (1000, 1400), (245, 245, 245))
    face_box = {'x': 320, 'y': 160, 'width': 280, 'height': 360}
    service.storage.upload_root = tmp_path
    service.storage.category_roots = {
        'base': tmp_path,
        'original': tmp_path / 'original',
        'preview': tmp_path / 'preview',
        'hd': tmp_path / 'hd',
        'print': tmp_path / 'print',
        'temp': tmp_path / 'temp',
    }
    for path in service.storage.category_roots.values():
        path.mkdir(parents=True, exist_ok=True)

    service.detector.detect = lambda _image: build_detection(face_box=face_box)  # type: ignore[assignment]
    service.segmenter.remove_background = lambda _image: Image.new('RGBA', image.size, (0, 0, 0, 0))  # type: ignore[assignment]
    service.geometry.assess_shoulder_neck = lambda *args, **kwargs: FormalWearGeometry().assess_shoulder_neck((1000, 1400), face_box, build_detection(face_box=face_box), 'male', 'standard')  # type: ignore[assignment]

    data = service.generate(image, 'male', 'standard', 'black', False, True)

    assert data.taskId.startswith('fw_')
    assert data.previewUrl.endswith('.jpg')
    assert data.hdUrl.endswith('.png')


def test_generate_rejects_incomplete_shoulder_neck(service: FormalWearService) -> None:
    image = Image.new('RGB', (900, 1000), 'white')
    face_box = {'x': 180, 'y': 90, 'width': 520, 'height': 620}
    service.detector.detect = lambda _image: build_detection(face_box=face_box)  # type: ignore[assignment]

    with pytest.raises(ShoulderNeckIncompleteError):
        service.generate(image, 'male', 'standard', 'black', False, False)
