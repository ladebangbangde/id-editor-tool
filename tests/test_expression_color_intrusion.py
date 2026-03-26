import numpy as np
from PIL import Image

from app.services.photo_metrics_service import FaceBox
from app.services.photo_precheck_service import PhotoPrecheckService


class _FakeLandmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _FakeFaceLandmarks:
    def __init__(self, points: dict[int, tuple[float, float]]) -> None:
        self.landmark = [_FakeLandmark(0.5, 0.5) for _ in range(468)]
        for idx, (x, y) in points.items():
            self.landmark[idx] = _FakeLandmark(x, y)


class _FakeFaceMeshResult:
    def __init__(self, points: dict[int, tuple[float, float]]) -> None:
        self.multi_face_landmarks = [_FakeFaceLandmarks(points)]


class _FakeFaceMeshDetector:
    def __init__(self, points: dict[int, tuple[float, float]]) -> None:
        self._points = points

    def process(self, rgb):  # noqa: ANN001
        return _FakeFaceMeshResult(self._points)


def _build_landmarks_for_closed_mouth() -> dict[int, tuple[float, float]]:
    points = {
        61: (0.45, 0.60),
        291: (0.55, 0.60),
        13: (0.50, 0.595),
        14: (0.50, 0.605),
        0: (0.50, 0.57),
    }
    inner_ring = [78, 191, 80, 81, 82, 312, 311, 310, 415, 308, 324, 318, 402, 317, 87, 178, 88, 95]
    for idx in inner_ring:
        points[idx] = (0.50, 0.60)
    return points


def test_red_background_intrusion_does_not_trigger_tongue_warning():
    svc = PhotoPrecheckService()
    svc._face_mesh = _FakeFaceMeshDetector(_build_landmarks_for_closed_mouth())

    arr = np.full((1000, 800, 3), 220, dtype=np.uint8)
    arr[605:700, 330:470] = [230, 60, 50]  # 嘴部下方大面积红色干扰
    image = Image.fromarray(arr, mode='RGB')

    state, metrics = svc._detect_expression_via_mesh(image, FaceBox(x=180, y=120, width=260, height=320))

    assert state not in {'tongue_out_warn', 'tongue_out_fail'}
    assert metrics['inner_mouth_area_ratio'] < 0.02
