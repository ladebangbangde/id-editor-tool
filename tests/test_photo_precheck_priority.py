from PIL import Image

from app.services.photo_precheck_service import FAIL, PhotoPrecheckService


def _mock_face():
    return {
        'x': 180,
        'y': 120,
        'width': 260,
        'height': 320,
        'score': 0.95,
        'keypoints': [{'x': 0.40, 'y': 0.40}, {'x': 0.60, 'y': 0.40}],
    }


def test_single_eye_closed_is_failed_and_primary_over_blur():
    svc = PhotoPrecheckService()
    svc.metrics_service.calculate = lambda image, face_box: {
        'blur_score': 120.0,
        'brightness': 130.0,
        'edge_density': 0.1,
        'face_width_ratio': 0.32,
        'face_height_ratio': 0.42,
        'face_center_x': 0.5,
        'face_center_y': 0.5,
    }
    svc._detect_faces = lambda image: [_mock_face()]
    svc._detect_visible_eyes = lambda image, face_box: 2
    svc._detect_eye_state_via_mesh = lambda image, face_box: ('single_eye_closed', {'left_eye_ear': 0.12, 'right_eye_ear': 0.27})
    svc._detect_hand_on_face = lambda image, face_box: ('clear', 0.0)
    svc._detect_neck_accessory = lambda image, face_box: (0.0, {})

    img = Image.new('RGB', (800, 1000), 'white')
    result = svc.precheck(img)

    assert result.status == FAIL
    assert 'EYE_OCCLUDED' in result.reason_codes
    # 合规主问题必须压过清晰度提醒
    assert result.primary_issue == 'EYE_OCCLUDED'


def test_hand_covering_face_is_failed():
    svc = PhotoPrecheckService()
    svc.metrics_service.calculate = lambda image, face_box: {
        'blur_score': 120.0,
        'brightness': 130.0,
        'edge_density': 0.1,
        'face_width_ratio': 0.32,
        'face_height_ratio': 0.42,
        'face_center_x': 0.5,
        'face_center_y': 0.5,
    }
    svc._detect_faces = lambda image: [_mock_face()]
    svc._detect_visible_eyes = lambda image, face_box: 2
    svc._detect_eye_state_via_mesh = lambda image, face_box: ('open', {'left_eye_ear': 0.30, 'right_eye_ear': 0.31})
    svc._detect_hand_on_face = lambda image, face_box: ('fail', 0.04)
    svc._detect_neck_accessory = lambda image, face_box: (0.0, {})

    img = Image.new('RGB', (800, 1000), 'white')
    result = svc.precheck(img)

    assert result.status == FAIL
    assert 'HAND_OCCLUSION' in result.reason_codes
