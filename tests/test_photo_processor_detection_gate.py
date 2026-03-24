from PIL import Image
import pytest

from app.core.exceptions import EyeOccludedError
from app.services.face_detection import DetectionIssue, DetectionReason, FaceDetectionResult, FAILED, WARNING
from app.services.photo_processor import PhotoProcessor


def build_result(*, status: str, can_generate: bool, recommended: bool, reasons=None, reason_codes=None, warnings=None, warning_codes=None):
    reason_texts = reasons or []
    codes = reason_codes or []
    return FaceDetectionResult(
        width=800,
        height=1000,
        face_count=1,
        has_face=True,
        recommended=recommended,
        can_generate=can_generate,
        status=status,
        result_level=status,
        reasons=[
            DetectionReason(code=code, title=message, detail=message)
            for code, message in zip(codes, reason_texts)
        ],
        suggestions=['请露出完整双眼与面部'] if codes else [],
        reason_codes=codes,
        warnings=warnings or [],
        warning_codes=warning_codes or [],
        issues=[
            DetectionIssue(code=code, message=message, severity=FAILED)
            for code, message in zip(codes, reason_texts)
        ]
        + [
            DetectionIssue(code=code, message=message, severity=WARNING)
            for code, message in zip(warning_codes or [], warnings or [])
        ],
        face_boxes=[{'x': 220, 'y': 140, 'width': 340, 'height': 460}],
        primary_face={'x': 220, 'y': 140, 'width': 340, 'height': 460},
        blur_score=0.01,
        occlusion_detected='FACE_OCCLUDED' in (reason_codes or []) or 'EYE_OCCLUDED' in (reason_codes or []),
        occlusion_areas=['eyes'] if 'EYE_OCCLUDED' in (reason_codes or []) else [],
        pose_accepted='INVALID_POSE' not in ((reason_codes or []) + (warning_codes or [])),
        landmark_stable='LANDMARK_UNSTABLE' not in ((reason_codes or []) + (warning_codes or [])),
        composition_accepted='BAD_COMPOSITION' not in ((reason_codes or []) + (warning_codes or [])),
        metrics={'centerOffsetRatio': 0.03},
    )


@pytest.fixture
def processor() -> PhotoProcessor:
    return PhotoProcessor()


def test_build_detect_data_includes_usability_fields(processor: PhotoProcessor) -> None:
    result = build_result(
        status=WARNING,
        can_generate=True,
        recommended=False,
        warnings=['人脸位置略偏，后续裁切存在一定风险'],
        warning_codes=['BAD_COMPOSITION'],
    )

    payload = processor._build_detect_data(result)

    assert payload.status == 'WARNING'
    assert payload.resultLevel == 'WARNING'
    assert payload.canGenerate is True
    assert payload.pass_ is True
    assert payload.suggestions == []
    assert payload.warningCodes == ['BAD_COMPOSITION']
    assert payload.compositionAccepted is False
    assert payload.warning == '人脸位置略偏，后续裁切存在一定风险'


def test_generate_blocks_failed_source_image_before_pipeline(processor: PhotoProcessor) -> None:
    image = Image.new('RGB', (800, 1000), 'white')
    processor.detector.detect = lambda _image: build_result(
        status=FAILED,
        can_generate=False,
        recommended=False,
        reasons=['双眼或单眼被遮挡'],
        reason_codes=['EYE_OCCLUDED'],
    )

    with pytest.raises(EyeOccludedError) as exc_info:
        processor.generate(
            image=image,
            size_key='one_inch',
            background_color='blue',
            enhance=False,
            save_output=False,
        )

    assert exc_info.value.details['resultLevel'] == 'FAILED'
    assert exc_info.value.details['reasons'][0]['code'] == 'EYE_OCCLUDED'
    assert exc_info.value.details['suggestions'] == ['请露出完整双眼与面部']
