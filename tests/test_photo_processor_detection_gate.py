from PIL import Image
import pytest

from app.core.exceptions import EyeOccludedError, InvalidArgumentError
from app.services.face_detection import DetectionIssue, DetectionReason, FaceDetectionResult, FAILED, WARNING
from app.services.photo_processor import PhotoProcessor
from app.schemas.generate import GenerateCandidate


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
    assert payload.processStatus == 'success'
    assert payload.complianceStatus == 'warning'
    assert '不建议直接用于正式证件照提交' in payload.complianceMessage


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

    assert exc_info.value.details['resultLevel'] == 'FAIL'
    assert exc_info.value.details['reasons'][0]['code'] == 'EYE_OCCLUDED'
    assert exc_info.value.details['suggestions'] == ['请露出完整双眼与面部']


def test_generate_returns_dual_status_for_warning_result(processor: PhotoProcessor) -> None:
    image = Image.new('RGB', (800, 1000), 'white')
    processor.detector.detect = lambda _image: build_result(
        status=WARNING,
        can_generate=True,
        recommended=False,
        warnings=['单眼疑似遮挡，建议更换无遮挡正脸照片'],
        warning_codes=['EYE_OCCLUDED'],
    )
    def fake_run_candidate_pipeline(**kwargs):
        cid = kwargs['candidate_id']
        return (
            GenerateCandidate(
                candidateId=cid,
                engineKey=cid,
                label='方案A' if cid == 'baidu' else '方案B',
                imagePath='',
                imageUrl='',
                width=413,
                height=579,
                previewPath='',
                previewUrl='',
                qualityStatus='WARNING' if cid == 'baidu' else 'PASS',
                qualityMessage='图片已生成，但仍存在质量风险，不建议直接用于正式证件照提交' if cid == 'baidu' else '输入与输出质量均通过，可用于正式证件照提交',
                outputQualityStatus='PASS',
                outputQualityMessage='输出成片质量正常',
                warnings=['单眼疑似遮挡，建议更换无遮挡正脸照片'],
                safeToSubmit=False if cid == 'baidu' else True,
                debugInfo={'durationMs': 1.0},
            ),
            None,
            {},
            [],
        )

    processor._run_candidate_pipeline = fake_run_candidate_pipeline

    payload = processor.generate(
        image=image,
        size_key='one_inch',
        background_color='blue',
        enhance=False,
        save_output=False,
    )

    assert payload.processStatus == 'generated'
    assert payload.complianceStatus == 'warning'
    assert payload.safeToSubmit is False
    assert payload.qualityStatus == 'WARNING'
    assert '不建议直接用于正式证件照提交' in payload.complianceMessage
    assert '建议放大检查边缘细节' in payload.qualityMessage
    assert payload.processMessage == '已生成两套候选结果，请先选择要保存的图片'
    assert payload.debugInfo is not None
    assert payload.requireUserSelection is True
    assert payload.selectedCandidateId is None
    assert len(payload.candidates) == 2


def test_generate_maps_quality_warning_codes_to_user_facing_messages(processor: PhotoProcessor) -> None:
    image = Image.new('RGB', (800, 1000), 'white')
    processor.detector.detect = lambda _image: build_result(
        status='PASS',
        can_generate=True,
        recommended=True,
        warnings=['建议确认肩部边缘细节'],
        warning_codes=['BAD_COMPOSITION'],
    )
    def fake_run_candidate_pipeline(**kwargs):
        cid = kwargs['candidate_id']
        return (
            GenerateCandidate(
                candidateId=cid,
                engineKey=cid,
                label='方案A' if cid == 'baidu' else '方案B',
                imagePath='',
                imageUrl='',
                width=413,
                height=579,
                previewPath='',
                previewUrl='',
                qualityStatus='WARNING',
                qualityMessage='图片已生成，但仍存在质量风险，不建议直接用于正式证件照提交',
                outputQualityStatus='WARNING',
                outputQualityMessage='衣领或肩部区域有少量底色影响，建议放大查看',
                warnings=['建议确认肩部边缘细节', '衣领或肩部区域有少量底色影响，建议放大查看'],
                safeToSubmit=False,
                debugInfo={'durationMs': 1.0},
            ),
            None,
            {},
            ['CLOTH_COLOR_POLLUTION'],
        )

    processor._run_candidate_pipeline = fake_run_candidate_pipeline

    payload = processor.generate(
        image=image,
        size_key='one_inch',
        background_color='blue',
        enhance=False,
        save_output=False,
    )

    assert '建议确认肩部边缘细节' in payload.warnings
    assert '衣领或肩部区域有少量底色影响，建议放大查看' in payload.warnings
    assert all('enginecomparison' not in warning.lower() for warning in payload.warnings)


def test_generate_marks_final_fail_when_output_quality_fails(processor: PhotoProcessor) -> None:
    image = Image.new('RGB', (800, 1000), 'white')
    processor.detector.detect = lambda _image: build_result(
        status='PASS',
        can_generate=True,
        recommended=True,
    )
    def fake_run_candidate_pipeline(**kwargs):
        cid = kwargs['candidate_id']
        status = 'FAIL' if cid == 'baidu' else 'PASS'
        return (
            GenerateCandidate(
                candidateId=cid,
                engineKey=cid,
                label='方案A' if cid == 'baidu' else '方案B',
                imagePath='',
                imageUrl='',
                width=413,
                height=579,
                previewPath='',
                previewUrl='',
                qualityStatus=status,
                qualityMessage='已生成候选图，但存在明显质量风险，请谨慎选择' if status == 'FAIL' else '输入与输出质量均通过，可用于正式证件照提交',
                outputQualityStatus=status,
                outputQualityMessage='脸部颜色存在轻微异常，建议确认后再保存' if status == 'FAIL' else '输出成片质量正常',
                outputReasonCodes=['FACE_COLOR_POLLUTION'] if status == 'FAIL' else [],
                warnings=[],
                safeToSubmit=False if status == 'FAIL' else True,
                debugInfo={'durationMs': 1.0},
            ),
            None,
            {},
            [],
        )

    processor._run_candidate_pipeline = fake_run_candidate_pipeline

    payload = processor.generate(
        image=image,
        size_key='one_inch',
        background_color='red',
        enhance=False,
        save_output=False,
    )

    assert payload.qualityStatus == 'FAIL'
    assert payload.safeToSubmit is False
    assert payload.candidates[0].qualityStatus in {'FAIL', 'PASS'}
    assert len(payload.candidates) == 2
    assert payload.allowHdSave is False


def test_select_candidate_requires_candidate_id(processor: PhotoProcessor) -> None:
    with pytest.raises(InvalidArgumentError):
        processor.select_candidate(task_id='task_x', candidate_id='')
