from dataclasses import dataclass, field

from PIL import Image

from app.services.photo_precheck_service import FAIL, PASS, WARNING, PhotoPrecheckService

FAILED = FAIL
PASSED = PASS


@dataclass
class DetectionIssue:
    code: str
    message: str
    severity: str


@dataclass
class DetectionReason:
    code: str
    title: str
    detail: str


@dataclass
class FaceDetectionResult:
    width: int
    height: int
    face_count: int
    has_face: bool
    recommended: bool
    can_generate: bool
    status: str
    result_level: str
    reasons: list[DetectionReason]
    suggestions: list[str]
    reason_codes: list[str]
    warnings: list[str]
    warning_codes: list[str]
    issues: list[DetectionIssue]
    face_boxes: list[dict[str, int]]
    primary_face: dict[str, int] | None
    blur_score: float | None = None
    occlusion_detected: bool = False
    occlusion_areas: list[str] = field(default_factory=list)
    pose_accepted: bool = True
    landmark_stable: bool = True
    composition_accepted: bool = True
    metrics: dict[str, float] = field(default_factory=dict)


class FaceDetectionService:
    SUGGESTIONS_BY_CODE = {
        'NO_FACE_DETECTED': ['请确保画面中只有一位人物并正对镜头'],
        'MULTIPLE_FACES_DETECTED': ['请仅保留单人入镜'],
        'RESOLUTION_TOO_LOW': ['请使用更高分辨率原图'],
        'IMAGE_TOO_BLURRY': ['请重新拍摄并保持稳定对焦'],
        'SEVERE_POSE': ['请尽量正对镜头，减少侧脸角度'],
        'FACE_RATIO_INVALID': ['请调整拍摄距离，让头肩比例更合适'],
        'HEAD_SHOULDER_INCOMPLETE': ['请保证头顶、下巴和肩颈完整入镜'],
        'NOT_SUITABLE_PORTRAIT': ['请使用单人半身或头像照片'],
        'EXTREME_LIGHTING': ['请在光线更均匀的环境中拍摄'],
    }

    def __init__(self) -> None:
        self.precheck = PhotoPrecheckService()

    def detect(self, image: Image.Image) -> FaceDetectionResult:
        result = self.precheck.precheck(image)
        issue_codes = {item.code for item in result.issues}
        failed = [item for item in result.issues if item.severity == FAIL]

        suggestions: list[str] = []
        for issue in failed:
            for suggestion in self.SUGGESTIONS_BY_CODE.get(issue.code, []):
                if suggestion not in suggestions:
                    suggestions.append(suggestion)

        return FaceDetectionResult(
            width=result.width,
            height=result.height,
            face_count=result.face_count,
            has_face=result.face_count > 0,
            recommended=result.status == PASS,
            can_generate=result.status != FAIL,
            status=result.status,
            result_level=result.status,
            reasons=[
                DetectionReason(code=item.code, title=item.title, detail=item.detail)
                for item in result.reasons
            ],
            suggestions=suggestions,
            reason_codes=result.reason_codes,
            warnings=result.warnings,
            warning_codes=result.warning_codes,
            issues=[
                DetectionIssue(code=item.code, message=item.message, severity=item.severity)
                for item in result.issues
            ],
            face_boxes=result.face_boxes,
            primary_face=result.primary_face,
            blur_score=result.metrics.get('blur_score'),
            occlusion_detected=False,
            occlusion_areas=[],
            pose_accepted='SEVERE_POSE' not in issue_codes,
            landmark_stable='IMAGE_TOO_BLURRY' not in issue_codes,
            composition_accepted='NOT_SUITABLE_PORTRAIT' not in issue_codes and 'FACE_RATIO_INVALID' not in issue_codes,
            metrics=result.metrics,
        )
