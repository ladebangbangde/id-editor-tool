from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.photo_metrics_service import FaceBox, PhotoMetricsService

logger = get_logger(__name__)

PASS = 'PASS'
WARNING = 'WARNING'
FAIL = 'FAIL'


@dataclass
class PrecheckIssue:
    code: str
    message: str
    severity: str


@dataclass
class PrecheckReason:
    code: str
    title: str
    detail: str


@dataclass
class PhotoPrecheckResult:
    width: int
    height: int
    face_count: int
    status: str
    reasons: list[PrecheckReason]
    warnings: list[str]
    reason_codes: list[str]
    warning_codes: list[str]
    metrics: dict[str, float]
    face_boxes: list[dict[str, int]]
    primary_face: dict[str, int] | None
    issues: list[PrecheckIssue]


class PhotoPrecheckService:
    FAILED_REASON_LABELS = {
        'NO_FACE_DETECTED': '未检测到人脸',
        'MULTIPLE_FACES_DETECTED': '检测到多人',
        'RESOLUTION_TOO_LOW': '图片分辨率过低',
        'IMAGE_TOO_BLURRY': '图片严重模糊',
        'SEVERE_POSE': '人脸姿态偏转过大',
        'FACE_RATIO_INVALID': '人脸比例异常',
        'HEAD_SHOULDER_INCOMPLETE': '头肩区域缺失严重',
        'NOT_SUITABLE_PORTRAIT': '图片不适合做人像处理',
        'EXTREME_LIGHTING': '光照异常严重',
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.metrics_service = PhotoMetricsService()
        self._detector = None

    def _face_detector(self):
        if self._detector is not None:
            return self._detector

        # Lazy import to keep startup light and avoid import-time failures.
        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
        return self._detector

    @staticmethod
    def _append_issue(issues: list[PrecheckIssue], code: str, message: str, severity: str) -> None:
        if any(item.code == code for item in issues):
            return
        issues.append(PrecheckIssue(code=code, message=message, severity=severity))

    @staticmethod
    def _clamp_box(box: FaceBox, width: int, height: int) -> FaceBox:
        x = max(0, min(width - 1, box.x))
        y = max(0, min(height - 1, box.y))
        right = max(x + 1, min(width, box.x + box.width))
        bottom = max(y + 1, min(height, box.y + box.height))
        return FaceBox(x=x, y=y, width=right - x, height=bottom - y)

    def _detect_faces(self, image: Image.Image) -> list[dict]:
        rgb = np.asarray(image.convert('RGB'))
        detector = self._face_detector()
        result = detector.process(rgb)

        width, height = image.size
        boxes: list[dict] = []
        for detection in result.detections or []:
            box = detection.location_data.relative_bounding_box
            face_box = self._clamp_box(
                FaceBox(
                    x=int(box.xmin * width),
                    y=int(box.ymin * height),
                    width=int(box.width * width),
                    height=int(box.height * height),
                ),
                width,
                height,
            )
            boxes.append(
                {
                    'x': face_box.x,
                    'y': face_box.y,
                    'width': face_box.width,
                    'height': face_box.height,
                    'score': float(detection.score[0]) if detection.score else 0.0,
                    'keypoints': [
                        {'x': float(kp.x), 'y': float(kp.y)}
                        for kp in detection.location_data.relative_keypoints
                    ],
                }
            )
        return boxes

    def _build_reason(self, code: str, fallback: str) -> PrecheckReason:
        return PrecheckReason(
            code=code,
            title=self.FAILED_REASON_LABELS.get(code, code),
            detail=fallback,
        )

    def precheck(self, image: Image.Image) -> PhotoPrecheckResult:
        width, height = image.size
        issues: list[PrecheckIssue] = []

        if width < self.settings.min_image_width or height < self.settings.min_image_height:
            self._append_issue(
                issues,
                'RESOLUTION_TOO_LOW',
                f'分辨率不足，至少需要 {self.settings.min_image_width}x{self.settings.min_image_height}px',
                FAIL,
            )

        face_boxes = self._detect_faces(image)
        face_count = len(face_boxes)
        primary_face_box = max(face_boxes, key=lambda b: b['width'] * b['height']) if face_boxes else None
        face_box = FaceBox(**{k: primary_face_box[k] for k in ('x', 'y', 'width', 'height')}) if primary_face_box else None
        metrics = self.metrics_service.calculate(image, face_box)
        metrics['face_count'] = float(face_count)

        if face_count == 0:
            self._append_issue(issues, 'NO_FACE_DETECTED', '未识别到清晰单人面部', FAIL)
        elif face_count > 1:
            self._append_issue(issues, 'MULTIPLE_FACES_DETECTED', '检测到多人，请上传单人照片', FAIL)

        if face_box is not None:
            face_width_ratio = metrics['face_width_ratio']
            face_height_ratio = metrics['face_height_ratio']
            center_x = metrics['face_center_x']
            center_y = metrics['face_center_y']
            top_margin = face_box.y / max(height, 1)
            bottom_margin = max(height - (face_box.y + face_box.height), 0) / max(height, 1)

            if face_width_ratio < 0.10 or face_height_ratio < 0.14:
                self._append_issue(issues, 'FACE_RATIO_INVALID', '人脸过小，后续处理无法保证质量', FAIL)
            elif face_width_ratio < 0.16 or face_height_ratio < 0.22:
                self._append_issue(issues, 'FACE_RATIO_INVALID', '人脸偏小，放大后可能损失细节', WARNING)

            if face_width_ratio > 0.72 or face_height_ratio > 0.82:
                self._append_issue(issues, 'FACE_RATIO_INVALID', '人脸过近，构图异常', FAIL)
            elif face_width_ratio > 0.62 or face_height_ratio > 0.72:
                self._append_issue(issues, 'FACE_RATIO_INVALID', '人脸偏近，建议稍远一点拍摄', WARNING)

            if abs(center_x - 0.5) > 0.30 or abs(center_y - 0.5) > 0.34:
                self._append_issue(issues, 'NOT_SUITABLE_PORTRAIT', '人脸偏移过大，难以稳定裁切', FAIL)
            elif abs(center_x - 0.5) > 0.20 or abs(center_y - 0.5) > 0.24:
                self._append_issue(issues, 'NOT_SUITABLE_PORTRAIT', '构图不够标准，建议把人脸放到中间', WARNING)

            if top_margin < 0.008 or bottom_margin < 0.01:
                self._append_issue(issues, 'HEAD_SHOULDER_INCOMPLETE', '头部或肩颈区域贴边严重', FAIL)
            elif top_margin < 0.03 or bottom_margin < 0.04:
                self._append_issue(issues, 'HEAD_SHOULDER_INCOMPLETE', '头肩区域略紧，建议留更多空间', WARNING)

            keypoints = primary_face_box.get('keypoints', []) if primary_face_box else []
            if len(keypoints) >= 2:
                eye_distance = abs(keypoints[1]['x'] - keypoints[0]['x'])
                if eye_distance < 0.12:
                    self._append_issue(issues, 'SEVERE_POSE', '明显侧脸或偏转角度过大', FAIL)
                elif eye_distance < 0.18:
                    self._append_issue(issues, 'SEVERE_POSE', '存在侧脸倾向，建议更正面', WARNING)

        blur_score = metrics['blur_score']
        brightness = metrics['brightness']
        edge_density = metrics['edge_density']

        if blur_score < 45:
            self._append_issue(issues, 'IMAGE_TOO_BLURRY', '画面严重模糊，无法可靠处理', FAIL)
        elif blur_score < 85:
            self._append_issue(issues, 'IMAGE_TOO_BLURRY', '清晰度一般，成片锐度可能不足', WARNING)

        if brightness < 35 or brightness > 225:
            self._append_issue(issues, 'EXTREME_LIGHTING', '光照极端异常', FAIL)
        elif brightness < 60 or brightness > 205:
            self._append_issue(issues, 'EXTREME_LIGHTING', '光照条件一般，建议调整后再拍', WARNING)

        if edge_density > 0.24:
            self._append_issue(issues, 'NOT_SUITABLE_PORTRAIT', '背景复杂，可能影响观感', WARNING)

        failed_issues = [issue for issue in issues if issue.severity == FAIL]
        warning_issues = [issue for issue in issues if issue.severity == WARNING]
        status = FAIL if failed_issues else WARNING if warning_issues else PASS

        reasons = [self._build_reason(issue.code, issue.message) for issue in failed_issues]
        warnings = [issue.message for issue in warning_issues]

        logger.info(
            'photo precheck status=%s face_count=%s metrics=%s reason_codes=%s warning_codes=%s',
            status,
            face_count,
            {
                'blur_score': round(blur_score, 2),
                'brightness': round(brightness, 2),
                'face_width_ratio': round(metrics['face_width_ratio'], 4),
                'face_height_ratio': round(metrics['face_height_ratio'], 4),
                'face_center_x': round(metrics['face_center_x'], 4),
                'face_center_y': round(metrics['face_center_y'], 4),
                'edge_density': round(edge_density, 4),
            },
            [item.code for item in failed_issues],
            [item.code for item in warning_issues],
        )

        return PhotoPrecheckResult(
            width=width,
            height=height,
            face_count=face_count,
            status=status,
            reasons=reasons,
            warnings=warnings,
            reason_codes=[item.code for item in failed_issues],
            warning_codes=[item.code for item in warning_issues],
            metrics=metrics,
            face_boxes=[{k: int(v) for k, v in box.items() if k in {'x', 'y', 'width', 'height'}} for box in face_boxes],
            primary_face={k: int(v) for k, v in primary_face_box.items() if k in {'x', 'y', 'width', 'height'}} if primary_face_box else None,
            issues=issues,
        )
