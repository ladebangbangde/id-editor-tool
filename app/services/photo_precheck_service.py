from __future__ import annotations

from dataclasses import dataclass, field

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
    primary_issue: str | None = None
    primary_message: str | None = None
    secondary_warnings: list[str] = field(default_factory=list)
    quality_status: str = PASS
    quality_message: str = '照片质量良好，可直接处理'


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
        'EYE_OCCLUDED': '眼部遮挡明显',
        'HAND_OCCLUSION': '手部遮挡面部',
    }

    ISSUE_PRIORITY = {
        'RESOLUTION_TOO_LOW': 100,
        'NO_FACE_DETECTED': 95,
        'MULTIPLE_FACES_DETECTED': 94,
        'IMAGE_TOO_BLURRY': 90,
        'SEVERE_POSE': 80,
        'HEAD_SHOULDER_INCOMPLETE': 78,
        'FACE_RATIO_INVALID': 76,
        'JEWELRY_DETECTED': 74,
        'EXTREME_LIGHTING': 70,
        'NOT_SUITABLE_PORTRAIT': 60,
        'HAND_OCCLUSION': 86,
        'EYE_OCCLUDED': 84,
    }

    QUALITY_MESSAGES = {
        PASS: '照片质量良好，可直接进入处理流程',
        WARNING: '照片可处理，但存在风险项，建议按提示优化后重拍',
        FAIL: '照片暂不适合处理，请根据主问题调整后重试',
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.metrics_service = PhotoMetricsService()
        self._detector = None

    def _face_detector(self):
        if self._detector is not None:
            return self._detector

        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
        return self._detector

    @staticmethod
    def _append_issue(issues: list[PrecheckIssue], code: str, message: str, severity: str) -> None:
        if any(item.code == code and item.severity == severity for item in issues):
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

    def _detect_neck_accessory(self, image: Image.Image, face_box: FaceBox) -> tuple[float, dict[str, float]]:
        cv2 = self.metrics_service._cv2()
        rgb = np.asarray(image.convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]

        neck_left = max(0, int(face_box.x - face_box.width * 0.10))
        neck_right = min(width, int(face_box.x + face_box.width * 1.10))
        neck_top = min(height - 1, int(face_box.y + face_box.height * 0.78))
        neck_bottom = min(height, int(face_box.y + face_box.height * 1.85))

        if neck_right - neck_left < 24 or neck_bottom - neck_top < 24:
            return 0.0, {'neck_edge_density': 0.0, 'neck_bright_ratio': 0.0, 'neck_line_score': 0.0}

        roi = gray[neck_top:neck_bottom, neck_left:neck_right]
        roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(roi_blur, 75, 170)
        edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))

        mean_val = float(np.mean(roi_blur))
        std_val = float(np.std(roi_blur))
        bright_mask = roi_blur > (mean_val + std_val * 1.25)
        bright_ratio = float(np.count_nonzero(bright_mask) / max(bright_mask.size, 1))

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thin_line_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < 12 or area > roi.shape[0] * roi.shape[1] * 0.20:
                continue
            long_side = max(w, h)
            short_side = max(min(w, h), 1)
            elongation = long_side / short_side
            if elongation > 2.8 and short_side <= 18:
                thin_line_count += 1

        line_score = min(thin_line_count / 18.0, 1.0)
        confidence = min(edge_density * 2.8 + bright_ratio * 4.2 + line_score * 0.55, 1.0)

        return confidence, {
            'neck_edge_density': edge_density,
            'neck_bright_ratio': bright_ratio,
            'neck_line_score': line_score,
        }

    def _build_reason(self, code: str, fallback: str) -> PrecheckReason:
        return PrecheckReason(
            code=code,
            title=self.FAILED_REASON_LABELS.get(code, code),
            detail=fallback,
        )

    def _detect_visible_eyes(self, image: Image.Image, face_box: FaceBox) -> int:
        cv2 = self.metrics_service._cv2()
        rgb = np.asarray(image.convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        x0 = max(face_box.x, 0)
        y0 = max(face_box.y, 0)
        x1 = min(face_box.x + face_box.width, gray.shape[1])
        y1 = min(face_box.y + face_box.height, gray.shape[0])
        if x1 - x0 < 24 or y1 - y0 < 24:
            return 0

        face_roi = gray[y0:y1, x0:x1]
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_detector.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(max(14, face_box.width // 10), max(10, face_box.height // 12)),
        )
        return int(len(eyes))

    def _select_primary_issue(self, issues: list[PrecheckIssue]) -> PrecheckIssue | None:
        if not issues:
            return None
        return max(issues, key=lambda issue: self.ISSUE_PRIORITY.get(issue.code, 0))

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
            elif face_width_ratio < 0.155 or face_height_ratio < 0.21:
                self._append_issue(issues, 'FACE_RATIO_INVALID', '人脸偏小，放大后可能损失细节', WARNING)

            if face_width_ratio > 0.72 or face_height_ratio > 0.82:
                self._append_issue(issues, 'FACE_RATIO_INVALID', '人脸过近，构图异常', FAIL)
            elif face_width_ratio > 0.64 or face_height_ratio > 0.74:
                self._append_issue(issues, 'FACE_RATIO_INVALID', '人脸偏近，建议稍远一点拍摄', WARNING)

            if abs(center_x - 0.5) > 0.30 or abs(center_y - 0.5) > 0.34:
                self._append_issue(issues, 'NOT_SUITABLE_PORTRAIT', '人脸偏移过大，难以稳定裁切', FAIL)
            elif abs(center_x - 0.5) > 0.22 or abs(center_y - 0.5) > 0.26:
                self._append_issue(issues, 'NOT_SUITABLE_PORTRAIT', '构图不够标准，建议把人脸放到中间', WARNING)

            if top_margin < 0.008 or bottom_margin < 0.01:
                self._append_issue(issues, 'HEAD_SHOULDER_INCOMPLETE', '头部或肩颈区域贴边严重', FAIL)
            elif top_margin < 0.03 or bottom_margin < 0.04:
                self._append_issue(issues, 'HEAD_SHOULDER_INCOMPLETE', '头肩区域略紧，建议留更多空间', WARNING)

            keypoints = primary_face_box.get('keypoints', []) if primary_face_box else []
            if len(keypoints) >= 2:
                eye_distance = abs(keypoints[1]['x'] - keypoints[0]['x'])
                metrics['eye_distance_ratio'] = eye_distance
                if eye_distance < 0.095:
                    self._append_issue(issues, 'SEVERE_POSE', '明显侧脸或偏转角度过大', FAIL)
                elif eye_distance < 0.14:
                    self._append_issue(issues, 'SEVERE_POSE', '存在轻微姿态偏差，建议更正面', WARNING)

            visible_eyes = self._detect_visible_eyes(image, face_box)
            metrics['visible_eye_count'] = float(visible_eyes)
            if visible_eyes == 0:
                self._append_issue(issues, 'EYE_OCCLUDED', '双眼不可见，疑似有手部或物体遮挡', FAIL)
                self._append_issue(issues, 'HAND_OCCLUSION', '面部遮挡严重，不建议用于正式证件照', FAIL)
            elif visible_eyes == 1:
                self._append_issue(issues, 'EYE_OCCLUDED', '单眼可见，存在明显遮挡风险', WARNING)

            jewelry_conf, jewelry_metrics = self._detect_neck_accessory(image, face_box)
            metrics.update(jewelry_metrics)
            metrics['jewelry_confidence'] = jewelry_conf
            if jewelry_conf >= 0.66:
                self._append_issue(issues, 'JEWELRY_DETECTED', '佩戴项链首饰，建议去除后重拍', WARNING)
                self._append_issue(issues, 'NECK_ACCESSORY', '证件照通常要求颈部无遮挡，避免审核失败', WARNING)

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
        quality_status = status

        reasons = [self._build_reason(issue.code, issue.message) for issue in failed_issues]
        warnings = [issue.message for issue in warning_issues]

        primary_candidates = failed_issues if failed_issues else warning_issues
        primary_issue_obj = self._select_primary_issue(primary_candidates)
        primary_issue = primary_issue_obj.code if primary_issue_obj else None
        primary_message = primary_issue_obj.message if primary_issue_obj else self.QUALITY_MESSAGES[status]

        secondary_warnings = [
            issue.message
            for issue in warning_issues
            if not primary_issue_obj or issue.code != primary_issue_obj.code or issue.message != primary_issue_obj.message
        ]

        logger.info(
            'photo precheck status=%s quality_status=%s face_count=%s primary_issue=%s metrics=%s reason_codes=%s warning_codes=%s',
            status,
            quality_status,
            face_count,
            primary_issue,
            {
                'blur_score': round(blur_score, 2),
                'brightness': round(brightness, 2),
                'face_width_ratio': round(metrics['face_width_ratio'], 4),
                'face_height_ratio': round(metrics['face_height_ratio'], 4),
                'face_center_x': round(metrics['face_center_x'], 4),
                'face_center_y': round(metrics['face_center_y'], 4),
                'edge_density': round(edge_density, 4),
                'jewelry_confidence': round(metrics.get('jewelry_confidence', 0.0), 4),
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
            primary_issue=primary_issue,
            primary_message=primary_message,
            secondary_warnings=secondary_warnings,
            quality_status=quality_status,
            quality_message=self.QUALITY_MESSAGES[status],
        )
