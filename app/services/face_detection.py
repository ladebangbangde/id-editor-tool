from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from skimage import color, data, filters, transform
from skimage.feature import Cascade

from app.core.config import get_settings
from app.core.exceptions import ImageTooSmallError


FAILED = 'FAILED'
WARNING = 'WARNING'
PASSED = 'PASSED'


@dataclass
class DetectionIssue:
    code: str
    message: str
    severity: str


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
    reasons: list[str]
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
    FAILED_REASON_LABELS = {
        'NO_FACE_DETECTED': '未检测到可用正脸',
        'MULTIPLE_FACES_DETECTED': '画面中存在多张人脸',
        'FACE_OCCLUDED': '面部关键区域被遮挡',
        'EYE_OCCLUDED': '单眼或双眼被遮挡',
        'INVALID_POSE': '非标准正脸姿态',
        'LANDMARK_UNSTABLE': '关键点检测不稳定',
        'BAD_COMPOSITION': '构图不适合证件照裁切',
    }
    SUGGESTIONS_BY_CODE = {
        'NO_FACE_DETECTED': ['请正对镜头拍摄', '请让人脸位于画面中央并保留完整头部'],
        'MULTIPLE_FACES_DETECTED': ['请仅保留一位拍摄对象', '请让人脸位于画面中央并保留完整头部'],
        'FACE_OCCLUDED': ['请露出完整双眼与面部', '请避免手、头发遮挡五官'],
        'EYE_OCCLUDED': ['请露出完整双眼与面部', '请避免手、头发遮挡五官'],
        'INVALID_POSE': ['请正对镜头拍摄'],
        'LANDMARK_UNSTABLE': ['请正对镜头拍摄', '请让人脸位于画面中央并保留完整头部'],
        'BAD_COMPOSITION': ['请让人脸位于画面中央并保留完整头部'],
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.detector = Cascade(data.lbp_frontal_face_cascade_filename())

    @staticmethod
    def _append_issue(
        issues: list[DetectionIssue],
        code: str,
        message: str,
        severity: str,
    ) -> None:
        if any(issue.code == code and issue.severity == severity for issue in issues):
            return
        issues.append(DetectionIssue(code=code, message=message, severity=severity))

    @staticmethod
    def _clip_box(box: dict[str, int], width: int, height: int) -> dict[str, int]:
        left = max(0, min(width - 1, box['x']))
        top = max(0, min(height - 1, box['y']))
        right = max(left + 1, min(width, box['x'] + box['width']))
        bottom = max(top + 1, min(height, box['y'] + box['height']))
        return {'x': left, 'y': top, 'width': right - left, 'height': bottom - top}

    @staticmethod
    def _region_from_ratio(face: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        height, width = face.shape
        left = max(0, min(width - 1, int(round(width * x0))))
        top = max(0, min(height - 1, int(round(height * y0))))
        right = max(left + 1, min(width, int(round(width * x1))))
        bottom = max(top + 1, min(height, int(round(height * y1))))
        return face[top:bottom, left:right]

    @staticmethod
    def _region_metrics(region: np.ndarray, face_mean: float) -> dict[str, float]:
        if region.size == 0:
            return {'edge': 0.0, 'contrast': 0.0, 'dark_ratio': 1.0}
        grad_y, grad_x = np.gradient(region)
        edge = float(np.mean(np.hypot(grad_x, grad_y)))
        contrast = float(np.std(region))
        dark_ratio = float(np.mean(region < max(face_mean * 0.72, 0.18)))
        return {'edge': edge, 'contrast': contrast, 'dark_ratio': dark_ratio}

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        return float(np.var(filters.laplace(gray)))

    def _build_failed_reasons_and_suggestions(self, failed_issues: list[DetectionIssue]) -> tuple[list[str], list[str]]:
        reasons: list[str] = []
        suggestions: list[str] = []
        seen_reasons: set[str] = set()
        seen_suggestions: set[str] = set()

        for issue in failed_issues:
            reason = self.FAILED_REASON_LABELS.get(issue.code, issue.message)
            if reason not in seen_reasons:
                reasons.append(reason)
                seen_reasons.add(reason)
            for suggestion in self.SUGGESTIONS_BY_CODE.get(issue.code, []):
                if suggestion not in seen_suggestions:
                    suggestions.append(suggestion)
                    seen_suggestions.add(suggestion)

        return reasons, suggestions

    def _prepare_gray(self, image: Image.Image) -> tuple[np.ndarray, float]:
        width, height = image.size
        rgb = np.asarray(image.convert('RGB'))
        gray = color.rgb2gray(rgb)
        scale = 1.0
        max_side = max(width, height)
        if max_side > 1200:
            scale = 1200.0 / max_side
            gray = transform.rescale(gray, scale, anti_aliasing=True)
        return gray, scale

    def _analyze_single_face(
        self,
        gray: np.ndarray,
        width: int,
        height: int,
        face_box: dict[str, int],
    ) -> tuple[list[DetectionIssue], dict[str, float], list[str]]:
        issues: list[DetectionIssue] = []
        occlusion_areas: list[str] = []
        box = self._clip_box(face_box, width, height)
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        face = gray[y : y + h, x : x + w]
        if face.size == 0 or w < 40 or h < 40:
            self._append_issue(issues, 'LANDMARK_UNSTABLE', '人脸关键区域过小，关键点无法稳定定位', FAILED)
            return issues, {}, occlusion_areas

        face_resized = transform.resize(face, (200, 160), anti_aliasing=True)
        face_mean = float(np.mean(face_resized))
        face_std = float(np.std(face_resized))
        face_edge = self._laplacian_variance(face_resized)
        left_eye = self._region_metrics(self._region_from_ratio(face_resized, 0.14, 0.18, 0.40, 0.38), face_mean)
        right_eye = self._region_metrics(self._region_from_ratio(face_resized, 0.60, 0.18, 0.86, 0.38), face_mean)
        nose = self._region_metrics(self._region_from_ratio(face_resized, 0.36, 0.34, 0.64, 0.60), face_mean)
        mouth = self._region_metrics(self._region_from_ratio(face_resized, 0.28, 0.60, 0.72, 0.82), face_mean)
        chin = self._region_metrics(self._region_from_ratio(face_resized, 0.28, 0.80, 0.72, 0.96), face_mean)
        forehead = self._region_metrics(self._region_from_ratio(face_resized, 0.25, 0.03, 0.75, 0.20), face_mean)

        left_score = left_eye['edge'] + left_eye['contrast']
        right_score = right_eye['edge'] + right_eye['contrast']
        face_score_floor = max(face_edge * 0.32 + face_std * 0.22, 0.018)
        eye_balance = min(left_score, right_score) / max(max(left_score, right_score), 1e-6)
        symmetry_score = float(
            np.mean(
                np.abs(
                    face_resized[:, : face_resized.shape[1] // 2]
                    - np.fliplr(face_resized[:, face_resized.shape[1] - face_resized.shape[1] // 2 :])
                )
            )
        )
        symmetry_ratio = symmetry_score / max(face_std, 1e-6)
        central_score = nose['edge'] + mouth['edge'] + mouth['contrast']
        central_floor = max(face_edge * 0.28 + face_std * 0.18, 0.016)

        if min(left_score, right_score) < face_score_floor:
            occlusion_areas.append('eyes')
            if eye_balance < 0.55 or max(left_eye['dark_ratio'], right_eye['dark_ratio']) > 0.65:
                self._append_issue(issues, 'EYE_OCCLUDED', '双眼或单眼存在明显遮挡，无法满足证件照要求', FAILED)
            else:
                self._append_issue(issues, 'LANDMARK_UNSTABLE', '眼部关键点稳定性不足，后续裁切风险较高', WARNING)
        elif eye_balance < 0.72:
            self._append_issue(issues, 'LANDMARK_UNSTABLE', '双眼特征不对称，关键点定位存在风险', WARNING)

        if forehead['edge'] > face_edge * 1.45 and min(left_score, right_score) < face_score_floor * 1.15:
            occlusion_areas.append('forehead')
            self._append_issue(issues, 'FACE_OCCLUDED', '头发或其他遮挡覆盖眉眼区域，不适合证件照生成', FAILED)

        if central_score < central_floor or mouth['dark_ratio'] > 0.7 or chin['dark_ratio'] > 0.78:
            occlusion_areas.append('nose_mouth_chin')
            self._append_issue(issues, 'FACE_OCCLUDED', '鼻子、嘴巴或下巴区域存在明显遮挡，无法安全生成证件照', FAILED)

        if face_edge < 0.0012 or face_std < 0.035:
            self._append_issue(issues, 'LANDMARK_UNSTABLE', '人脸细节不足或过于模糊，关键点检测不稳定', FAILED)
        elif face_edge < 0.0022:
            self._append_issue(issues, 'LANDMARK_UNSTABLE', '人脸清晰度一般，关键点稳定性存在一定风险', WARNING)

        if symmetry_ratio > 1.95:
            self._append_issue(issues, 'INVALID_POSE', '人脸左右差异过大，疑似明显侧脸或偏转过大', FAILED)
        elif symmetry_ratio > 1.45:
            self._append_issue(issues, 'INVALID_POSE', '人脸姿态存在偏转，建议使用更正的正脸照片', WARNING)

        metrics = {
            'faceHeightRatio': h / max(height, 1),
            'faceWidthRatio': w / max(width, 1),
            'centerOffsetRatio': abs((x + w / 2) - width / 2) / max(width, 1),
            'topMarginRatio': y / max(height, 1),
            'bottomMarginRatio': max(height - (y + h), 0) / max(height, 1),
            'leftMarginRatio': x / max(width, 1),
            'rightMarginRatio': max(width - (x + w), 0) / max(width, 1),
            'aspectRatio': w / max(h, 1),
            'symmetryRatio': symmetry_ratio,
            'eyeBalance': eye_balance,
            'blurScore': self._laplacian_variance(gray),
        }
        return issues, metrics, sorted(set(occlusion_areas))

    def detect(self, image: Image.Image) -> FaceDetectionResult:
        width, height = image.size
        if width < self.settings.min_image_width or height < self.settings.min_image_height:
            raise ImageTooSmallError(
                f'Image is too small: minimum is '
                f'{self.settings.min_image_width}x{self.settings.min_image_height}px'
            )

        gray, scale = self._prepare_gray(image)
        detections = self.detector.detect_multi_scale(
            img=gray,
            scale_factor=1.2,
            step_ratio=1,
            min_size=(60, 60),
            max_size=(int(gray.shape[0] * 0.9), int(gray.shape[1] * 0.9)),
        )

        face_boxes: list[dict[str, int]] = []
        for item in detections:
            x = int(item['c'] / scale)
            y = int(item['r'] / scale)
            w = int(item['width'] / scale)
            h = int(item['height'] / scale)
            face_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})

        face_count = len(face_boxes)
        issues: list[DetectionIssue] = []
        metrics: dict[str, float] = {}
        occlusion_areas: list[str] = []

        if face_count == 0:
            self._append_issue(issues, 'NO_FACE_DETECTED', '未检测到可用于证件照的单人正脸', FAILED)
        elif face_count > 1:
            self._append_issue(issues, 'MULTIPLE_FACES_DETECTED', '检测到多张人脸，请上传单人照片', FAILED)
        else:
            primary_face = max(face_boxes, key=lambda box: box['width'] * box['height'])
            single_face_issues, metrics, occlusion_areas = self._analyze_single_face(
                np.asarray(image.convert('L'), dtype=np.float32) / 255.0,
                width,
                height,
                primary_face,
            )
            issues.extend(single_face_issues)

            face_height_ratio = primary_face['height'] / max(height, 1)
            center_offset_ratio = abs((primary_face['x'] + primary_face['width'] / 2) - width / 2) / max(width, 1)
            top_margin_ratio = primary_face['y'] / max(height, 1)
            bottom_margin_ratio = max(height - (primary_face['y'] + primary_face['height']), 0) / max(height, 1)
            left_margin_ratio = primary_face['x'] / max(width, 1)
            right_margin_ratio = max(width - (primary_face['x'] + primary_face['width']), 0) / max(width, 1)
            aspect_ratio = primary_face['width'] / max(primary_face['height'], 1)

            if face_height_ratio < 0.22:
                self._append_issue(issues, 'BAD_COMPOSITION', '人脸在画面中占比过小，裁切后容易放大失真', FAILED)
            elif face_height_ratio < 0.28:
                self._append_issue(issues, 'BAD_COMPOSITION', '人脸占比偏小，生成时存在放大风险', WARNING)

            if center_offset_ratio > 0.18:
                self._append_issue(issues, 'BAD_COMPOSITION', '人脸明显偏离画面中心，容易裁成半张脸', FAILED)
            elif center_offset_ratio > 0.11:
                self._append_issue(issues, 'BAD_COMPOSITION', '人脸位置略偏，后续裁切存在一定风险', WARNING)

            if min(left_margin_ratio, right_margin_ratio) < 0.02:
                self._append_issue(issues, 'BAD_COMPOSITION', '人脸过于贴近画面边缘，无法安全裁切证件照', FAILED)
            elif min(left_margin_ratio, right_margin_ratio) < 0.05:
                self._append_issue(issues, 'BAD_COMPOSITION', '人脸靠近画面边缘，裁切容错较低', WARNING)

            if top_margin_ratio < 0.02 or bottom_margin_ratio < 0.015:
                self._append_issue(issues, 'BAD_COMPOSITION', '头顶或下巴距离边缘过近，存在严重截断风险', FAILED)
            elif top_margin_ratio < 0.045 or bottom_margin_ratio < 0.04:
                self._append_issue(issues, 'BAD_COMPOSITION', '头顶或下巴留白不足，裁切风险较高', WARNING)

            if top_margin_ratio > 0.38:
                self._append_issue(issues, 'BAD_COMPOSITION', '头顶留白过多，当前构图不适合直接裁成证件照', FAILED)
            elif top_margin_ratio > 0.30:
                self._append_issue(issues, 'BAD_COMPOSITION', '头顶留白偏多，证件照构图不够理想', WARNING)

            if aspect_ratio < 0.62 or aspect_ratio > 1.08:
                self._append_issue(issues, 'INVALID_POSE', '人脸姿态明显偏离正脸，不符合证件照要求', FAILED)
            elif aspect_ratio < 0.70 or aspect_ratio > 0.98:
                self._append_issue(issues, 'INVALID_POSE', '人脸姿态存在轻微偏转，建议更换更正的正脸照片', WARNING)

        primary_face = max(face_boxes, key=lambda box: box['width'] * box['height']) if face_boxes else None
        failed_issues = [issue for issue in issues if issue.severity == FAILED]
        warning_issues = [issue for issue in issues if issue.severity == WARNING]
        status = FAILED if failed_issues else WARNING if warning_issues else PASSED
        reasons, suggestions = self._build_failed_reasons_and_suggestions(failed_issues)
        reason_codes = [issue.code for issue in failed_issues]
        warnings = [issue.message for issue in warning_issues]
        warning_codes = [issue.code for issue in warning_issues]
        issue_codes = {issue.code for issue in issues}
        blur_score = metrics.get('blurScore')

        return FaceDetectionResult(
            width=width,
            height=height,
            face_count=face_count,
            has_face=face_count > 0,
            recommended=status == PASSED,
            can_generate=status != FAILED,
            status=status,
            result_level=status,
            reasons=reasons,
            suggestions=suggestions,
            reason_codes=reason_codes,
            warnings=warnings,
            warning_codes=warning_codes,
            issues=issues,
            face_boxes=face_boxes,
            primary_face=primary_face,
            blur_score=blur_score,
            occlusion_detected=bool({'FACE_OCCLUDED', 'EYE_OCCLUDED'} & issue_codes),
            occlusion_areas=occlusion_areas,
            pose_accepted='INVALID_POSE' not in issue_codes,
            landmark_stable='LANDMARK_UNSTABLE' not in issue_codes,
            composition_accepted='BAD_COMPOSITION' not in issue_codes,
            metrics=metrics,
        )
