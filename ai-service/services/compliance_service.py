from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.exceptions import (
    ERROR_EYE_OCCLUDED,
    ERROR_FACE_OCCLUDED,
    ERROR_FACIAL_KEYPOINTS_INCOMPLETE,
    ERROR_HEADWEAR_DETECTED,
    ERROR_NOT_SINGLE_FRONTAL_FACE,
    ERROR_POSE_INVALID,
)
from utils.config import get_settings


@dataclass
class ComplianceDetail:
    code: str
    message: str
    status: str = 'failed'
    stage: str = 'compliance'
    score: float | None = None
    threshold: float | None = None
    severity_rank: int = 0  # 0: failed > 1: warning > 2: info
    priority_rank: int = 99  # 证件照合规问题优先于画质提示


class ComplianceService:
    """证件照合规审核：睁眼、姿态、关键点与遮挡均走保守策略。"""

    ISSUE_PRIORITY = {
        ERROR_NOT_SINGLE_FRONTAL_FACE: 0,
        ERROR_EYE_OCCLUDED: 1,
        ERROR_POSE_INVALID: 2,
        ERROR_FACE_OCCLUDED: 3,
        ERROR_HEADWEAR_DETECTED: 4,
        ERROR_FACIAL_KEYPOINTS_INCOMPLETE: 5,
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        cascades = cv2.data.haarcascades
        self.eye_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_mcs_mouth.xml')
        self.smile_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_smile.xml')
        self.profile_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_profileface.xml')
        self._empty_cascades = {
            'eyes': self.eye_cascade.empty(),
            'nose': self.nose_cascade.empty(),
            'mouth': self.mouth_cascade.empty(),
            'smile': self.smile_cascade.empty(),
            'profile': self.profile_cascade.empty(),
        }

    @staticmethod
    def _safe_confidence(raw_weight: float | None) -> float:
        if raw_weight is None:
            return 0.0
        return max(0.0, min(1.0, float(raw_weight) / 8.0))

    @staticmethod
    def _clamp_roi(image: np.ndarray, box: dict) -> np.ndarray:
        x = max(int(box['x']), 0)
        y = max(int(box['y']), 0)
        w = max(int(box['width']), 1)
        h = max(int(box['height']), 1)
        return image[y : y + h, x : x + w]

    @staticmethod
    def _detail_payload(item: ComplianceDetail) -> dict:
        return {
            'code': item.code,
            'message': item.message,
            'status': item.status,
            'stage': item.stage,
            'score': item.score,
            'threshold': item.threshold,
        }

    def _sort_details(self, details: list[ComplianceDetail]) -> list[ComplianceDetail]:
        return sorted(
            details,
            key=lambda item: (
                item.severity_rank,
                item.priority_rank,
            ),
        )

    def _detect_confidence(self, cascade: cv2.CascadeClassifier, gray_face: np.ndarray, min_size: tuple[int, int]) -> float:
        if cascade.empty():
            return 0.0
        try:
            _rects, _reject, weights = cascade.detectMultiScale3(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=min_size,
                outputRejectLevels=True,
            )
        except Exception:
            weights = []
        if len(weights) == 0:
            return 0.0
        return self._safe_confidence(max(float(w) for w in weights))

    def _detect_keypoints(self, gray_face: np.ndarray) -> tuple[dict[str, float], list[ComplianceDetail], list[str]]:
        details: list[ComplianceDetail] = []
        warnings: list[str] = []

        face_h, face_w = gray_face.shape[:2]
        eye_conf = self._detect_confidence(self.eye_cascade, gray_face, (max(12, face_w // 9), max(12, face_h // 10)))
        nose_conf = self._detect_confidence(self.nose_cascade, gray_face, (max(16, face_w // 8), max(16, face_h // 8)))
        mouth_conf = self._detect_confidence(self.mouth_cascade, gray_face, (max(20, face_w // 7), max(20, face_h // 8)))
        if mouth_conf <= 0:
            mouth_conf = self._detect_confidence(self.smile_cascade, gray_face, (max(20, face_w // 7), max(20, face_h // 8)))

        conf_map = {
            'eyes': eye_conf,
            'nose': nose_conf,
            'mouth': mouth_conf,
        }
        missing = [k for k, v in conf_map.items() if v < self.settings.landmark_confidence_threshold]
        if 'eyes' in missing:
            details.append(
                ComplianceDetail(
                    code=ERROR_FACIAL_KEYPOINTS_INCOMPLETE,
                    message='双眼关键点置信度不足，无法确认睁眼状态，请更换正视且无遮挡照片',
                    status='failed',
                    stage='keypoint_detection',
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_FACIAL_KEYPOINTS_INCOMPLETE],
                )
            )
        non_critical_missing = [item for item in missing if item != 'eyes']
        if non_critical_missing:
            warnings.append(f"关键点低置信度(鼻/嘴): {', '.join(non_critical_missing)}")
        if self._empty_cascades['nose'] or self._empty_cascades['mouth']:
            unavailable = [name for name in ('nose', 'mouth') if self._empty_cascades[name]]
            warnings.append(f"关键点分类器缺失: {', '.join(unavailable)}")
        return conf_map, details, warnings

    @staticmethod
    def _eye_open_ratio(eye_box: tuple[int, int, int, int]) -> float:
        _, _, w, h = eye_box
        return float(h) / max(float(w), 1.0)

    def _analyze_eye_state(self, gray_face: np.ndarray, keypoint_conf: dict[str, float]) -> tuple[list[ComplianceDetail], dict]:
        details: list[ComplianceDetail] = []
        eye_metrics: dict = {'left': None, 'right': None, 'asymmetry': None}
        if self.eye_cascade.empty():
            return details, eye_metrics

        face_h, face_w = gray_face.shape[:2]
        min_size = (max(14, face_w // 10), max(10, face_h // 12))
        eyes = self.eye_cascade.detectMultiScale(gray_face, scaleFactor=1.08, minNeighbors=4, minSize=min_size)
        candidate_eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)
        if len(candidate_eyes) < 2:
            # 睁眼无法稳定估计时保持保守，不直接放行。
            details.append(
                ComplianceDetail(
                    code=ERROR_EYE_OCCLUDED,
                    message='眼睛区域检测不稳定，无法确认睁眼状态，请使用双眼清晰可见的正脸照片',
                    status='warning',
                    stage='eye_state',
                    severity_rank=1,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_EYE_OCCLUDED],
                )
            )
            return details, eye_metrics

        top_eyes = sorted(candidate_eyes[:2], key=lambda e: e[0])
        left_eye, right_eye = top_eyes[0], top_eyes[1]
        left_ratio = self._eye_open_ratio(left_eye)
        right_ratio = self._eye_open_ratio(right_eye)
        asymmetry = abs(left_ratio - right_ratio)
        eye_metrics = {'left': round(left_ratio, 3), 'right': round(right_ratio, 3), 'asymmetry': round(asymmetry, 3)}

        low_confidence = keypoint_conf.get('eyes', 1.0) < self.settings.eye_confidence_threshold
        fail_th = self.settings.eye_open_ratio_fail_threshold
        warn_th = self.settings.eye_open_ratio_warn_threshold

        both_closed = left_ratio <= fail_th and right_ratio <= fail_th
        single_closed = (left_ratio <= fail_th < right_ratio) or (right_ratio <= fail_th < left_ratio)
        heavy_squint = min(left_ratio, right_ratio) <= warn_th

        if both_closed:
            details.append(
                ComplianceDetail(
                    code=ERROR_EYE_OCCLUDED,
                    message='检测到双眼明显闭合，当前照片不适合作为证件照原图',
                    status='failed',
                    stage='eye_state',
                    score=round(min(left_ratio, right_ratio), 3),
                    threshold=fail_th,
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_EYE_OCCLUDED],
                )
            )
        elif single_closed:
            details.append(
                ComplianceDetail(
                    code=ERROR_EYE_OCCLUDED,
                    message='检测到单眼闭合或严重眯眼，证件照审核高概率不通过',
                    status='failed',
                    stage='eye_state',
                    score=round(min(left_ratio, right_ratio), 3),
                    threshold=fail_th,
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_EYE_OCCLUDED],
                )
            )
        elif heavy_squint or low_confidence:
            details.append(
                ComplianceDetail(
                    code=ERROR_EYE_OCCLUDED,
                    message='双眼开合偏小或检测置信度偏低，存在闭眼风险，建议更换睁眼更自然照片',
                    status='warning',
                    stage='eye_state',
                    score=round(min(left_ratio, right_ratio), 3),
                    threshold=warn_th,
                    severity_rank=1,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_EYE_OCCLUDED],
                )
            )

        if asymmetry >= self.settings.eye_asymmetry_warn_threshold:
            details.append(
                ComplianceDetail(
                    code=ERROR_EYE_OCCLUDED,
                    message='左右眼开合明显不对称，建议正视镜头并保持双眼自然睁开',
                    status='warning',
                    stage='eye_state',
                    score=round(asymmetry, 3),
                    threshold=self.settings.eye_asymmetry_warn_threshold,
                    severity_rank=1,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_EYE_OCCLUDED],
                )
            )
        return details, eye_metrics

    def _analyze_pose(self, gray_face: np.ndarray) -> tuple[list[ComplianceDetail], dict]:
        details: list[ComplianceDetail] = []
        face_h, face_w = gray_face.shape[:2]
        metrics = {'roll': None, 'yaw': None, 'pitch': None}
        if self.eye_cascade.empty():
            return details, metrics

        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(max(14, face_w // 10), max(10, face_h // 12)),
        )
        if len(eyes) < 2:
            return details, metrics

        top_eyes = sorted(sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2], key=lambda e: e[0])
        left_eye, right_eye = top_eyes[0], top_eyes[1]
        left_center = np.array([left_eye[0] + left_eye[2] / 2.0, left_eye[1] + left_eye[3] / 2.0], dtype=np.float32)
        right_center = np.array([right_eye[0] + right_eye[2] / 2.0, right_eye[1] + right_eye[3] / 2.0], dtype=np.float32)
        interocular = float(np.linalg.norm(right_center - left_center))
        if interocular <= 1:
            return details, metrics

        roll = float(np.degrees(np.arctan2(right_center[1] - left_center[1], right_center[0] - left_center[0])))
        metrics['roll'] = round(abs(roll), 2)

        # yaw: 使用鼻子中心相对于双眼中点的水平偏移，归一化到眼间距。
        noses = self.nose_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(max(16, face_w // 8), max(16, face_h // 8)),
        )
        if len(noses) > 0:
            nose = max(noses, key=lambda n: n[2] * n[3])
            nose_center_x = float(nose[0] + nose[2] / 2.0)
            eye_mid_x = float((left_center[0] + right_center[0]) / 2.0)
            yaw = abs(nose_center_x - eye_mid_x) / interocular
            metrics['yaw'] = round(yaw, 3)
        else:
            yaw = 0.0

        mouths = self.mouth_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(max(20, face_w // 7), max(20, face_h // 8)),
        )
        if len(mouths) > 0 and len(noses) > 0:
            mouth = max(mouths, key=lambda m: m[2] * m[3])
            nose = max(noses, key=lambda n: n[2] * n[3])
            eye_mid_y = float((left_center[1] + right_center[1]) / 2.0)
            nose_mid_y = float(nose[1] + nose[3] / 2.0)
            mouth_mid_y = float(mouth[1] + mouth[3] / 2.0)
            up = max(nose_mid_y - eye_mid_y, 1.0)
            down = max(mouth_mid_y - nose_mid_y, 1.0)
            pitch = abs((down / up) - 1.0)
            metrics['pitch'] = round(pitch, 3)
        else:
            pitch = 0.0

        abs_roll = abs(roll)
        if (
            abs_roll >= self.settings.head_roll_fail_degrees
            or yaw >= self.settings.yaw_fail_threshold
            or pitch >= self.settings.pitch_fail_threshold
        ):
            details.append(
                ComplianceDetail(
                    code=ERROR_POSE_INVALID,
                    message='头部倾斜或偏转较明显，请保持正视镜头、头部摆正后重试',
                    status='failed',
                    stage='pose_estimation',
                    score=round(max(abs_roll / 30.0, yaw, pitch), 3),
                    threshold=round(
                        max(
                            self.settings.head_roll_fail_degrees / 30.0,
                            self.settings.yaw_fail_threshold,
                            self.settings.pitch_fail_threshold,
                        ),
                        3,
                    ),
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_POSE_INVALID],
                )
            )
        elif (
            abs_roll >= self.settings.head_roll_warn_degrees
            or yaw >= self.settings.yaw_warn_threshold
            or pitch >= self.settings.pitch_warn_threshold
        ):
            details.append(
                ComplianceDetail(
                    code=ERROR_POSE_INVALID,
                    message='头部姿态不够端正，建议微调到更标准的正脸角度',
                    status='warning',
                    stage='pose_estimation',
                    score=round(max(abs_roll / 30.0, yaw, pitch), 3),
                    threshold=round(
                        max(
                            self.settings.head_roll_warn_degrees / 30.0,
                            self.settings.yaw_warn_threshold,
                            self.settings.pitch_warn_threshold,
                        ),
                        3,
                    ),
                    severity_rank=1,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_POSE_INVALID],
                )
            )
        return details, metrics

    def _detect_hat_or_head_occlusion(self, face_roi: np.ndarray) -> list[ComplianceDetail]:
        details: list[ComplianceDetail] = []
        face_h, _face_w = face_roi.shape[:2]
        top_h = max(1, int(face_h * self.settings.head_top_region_ratio))
        top_region = face_roi[:top_h, :]
        edges = cv2.Canny(top_region, 50, 120)
        edge_ratio = float(np.count_nonzero(edges)) / max(edges.size, 1)
        if edge_ratio > self.settings.headwear_edge_ratio_threshold:
            details.append(
                ComplianceDetail(
                    code=ERROR_HEADWEAR_DETECTED,
                    message='检测到头部区域疑似存在帽檐/硬遮挡，不符合证件照要求',
                    status='failed',
                    stage='compliance_occlusion',
                    score=round(edge_ratio, 3),
                    threshold=self.settings.headwear_edge_ratio_threshold,
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_HEADWEAR_DETECTED],
                )
            )
        return details

    def _detect_expression(self, gray_face: np.ndarray) -> list[ComplianceDetail]:
        details: list[ComplianceDetail] = []
        if self.smile_cascade.empty():
            return details
        face_h, face_w = gray_face.shape[:2]
        smiles = self.smile_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(max(24, face_w // 6), max(16, face_h // 12)),
        )
        if len(smiles) > 0:
            major = max(smiles, key=lambda s: s[2] * s[3])
            smile_ratio = float(major[2]) / max(float(face_w), 1.0)
            if smile_ratio >= self.settings.smile_ratio_warn_threshold:
                details.append(
                    ComplianceDetail(
                        code=ERROR_NOT_SINGLE_FRONTAL_FACE,
                        message='检测到表情幅度偏大，建议保持自然中性表情以提升证件照通过率',
                        status='warning',
                        stage='expression',
                        score=round(smile_ratio, 3),
                        threshold=self.settings.smile_ratio_warn_threshold,
                        severity_rank=1,
                        priority_rank=self.ISSUE_PRIORITY[ERROR_NOT_SINGLE_FRONTAL_FACE],
                    )
                )
        return details

    def _detect_pose_and_single_face(self, gray_image: np.ndarray, face_box: dict, face_count: int) -> list[ComplianceDetail]:
        details: list[ComplianceDetail] = []
        if face_count != 1:
            details.append(
                ComplianceDetail(
                    code=ERROR_NOT_SINGLE_FRONTAL_FACE,
                    message='当前照片不符合证件照规范，请使用单人正脸、无遮挡照片',
                    status='failed',
                    stage='compliance_pose',
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_NOT_SINGLE_FRONTAL_FACE],
                )
            )
            return details

        x, y, w, h = int(face_box['x']), int(face_box['y']), int(face_box['width']), int(face_box['height'])
        pad_w = int(w * 0.5)
        pad_h = int(h * 0.35)
        roi = gray_image[max(0, y - pad_h) : y + h + pad_h, max(0, x - pad_w) : x + w + pad_w]
        profiles = self.profile_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        if len(profiles) > self.settings.max_profile_face_count:
            details.append(
                ComplianceDetail(
                    code=ERROR_NOT_SINGLE_FRONTAL_FACE,
                    message='检测到侧脸特征，建议使用正视镜头的单人证件照原图',
                    status='failed',
                    stage='compliance_pose',
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_NOT_SINGLE_FRONTAL_FACE],
                )
            )
        return details

    def evaluate(self, image_bgr: np.ndarray, gray_image: np.ndarray, face_box: dict | None, face_count: int) -> dict:
        if face_box is None:
            default_detail = ComplianceDetail(
                code=ERROR_NOT_SINGLE_FRONTAL_FACE,
                message='当前照片不符合证件照规范，请使用单人正脸、无遮挡照片',
                status='failed',
                stage='compliance_pose',
                severity_rank=0,
                priority_rank=self.ISSUE_PRIORITY[ERROR_NOT_SINGLE_FRONTAL_FACE],
            )
            return {
                'status': 'failed',
                'code': default_detail.code,
                'message': default_detail.message,
                'details': [self._detail_payload(default_detail)],
                'keypointConfidences': {},
                'warnings': [],
                'metrics': {},
            }

        details: list[ComplianceDetail] = []
        warnings: list[str] = []
        face_roi = self._clamp_roi(image_bgr, face_box)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        keypoints, keypoint_issues, keypoint_warnings = self._detect_keypoints(gray_face)
        details.extend(keypoint_issues)
        warnings.extend(keypoint_warnings)
        details.extend(self._detect_pose_and_single_face(gray_image, face_box, face_count))
        details.extend(self._detect_hat_or_head_occlusion(face_roi))
        details.extend(self._detect_expression(gray_face))
        eye_details, eye_metrics = self._analyze_eye_state(gray_face, keypoints)
        pose_details, pose_metrics = self._analyze_pose(gray_face)
        details.extend(eye_details)
        details.extend(pose_details)

        has_keypoint_issue = any(item.code == ERROR_FACIAL_KEYPOINTS_INCOMPLETE and item.status == 'failed' for item in details)
        if has_keypoint_issue:
            details.append(
                ComplianceDetail(
                    code=ERROR_FACE_OCCLUDED,
                    message='人脸关键区域存在遮挡或检测不完整，请露出双眼和完整面部后重试',
                    status='failed',
                    stage='compliance_occlusion',
                    severity_rank=0,
                    priority_rank=self.ISSUE_PRIORITY[ERROR_FACE_OCCLUDED],
                )
            )

        sorted_details = self._sort_details(details)
        failed_details = [item for item in sorted_details if item.status == 'failed']
        warning_details = [item for item in sorted_details if item.status == 'warning']
        if failed_details:
            primary = failed_details[0]
            return {
                'status': 'failed',
                'code': primary.code,
                'message': primary.message,
                'details': [self._detail_payload(item) for item in failed_details + warning_details],
                'keypointConfidences': keypoints,
                'warnings': warnings,
                'metrics': {'eyes': eye_metrics, 'pose': pose_metrics},
            }
        if warning_details or warnings:
            primary = warning_details[0] if warning_details else ComplianceDetail(
                code='COMPLIANCE_WARNING',
                message='合规审核通过（存在提醒项）',
                status='warning',
                stage='compliance',
                severity_rank=1,
                priority_rank=99,
            )
            return {
                'status': 'warning',
                'code': primary.code,
                'message': primary.message,
                'details': [self._detail_payload(item) for item in warning_details]
                + [
                    {'code': 'COMPLIANCE_WARNING', 'message': warning, 'status': 'warning', 'stage': 'keypoint_detection'}
                    for warning in warnings
                ],
                'keypointConfidences': keypoints,
                'warnings': warnings,
                'metrics': {'eyes': eye_metrics, 'pose': pose_metrics},
            }
        return {
            'status': 'passed',
            'code': 'COMPLIANCE_PASSED',
            'message': '合规性审核通过',
            'details': [],
            'keypointConfidences': keypoints,
            'warnings': [],
            'metrics': {'eyes': eye_metrics, 'pose': pose_metrics},
        }
