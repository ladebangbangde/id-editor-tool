from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.exceptions import (
    ERROR_FACE_OCCLUDED,
    ERROR_FACIAL_KEYPOINTS_INCOMPLETE,
    ERROR_HEADWEAR_DETECTED,
    ERROR_NOT_SINGLE_FRONTAL_FACE,
)
from utils.config import get_settings


@dataclass
class ComplianceDetail:
    code: str
    message: str


class ComplianceService:
    """证件照合规性审核：尽量复用现有 OpenCV 人脸检测链路，补充关键点/遮挡/正脸规则。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        cascades = cv2.data.haarcascades
        self.eye_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_eye.xml')
        self.nose_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_mcs_mouth.xml')
        self.smile_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_smile.xml')
        self.profile_cascade = cv2.CascadeClassifier(cascades + 'haarcascade_profileface.xml')

    @staticmethod
    def _clamp_roi(image: np.ndarray, box: dict) -> np.ndarray:
        x = max(int(box['x']), 0)
        y = max(int(box['y']), 0)
        w = max(int(box['width']), 1)
        h = max(int(box['height']), 1)
        return image[y : y + h, x : x + w]

    @staticmethod
    def _safe_confidence(raw_weight: float | None) -> float:
        if raw_weight is None:
            return 0.0
        return max(0.0, min(1.0, float(raw_weight) / 8.0))

    def _detect_keypoints(self, gray_face: np.ndarray) -> tuple[dict[str, float], list[ComplianceDetail]]:
        details: list[ComplianceDetail] = []

        def pick_confidence(cascade: cv2.CascadeClassifier, min_size: tuple[int, int]) -> float:
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

        face_h, face_w = gray_face.shape[:2]
        eye_conf = pick_confidence(self.eye_cascade, (max(12, face_w // 9), max(12, face_h // 10)))
        nose_conf = pick_confidence(self.nose_cascade, (max(16, face_w // 8), max(16, face_h // 8)))
        mouth_conf = pick_confidence(self.mouth_cascade, (max(20, face_w // 7), max(20, face_h // 8)))
        if mouth_conf <= 0:
            mouth_conf = pick_confidence(self.smile_cascade, (max(20, face_w // 7), max(20, face_h // 8)))

        conf_map = {
            'eyes': eye_conf,
            'nose': nose_conf,
            'mouth': mouth_conf,
        }

        missing = [k for k, v in conf_map.items() if v < self.settings.landmark_confidence_threshold]
        if missing:
            details.append(
                ComplianceDetail(
                    code=ERROR_FACIAL_KEYPOINTS_INCOMPLETE,
                    message='人脸关键点不完整，请确保双眼、鼻尖、嘴部清晰可见',
                )
            )
        return conf_map, details

    def _detect_pose_and_single_face(self, gray_image: np.ndarray, face_box: dict, face_count: int) -> list[ComplianceDetail]:
        details: list[ComplianceDetail] = []
        if face_count != 1:
            details.append(
                ComplianceDetail(
                    code=ERROR_NOT_SINGLE_FRONTAL_FACE,
                    message='当前照片不符合证件照规范，请使用正脸、无遮挡照片',
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
                    message='当前照片不符合证件照规范，请使用正脸、无遮挡照片',
                )
            )
        return details

    def _detect_hat_or_head_occlusion(self, face_roi: np.ndarray) -> list[ComplianceDetail]:
        details: list[ComplianceDetail] = []
        face_h, face_w = face_roi.shape[:2]
        top_h = max(1, int(face_h * self.settings.head_top_region_ratio))
        top_region = face_roi[:top_h, :]
        edges = cv2.Canny(top_region, 50, 120)
        edge_ratio = float(np.count_nonzero(edges)) / max(edges.size, 1)

        # 帽檐/头部遮挡会在头顶部形成明显硬边和不自然高对比。
        if edge_ratio > self.settings.headwear_edge_ratio_threshold:
            details.append(
                ComplianceDetail(
                    code=ERROR_HEADWEAR_DETECTED,
                    message='检测到帽子或头部遮挡，不符合证件照要求',
                )
            )
        return details

    def evaluate(self, image_bgr: np.ndarray, gray_image: np.ndarray, face_box: dict | None, face_count: int) -> dict:
        if face_box is None:
            return {
                'status': 'failed',
                'code': ERROR_NOT_SINGLE_FRONTAL_FACE,
                'message': '当前照片不符合证件照规范，请使用正脸、无遮挡照片',
                'details': [
                    {
                        'code': ERROR_NOT_SINGLE_FRONTAL_FACE,
                        'message': '当前照片不符合证件照规范，请使用正脸、无遮挡照片',
                    }
                ],
                'keypointConfidences': {},
            }

        details: list[ComplianceDetail] = []
        face_roi = self._clamp_roi(image_bgr, face_box)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        keypoints, keypoint_issues = self._detect_keypoints(gray_face)
        details.extend(keypoint_issues)
        details.extend(self._detect_pose_and_single_face(gray_image, face_box, face_count))
        details.extend(self._detect_hat_or_head_occlusion(face_roi))

        has_keypoint_issue = any(item.code == ERROR_FACIAL_KEYPOINTS_INCOMPLETE for item in details)
        if has_keypoint_issue:
            details.append(
                ComplianceDetail(
                    code=ERROR_FACE_OCCLUDED,
                    message='人脸存在明显遮挡，请露出双眼和完整面部后重试',
                )
            )

        if details:
            primary = details[0]
            return {
                'status': 'failed',
                'code': primary.code,
                'message': primary.message,
                'details': [{'code': item.code, 'message': item.message} for item in details],
                'keypointConfidences': keypoints,
            }

        return {
            'status': 'passed',
            'code': 'COMPLIANCE_PASSED',
            'message': '合规性审核通过',
            'details': [],
            'keypointConfidences': keypoints,
        }
