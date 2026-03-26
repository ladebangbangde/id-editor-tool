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
        'EXAGGERATED_EXPRESSION': '表情不规范',
        'TONGUE_OUT': '检测到吐舌',
        'MOUTH_OPEN': '检测到明显张嘴',
        'SMILE_TOO_BROAD': '笑容幅度过大',
        'MOUTH_ASYMMETRY': '嘴部明显歪斜',
        'WINK_EXPRESSION': '检测到明显挤眼/搞怪表情',
    }

    ISSUE_PRIORITY = {
        'NO_FACE_DETECTED': 95,
        'MULTIPLE_FACES_DETECTED': 94,
        'HAND_OCCLUSION': 93,
        'EYE_OCCLUDED': 92,
        'TONGUE_OUT': 91,
        'MOUTH_OPEN': 90,
        'SMILE_TOO_BROAD': 89,
        'MOUTH_ASYMMETRY': 88,
        'WINK_EXPRESSION': 92,
        'EXAGGERATED_EXPRESSION': 91,
        'SEVERE_POSE': 80,
        'HEAD_SHOULDER_INCOMPLETE': 78,
        'FACE_RATIO_INVALID': 76,
        'JEWELRY_DETECTED': 74,
        'EXTREME_LIGHTING': 70,
        'NOT_SUITABLE_PORTRAIT': 60,
        'IMAGE_TOO_BLURRY': 55,
        'RESOLUTION_TOO_LOW': 54,
    }

    # 证件照审核阈值（集中管理，便于调参）
    EYE_ASPECT_FAIL_THRESHOLD = 0.17  # 明显闭眼：建议范围 0.15~0.20
    EYE_ASPECT_WARN_THRESHOLD = 0.23  # 轻微眯眼：建议范围 0.21~0.27
    EYE_ASYMMETRY_FAIL_THRESHOLD = 0.12  # 单眼闭合风险：建议范围 0.10~0.16
    EYE_ASYMMETRY_WARN_THRESHOLD = 0.08  # 开眼不对称告警：建议范围 0.06~0.12
    HAND_FACE_OVERLAP_FAIL_THRESHOLD = 0.015  # 手部遮挡面部面积占比失败阈值
    HAND_FACE_OVERLAP_WARN_THRESHOLD = 0.006  # 手部遮挡面部面积占比告警阈值
    MOUTH_OPEN_RATIO_FAIL_THRESHOLD = 0.34  # 明显张嘴（发音/大笑）失败阈值
    MOUTH_OPEN_RATIO_WARN_THRESHOLD = 0.24  # 张口偏大提醒阈值（不直接判鬼脸）
    TONGUE_PIXEL_RATIO_FAIL_THRESHOLD = 0.115  # 口腔区域舌头特征显著占比失败阈值
    TONGUE_PIXEL_RATIO_WARN_THRESHOLD = 0.070  # 口腔区域疑似舌头占比提醒阈值
    MOUTH_CORNER_RISE_WARN_THRESHOLD = 0.045  # 自然微笑以上时提醒
    MOUTH_CORNER_RISE_FAIL_THRESHOLD = 0.095  # 夸张咧嘴/上扬失败阈值
    MOUTH_ASYMMETRY_WARN_THRESHOLD = 0.080  # 轻微歪嘴提醒阈值
    MOUTH_ASYMMETRY_FAIL_THRESHOLD = 0.145  # 明显歪嘴失败阈值
    TONGUE_OUT_SCORE_FAIL_THRESHOLD = 0.70
    TONGUE_OUT_SCORE_WARN_THRESHOLD = 0.48
    MOUTH_DARK_RATIO_FAIL_THRESHOLD = 0.18
    MOUTH_DARK_RATIO_WARN_THRESHOLD = 0.10
    EXPRESSION_NOT_NEUTRAL_FAIL_THRESHOLD = 0.70
    EXPRESSION_NOT_NEUTRAL_WARN_THRESHOLD = 0.45
    WINK_INTENSITY_FAIL_THRESHOLD = 0.72
    WINK_INTENSITY_WARN_THRESHOLD = 0.48

    QUALITY_MESSAGES = {
        PASS: '照片质量良好，可直接进入处理流程',
        WARNING: '照片可处理，但存在风险项，建议按提示优化后重拍',
        FAIL: '照片暂不适合处理，请根据主问题调整后重试',
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.metrics_service = PhotoMetricsService()
        self._detector = None
        self._face_mesh = None
        self._hands = None

    def _face_detector(self):
        if self._detector is not None:
            return self._detector

        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
        return self._detector

    def _face_mesh_detector(self):
        if self._face_mesh is not None:
            return self._face_mesh
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
        return self._face_mesh

    def _hands_detector(self):
        if self._hands is not None:
            return self._hands
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
        return self._hands

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

    @staticmethod
    def _landmark_point(face_landmarks, index: int, width: int, height: int) -> tuple[float, float]:
        lm = face_landmarks.landmark[index]
        return lm.x * width, lm.y * height

    def _eye_aspect_ratio_from_mesh(
        self,
        face_landmarks,
        width: int,
        height: int,
        eye_type: str,
    ) -> float:
        # 使用 FaceMesh 关键点估计眼裂开合，低于阈值时保守阻断。
        if eye_type == 'left':
            horizontal = (33, 133)
            vertical_a = (159, 145)
            vertical_b = (160, 144)
        else:
            horizontal = (362, 263)
            vertical_a = (386, 374)
            vertical_b = (385, 380)

        h0 = self._landmark_point(face_landmarks, horizontal[0], width, height)
        h1 = self._landmark_point(face_landmarks, horizontal[1], width, height)
        va0 = self._landmark_point(face_landmarks, vertical_a[0], width, height)
        va1 = self._landmark_point(face_landmarks, vertical_a[1], width, height)
        vb0 = self._landmark_point(face_landmarks, vertical_b[0], width, height)
        vb1 = self._landmark_point(face_landmarks, vertical_b[1], width, height)

        horizontal_dist = float(np.linalg.norm(np.array(h0) - np.array(h1)))
        vertical_dist = (
            float(np.linalg.norm(np.array(va0) - np.array(va1)))
            + float(np.linalg.norm(np.array(vb0) - np.array(vb1)))
        ) / 2.0
        return vertical_dist / max(horizontal_dist, 1.0)

    def _detect_eye_state_via_mesh(self, image: Image.Image, face_box: FaceBox) -> tuple[str, dict[str, float]]:
        rgb = np.asarray(image.convert('RGB'))
        h, w = rgb.shape[:2]
        detector = self._face_mesh_detector()
        result = detector.process(rgb)
        if not result.multi_face_landmarks:
            return 'unknown', {}

        face_landmarks = result.multi_face_landmarks[0]
        left_ear = self._eye_aspect_ratio_from_mesh(face_landmarks, w, h, 'left')
        right_ear = self._eye_aspect_ratio_from_mesh(face_landmarks, w, h, 'right')
        min_ear = min(left_ear, right_ear)
        asym = abs(left_ear - right_ear)
        metrics = {
            'left_eye_ear': float(left_ear),
            'right_eye_ear': float(right_ear),
            'eye_asymmetry': float(asym),
        }

        if min_ear <= self.EYE_ASPECT_FAIL_THRESHOLD:
            if asym >= self.EYE_ASYMMETRY_FAIL_THRESHOLD:
                return 'single_eye_closed', metrics
            return 'both_eyes_closed', metrics
        if min_ear <= self.EYE_ASPECT_WARN_THRESHOLD or asym >= self.EYE_ASYMMETRY_WARN_THRESHOLD:
            return 'eye_risk', metrics
        return 'open', metrics

    def _detect_hand_on_face(self, image: Image.Image, face_box: FaceBox) -> tuple[str, float]:
        rgb = np.asarray(image.convert('RGB'))
        h, w = rgb.shape[:2]
        detector = self._hands_detector()
        result = detector.process(rgb)
        if not result.multi_hand_landmarks:
            return 'clear', 0.0

        face_left = face_box.x
        face_top = face_box.y
        face_right = face_box.x + face_box.width
        face_bottom = face_box.y + face_box.height
        face_area = max(face_box.width * face_box.height, 1)

        overlap_pixels = 0.0
        for hand in result.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand.landmark]
            ys = [lm.y * h for lm in hand.landmark]
            hand_left = max(min(xs), 0.0)
            hand_right = min(max(xs), float(w))
            hand_top = max(min(ys), 0.0)
            hand_bottom = min(max(ys), float(h))
            inter_w = max(0.0, min(face_right, hand_right) - max(face_left, hand_left))
            inter_h = max(0.0, min(face_bottom, hand_bottom) - max(face_top, hand_top))
            overlap_pixels += inter_w * inter_h

        overlap_ratio = float(overlap_pixels) / float(face_area)
        if overlap_ratio >= self.HAND_FACE_OVERLAP_FAIL_THRESHOLD:
            return 'fail', overlap_ratio
        if overlap_ratio >= self.HAND_FACE_OVERLAP_WARN_THRESHOLD:
            return 'warn', overlap_ratio
        return 'clear', overlap_ratio

    def _detect_expression_via_mesh(self, image: Image.Image, face_box: FaceBox) -> tuple[str, dict[str, float]]:
        rgb = np.asarray(image.convert('RGB'))
        h, w = rgb.shape[:2]
        detector = self._face_mesh_detector()
        result = detector.process(rgb)
        if not result.multi_face_landmarks:
            return 'unknown', {}

        face_landmarks = result.multi_face_landmarks[0]
        upper_lip = self._landmark_point(face_landmarks, 13, w, h)
        lower_lip = self._landmark_point(face_landmarks, 14, w, h)
        mouth_left = self._landmark_point(face_landmarks, 61, w, h)
        mouth_right = self._landmark_point(face_landmarks, 291, w, h)
        upper_outer = self._landmark_point(face_landmarks, 0, w, h)

        mouth_width = float(np.linalg.norm(np.array(mouth_left) - np.array(mouth_right)))
        mouth_open = float(np.linalg.norm(np.array(upper_lip) - np.array(lower_lip)))
        mouth_open_ratio = mouth_open / max(mouth_width, 1.0)
        mouth_vertical_gap = mouth_open / max(face_box.height, 1.0)
        corner_center_y = (mouth_left[1] + mouth_right[1]) / 2.0
        mouth_corner_rise = (upper_outer[1] - corner_center_y) / max(mouth_width, 1.0)
        mouth_asymmetry = abs(mouth_left[1] - mouth_right[1]) / max(mouth_width, 1.0)

        center_x = int((mouth_left[0] + mouth_right[0]) / 2.0)
        center_y = int((upper_lip[1] + lower_lip[1]) / 2.0)
        roi_half_w = int(max(4, mouth_width * 0.26))
        roi_h = int(max(6, mouth_width * 0.30))
        x0 = max(0, center_x - roi_half_w)
        x1 = min(w, center_x + roi_half_w)
        y0 = max(0, int(min(upper_lip[1], lower_lip[1]) - max(2.0, mouth_open * 0.25)))
        y1 = min(h, int(max(upper_lip[1], lower_lip[1]) + max(4.0, mouth_open * 0.95)))

        tongue_ratio = 0.0
        inner_mouth_area_ratio = 0.0
        mouth_dark_ratio = 0.0
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            mouth_roi = rgb[y0:y1, x0:x1]
            inner_lip_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            inner_points = []
            for idx in inner_lip_indices:
                px, py = self._landmark_point(face_landmarks, idx, w, h)
                inner_points.append((int(round(px - x0)), int(round(py - y0))))

            mouth_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            if len(inner_points) >= 3:
                from PIL import ImageDraw

                mask_img = Image.new('L', (x1 - x0, y1 - y0), 0)
                ImageDraw.Draw(mask_img).polygon(inner_points, outline=1, fill=1)
                mouth_mask = np.array(mask_img, dtype=np.uint8)
                contour = np.array(inner_points, dtype=np.float32)
                x_coords = contour[:, 0]
                y_coords = contour[:, 1]
                contour_area = 0.5 * abs(np.dot(x_coords, np.roll(y_coords, -1)) - np.dot(y_coords, np.roll(x_coords, -1)))
                inner_mouth_area_ratio = contour_area / max((mouth_width * mouth_width), 1.0)

            r = mouth_roi[:, :, 0].astype(np.float32)
            g = mouth_roi[:, :, 1].astype(np.float32)
            b = mouth_roi[:, :, 2].astype(np.float32)
            red_dominant = (r > 95.0) & (r - np.maximum(g, b) > 12.0)
            warm_balance = (g > 45.0) & (b > 35.0)
            inside_mouth = mouth_mask > 0
            tongue_mask = red_dominant & warm_balance & inside_mouth
            tongue_ratio = float(np.count_nonzero(tongue_mask) / max(np.count_nonzero(inside_mouth), 1))
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            dark_mask = (luma < 70.0) & inside_mouth
            mouth_dark_ratio = float(np.count_nonzero(dark_mask) / max(np.count_nonzero(inside_mouth), 1))

        tongue_out_score = min(
            1.0,
            0.60 * (tongue_ratio / max(self.TONGUE_PIXEL_RATIO_FAIL_THRESHOLD, 1e-6))
            + 0.30 * (mouth_open_ratio / max(self.MOUTH_OPEN_RATIO_WARN_THRESHOLD, 1e-6))
            + 0.10 * (mouth_vertical_gap / 0.08),
        )
        expression_not_neutral_score = min(
            1.0,
            0.45 * (mouth_open_ratio / max(self.MOUTH_OPEN_RATIO_WARN_THRESHOLD, 1e-6))
            + 0.30 * (max(mouth_corner_rise, 0.0) / max(self.MOUTH_CORNER_RISE_FAIL_THRESHOLD, 1e-6))
            + 0.25 * (mouth_asymmetry / max(self.MOUTH_ASYMMETRY_FAIL_THRESHOLD, 1e-6)),
        )

        metrics = {
            'mouth_aspect_ratio': mouth_open_ratio,
            'mouth_open_ratio': mouth_open_ratio,
            'mouth_vertical_gap': mouth_vertical_gap,
            'mouth_width': mouth_width,
            'tongue_pixel_ratio': tongue_ratio,
            'inner_mouth_area_ratio': float(inner_mouth_area_ratio),
            'mouth_dark_ratio': float(mouth_dark_ratio),
            'mouth_corner_rise': float(mouth_corner_rise),
            'mouth_asymmetry': float(mouth_asymmetry),
            'tongue_out_score': float(tongue_out_score),
            'expression_not_neutral_score': float(expression_not_neutral_score),
        }

        if (
            tongue_ratio >= self.TONGUE_PIXEL_RATIO_FAIL_THRESHOLD
            and mouth_open_ratio >= 0.14
            and inner_mouth_area_ratio >= 0.035
            and mouth_dark_ratio >= self.MOUTH_DARK_RATIO_FAIL_THRESHOLD
            and tongue_out_score >= self.TONGUE_OUT_SCORE_FAIL_THRESHOLD
        ):
            return 'tongue_out_fail', metrics
        if mouth_open_ratio >= self.MOUTH_OPEN_RATIO_FAIL_THRESHOLD:
            return 'mouth_open_fail', metrics
        if (
            mouth_corner_rise >= self.MOUTH_CORNER_RISE_FAIL_THRESHOLD
            and mouth_open_ratio >= self.MOUTH_OPEN_RATIO_WARN_THRESHOLD
        ) or mouth_asymmetry >= self.MOUTH_ASYMMETRY_FAIL_THRESHOLD:
            return 'exaggerated_fail', metrics

        if (
            tongue_ratio >= self.TONGUE_PIXEL_RATIO_WARN_THRESHOLD
            and mouth_open_ratio >= 0.12
            and inner_mouth_area_ratio >= 0.020
            and mouth_dark_ratio >= self.MOUTH_DARK_RATIO_WARN_THRESHOLD
            and tongue_out_score >= self.TONGUE_OUT_SCORE_WARN_THRESHOLD
        ):
            return 'tongue_out_warn', metrics
        if mouth_open_ratio >= self.MOUTH_OPEN_RATIO_WARN_THRESHOLD:
            return 'mouth_open_warn', metrics
        if mouth_asymmetry >= self.MOUTH_ASYMMETRY_WARN_THRESHOLD:
            return 'expression_warn', metrics
        if (
            mouth_corner_rise >= self.MOUTH_CORNER_RISE_WARN_THRESHOLD
            and mouth_open_ratio >= 0.15
            and expression_not_neutral_score >= self.EXPRESSION_NOT_NEUTRAL_WARN_THRESHOLD
        ):
            return 'expression_warn', metrics
        return 'neutral', metrics

    def _detect_wink_expression(
        self,
        image: Image.Image,
        face_box: FaceBox,
        eye_metrics: dict[str, float],
        expression_metrics: dict[str, float],
    ) -> tuple[str, dict[str, float]]:
        rgb = np.asarray(image.convert('RGB'))
        h, w = rgb.shape[:2]
        try:
            detector = self._face_mesh_detector()
            result = detector.process(rgb)
        except Exception as exc:
            logger.warning('wink expression mesh unavailable: %s', exc)
            return 'unknown', {}
        if not result.multi_face_landmarks:
            return 'unknown', {}

        face_landmarks = result.multi_face_landmarks[0]
        left_eye = eye_metrics.get('left_eye_ear', 0.0)
        right_eye = eye_metrics.get('right_eye_ear', 0.0)
        eye_asym = eye_metrics.get('eye_asymmetry', abs(left_eye - right_eye))
        min_eye = min(left_eye, right_eye)
        max_eye = max(left_eye, right_eye)
        one_eye_closed_strength = np.clip((self.EYE_ASPECT_WARN_THRESHOLD - min_eye) / max(self.EYE_ASPECT_WARN_THRESHOLD, 1e-6), 0.0, 1.0)
        asym_strength = np.clip(eye_asym / max(self.EYE_ASYMMETRY_FAIL_THRESHOLD, 1e-6), 0.0, 1.0)

        nose_left = self._landmark_point(face_landmarks, 48, w, h)
        nose_right = self._landmark_point(face_landmarks, 278, w, h)
        nose_bridge = self._landmark_point(face_landmarks, 6, w, h)
        nose_tip = self._landmark_point(face_landmarks, 1, w, h)
        left_cheek = self._landmark_point(face_landmarks, 187, w, h)
        right_cheek = self._landmark_point(face_landmarks, 411, w, h)
        mouth_left = self._landmark_point(face_landmarks, 61, w, h)
        mouth_right = self._landmark_point(face_landmarks, 291, w, h)

        nose_width = float(np.linalg.norm(np.array(nose_left) - np.array(nose_right)))
        bridge_to_tip = float(np.linalg.norm(np.array(nose_bridge) - np.array(nose_tip)))
        nose_squeeze = np.clip((bridge_to_tip / max(nose_width, 1.0) - 0.45) / 0.35, 0.0, 1.0)

        cheek_gap = float(np.linalg.norm(np.array(left_cheek) - np.array(right_cheek)))
        mouth_width = float(np.linalg.norm(np.array(mouth_left) - np.array(mouth_right)))
        cheek_compress = np.clip((mouth_width / max(cheek_gap, 1.0) - 0.28) / 0.18, 0.0, 1.0)

        mouth_asym = abs(expression_metrics.get('mouth_asymmetry', 0.0))
        mouth_asym_strength = np.clip(mouth_asym / max(self.MOUTH_ASYMMETRY_WARN_THRESHOLD, 1e-6), 0.0, 1.0)

        wink_score = float(
            min(
                1.0,
                0.40 * one_eye_closed_strength
                + 0.28 * asym_strength
                + 0.18 * nose_squeeze
                + 0.09 * cheek_compress
                + 0.05 * mouth_asym_strength,
            )
        )
        metrics = {
            'wink_score': wink_score,
            'wink_eye_strength': float(one_eye_closed_strength),
            'wink_asym_strength': float(asym_strength),
            'wink_nose_squeeze': float(nose_squeeze),
            'wink_cheek_compress': float(cheek_compress),
        }
        if (
            wink_score >= self.WINK_INTENSITY_FAIL_THRESHOLD
            and min_eye <= self.EYE_ASPECT_WARN_THRESHOLD
            and eye_asym >= self.EYE_ASYMMETRY_WARN_THRESHOLD
            and max_eye >= self.EYE_ASPECT_WARN_THRESHOLD
        ):
            return 'wink_fail', metrics
        if wink_score >= self.WINK_INTENSITY_WARN_THRESHOLD:
            return 'wink_warn', metrics
        return 'neutral', metrics

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
            eye_state, eye_metrics = self._detect_eye_state_via_mesh(image, face_box)
            metrics.update(eye_metrics)
            if eye_state == 'single_eye_closed':
                self._append_issue(issues, 'EYE_OCCLUDED', '检测到单眼闭合，当前照片不适合作为证件照提交', FAIL)
            elif eye_state == 'both_eyes_closed':
                self._append_issue(issues, 'EYE_OCCLUDED', '检测到双眼闭合，当前照片不适合作为证件照提交', FAIL)
            elif eye_state == 'eye_risk':
                self._append_issue(issues, 'EYE_OCCLUDED', '双眼开合异常或不对称，存在审核风险', WARNING)
            elif eye_state == 'unknown':
                if visible_eyes <= 1:
                    self._append_issue(issues, 'EYE_OCCLUDED', '眼部状态不稳定，建议重拍为双眼自然睁开', FAIL)
                else:
                    self._append_issue(issues, 'EYE_OCCLUDED', '眼部状态识别不稳定，存在审核风险', WARNING)

            hand_state, hand_overlap = self._detect_hand_on_face(image, face_box)
            metrics['hand_face_overlap'] = float(hand_overlap)
            if hand_state == 'fail':
                self._append_issue(issues, 'HAND_OCCLUSION', '检测到手势遮挡面部关键区域，不建议用于正式证件照', FAIL)
            elif hand_state == 'warn':
                self._append_issue(issues, 'HAND_OCCLUSION', '检测到手部接近面部，可能影响证件照审核', WARNING)

            expression_state, expression_metrics = self._detect_expression_via_mesh(image, face_box)
            metrics.update(expression_metrics)
            if expression_state == 'tongue_out_fail':
                self._append_issue(issues, 'TONGUE_OUT', '检测到明显吐舌，不符合证件照规范，请自然闭口重拍', FAIL)
            elif expression_state == 'mouth_open_fail':
                self._append_issue(issues, 'MOUTH_OPEN', '检测到明显张嘴或露齿，不符合证件照规范，请闭口重拍', FAIL)
            elif expression_state == 'exaggerated_fail':
                if expression_metrics.get('mouth_asymmetry', 0.0) >= self.MOUTH_ASYMMETRY_FAIL_THRESHOLD:
                    self._append_issue(issues, 'MOUTH_ASYMMETRY', '检测到嘴部明显歪斜，建议保持面部自然对称后重拍', FAIL)
                else:
                    self._append_issue(issues, 'SMILE_TOO_BROAD', '检测到明显夸张表情，不符合证件照规范，请保持自然中性表情重拍', FAIL)
            elif expression_state == 'tongue_out_warn':
                self._append_issue(issues, 'TONGUE_OUT', '检测到疑似吐舌趋势，建议闭口自然表情后重拍', WARNING)
            elif expression_state == 'mouth_open_warn':
                self._append_issue(issues, 'MOUTH_OPEN', '嘴部开合偏大，建议闭口拍摄以降低审核风险', WARNING)
            elif expression_state == 'expression_warn':
                if expression_metrics.get('mouth_asymmetry', 0.0) >= self.MOUTH_ASYMMETRY_WARN_THRESHOLD:
                    self._append_issue(issues, 'MOUTH_ASYMMETRY', '嘴部轻微不对称，建议放松后保持正面重拍', WARNING)
                else:
                    self._append_issue(issues, 'SMILE_TOO_BROAD', '笑意稍明显，建议表情更中性以提升证件照通过率', WARNING)

            wink_state, wink_metrics = self._detect_wink_expression(image, face_box, eye_metrics=eye_metrics, expression_metrics=expression_metrics)
            metrics.update(wink_metrics)
            if wink_state == 'wink_fail':
                if self.settings.enable_wink_hard_fail:
                    self._append_issue(issues, 'WINK_EXPRESSION', '检测到明显单眼搞怪/挤眉弄眼表情，不符合证件照规范', FAIL)
                else:
                    self._append_issue(issues, 'WINK_EXPRESSION', '检测到明显单眼搞怪/挤眉弄眼风险，建议自然表情重拍', WARNING)
            elif wink_state == 'wink_warn':
                self._append_issue(issues, 'WINK_EXPRESSION', '检测到疑似单眼挤压表情，建议双眼自然睁开重拍', WARNING)

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
