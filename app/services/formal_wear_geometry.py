from __future__ import annotations

from dataclasses import dataclass, field

from app.services.face_detection import FaceDetectionResult


@dataclass
class FormalWearAnchors:
    image_width: int
    image_height: int
    face_box: dict[str, int]
    head_center_x: float
    head_center_y: float
    chin_y: float
    neck_center_x: float
    neck_top_y: float
    neck_bottom_y: float
    neck_width: float
    shoulder_y: float
    left_shoulder_x: float
    right_shoulder_x: float
    chest_top_y: float
    chest_bottom_y: float
    jacket_top_y: float
    lapel_inner_gap: float
    lapel_outer_span: float
    tie_top_y: float
    tie_bottom_y: float


@dataclass
class ShoulderNeckAssessment:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


class FormalWearGeometry:
    def estimate_anchors(
        self,
        image_size: tuple[int, int],
        face_box: dict[str, int],
        gender: str,
        style: str,
    ) -> FormalWearAnchors:
        width, height = image_size
        x = float(face_box['x'])
        y = float(face_box['y'])
        face_w = float(face_box['width'])
        face_h = float(face_box['height'])

        style_span = {'simple': 1.55, 'standard': 1.72, 'business': 1.82}[style]
        gender_span_boost = 0.0 if gender == 'male' else 0.08
        shoulder_span = face_w * (style_span + gender_span_boost)
        shoulder_half = shoulder_span / 2.0

        chin_y = y + face_h * 0.94
        neck_top_y = y + face_h * 0.80
        neck_bottom_y = chin_y + face_h * (0.17 if gender == 'male' else 0.15)
        shoulder_y = chin_y + face_h * (0.24 if gender == 'male' else 0.20)
        chest_top_y = shoulder_y - face_h * 0.03
        chest_bottom_y = min(height - 1.0, shoulder_y + face_h * 0.98)
        jacket_top_y = chin_y + face_h * 0.10
        head_center_x = x + face_w / 2.0
        head_center_y = y + face_h * 0.42
        neck_width = face_w * (0.30 if gender == 'male' else 0.28)
        lapel_inner_gap = face_w * (0.18 if style == 'simple' else 0.16)
        lapel_outer_span = face_w * (0.60 if style == 'business' else 0.52)
        tie_top_y = neck_bottom_y - face_h * 0.03
        tie_bottom_y = min(height - 1.0, chest_bottom_y - face_h * 0.16)

        return FormalWearAnchors(
            image_width=width,
            image_height=height,
            face_box=face_box,
            head_center_x=head_center_x,
            head_center_y=head_center_y,
            chin_y=chin_y,
            neck_center_x=head_center_x,
            neck_top_y=neck_top_y,
            neck_bottom_y=neck_bottom_y,
            neck_width=neck_width,
            shoulder_y=shoulder_y,
            left_shoulder_x=max(0.0, head_center_x - shoulder_half),
            right_shoulder_x=min(width - 1.0, head_center_x + shoulder_half),
            chest_top_y=chest_top_y,
            chest_bottom_y=chest_bottom_y,
            jacket_top_y=jacket_top_y,
            lapel_inner_gap=lapel_inner_gap,
            lapel_outer_span=lapel_outer_span,
            tie_top_y=tie_top_y,
            tie_bottom_y=tie_bottom_y,
        )

    def assess_shoulder_neck(
        self,
        image_size: tuple[int, int],
        face_box: dict[str, int],
        detect_result: FaceDetectionResult,
        gender: str,
        style: str,
    ) -> ShoulderNeckAssessment:
        width, height = image_size
        anchors = self.estimate_anchors(image_size, face_box, gender, style)
        x = float(face_box['x'])
        y = float(face_box['y'])
        face_w = float(face_box['width'])
        face_h = float(face_box['height'])
        center_x = x + face_w / 2.0

        bottom_room_ratio = max(height - anchors.chin_y, 0.0) / max(height, 1)
        left_shoulder_room_ratio = max(center_x - anchors.left_shoulder_x, 0.0) / max(width, 1)
        right_shoulder_room_ratio = max(anchors.right_shoulder_x - center_x, 0.0) / max(width, 1)
        face_height_ratio = face_h / max(height, 1)
        face_width_ratio = face_w / max(width, 1)
        top_margin_ratio = y / max(height, 1)
        blur_score = detect_result.blur_score or 0.0

        reasons: list[str] = []
        warnings: list[str] = []

        if face_height_ratio > 0.58:
            reasons.append('头像占比过大，画面下方缺少可绘制服装区域')
        elif face_height_ratio > 0.50:
            warnings.append('头像占比偏大，正装可绘制空间有限')

        if bottom_room_ratio < 0.16:
            reasons.append('下巴以下留白不足，肩颈区域不完整')
        elif bottom_room_ratio < 0.22:
            warnings.append('下巴以下空间偏少，换装效果可能偏紧凑')

        if min(left_shoulder_room_ratio, right_shoulder_room_ratio) < 0.16:
            reasons.append('左右肩部空间不足，无法稳定绘制正装肩线')
        elif min(left_shoulder_room_ratio, right_shoulder_room_ratio) < 0.19:
            warnings.append('肩部空间偏窄，正装肩线容错较低')

        if face_width_ratio > 0.52:
            reasons.append('人脸横向占比过大，缺少自然肩宽空间')
        if top_margin_ratio > 0.34:
            warnings.append('头顶留白较多，正装位置会更依赖启发式估算')
        if blur_score and blur_score < 0.0018:
            reasons.append('图片清晰度不足，无法稳定估算肩颈结构')

        return ShoulderNeckAssessment(
            passed=not reasons,
            reasons=reasons,
            warnings=warnings,
            metrics={
                'bottomRoomRatio': bottom_room_ratio,
                'leftShoulderRoomRatio': left_shoulder_room_ratio,
                'rightShoulderRoomRatio': right_shoulder_room_ratio,
                'faceHeightRatio': face_height_ratio,
                'faceWidthRatio': face_width_ratio,
                'topMarginRatio': top_margin_ratio,
                'blurScore': blur_score,
            },
        )
