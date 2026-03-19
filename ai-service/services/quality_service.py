from __future__ import annotations

from typing import Tuple

from PIL import Image

from constants.status import QUALITY_STATUS_FAILED, QUALITY_STATUS_PASSED, QUALITY_STATUS_WARNING
from utils.config import get_settings


class QualityService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(self, image: Image.Image) -> Tuple[str, str]:
        width, height = image.size
        if width < self.settings.min_image_width or height < self.settings.min_image_height:
            return QUALITY_STATUS_FAILED, '输出尺寸过小'
        if width < 480 or height < 640:
            return QUALITY_STATUS_WARNING, '清晰度一般，建议使用更高分辨率原图'
        return QUALITY_STATUS_PASSED, '质量通过'

    def is_image_too_blurry(self, blur_score: float) -> bool:
        return blur_score < self.settings.blur_score_threshold

    def is_face_too_small(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        image_height, image_width = image_shape[:2]
        face_area = max(face_box['width'], 0) * max(face_box['height'], 0)
        image_area = max(image_height * image_width, 1)
        face_height_ratio = face_box['height'] / max(image_height, 1)
        return (
            face_box['width'] < self.settings.min_face_width
            or face_box['height'] < self.settings.min_face_height
            or face_area / image_area < self.settings.min_face_area_ratio
            or face_height_ratio < self.settings.min_face_height_ratio
        )

    def is_pose_invalid(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        image_height, image_width = image_shape[:2]
        face_center_x = face_box['x'] + face_box['width'] / 2
        image_center_x = image_width / 2
        center_offset_ratio = abs(face_center_x - image_center_x) / max(image_width, 1)
        aspect_ratio = face_box['width'] / max(face_box['height'], 1)
        return (
            center_offset_ratio > self.settings.max_face_center_offset_ratio
            or aspect_ratio < self.settings.min_face_aspect_ratio
            or aspect_ratio > self.settings.max_face_aspect_ratio
        )

    def is_face_occluded(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        _image_height, _image_width = image_shape[:2]
        aspect_ratio = face_box['width'] / max(face_box['height'], 1)
        return aspect_ratio < self.settings.occluded_face_aspect_ratio

    def is_head_cropped(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        image_height, image_width = image_shape[:2]
        margin_x = image_width * self.settings.edge_touch_ratio
        margin_y = image_height * self.settings.edge_touch_ratio
        left = face_box['x']
        top = face_box['y']
        right = face_box['x'] + face_box['width']
        bottom = face_box['y'] + face_box['height']
        return left <= margin_x or top <= margin_y or right >= image_width - margin_x or bottom >= image_height - margin_y
