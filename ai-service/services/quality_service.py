from __future__ import annotations

from PIL import Image

from constants.status import QUALITY_STATUS_FAILED, QUALITY_STATUS_PASSED, QUALITY_STATUS_WARNING
from utils.config import get_settings


class QualityService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate_details(self, image: Image.Image) -> dict:
        width, height = image.size
        resolution_too_low = width < self.settings.min_image_width or height < self.settings.min_image_height
        clarity_insufficient = width < 480 or height < 640
        suitable = not resolution_too_low

        if resolution_too_low:
            status = QUALITY_STATUS_FAILED
            message = f'分辨率过低，建议至少达到 {self.settings.min_image_width}x{self.settings.min_image_height}'
        elif clarity_insufficient:
            status = QUALITY_STATUS_WARNING
            message = '清晰度一般，建议使用更高分辨率原图'
        else:
            status = QUALITY_STATUS_PASSED
            message = '质量通过'

        return {
            'qualityStatus': status,
            'qualityMessage': message,
            'resolutionTooLow': resolution_too_low,
            'clarityInsufficient': clarity_insufficient,
            'suitableForIdPhoto': suitable,
        }

    def evaluate(self, image: Image.Image) -> tuple[str, str]:
        details = self.evaluate_details(image)
        return details['qualityStatus'], details['qualityMessage']

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
        _image_height, image_width = image_shape[:2]
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
