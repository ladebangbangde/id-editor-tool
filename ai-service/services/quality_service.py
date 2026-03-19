from typing import Tuple

from PIL import Image

from constants.status import QUALITY_STATUS_FAILED, QUALITY_STATUS_PASSED, QUALITY_STATUS_WARNING


class QualityService:
    min_blur_score = 0.35
    min_face_height_ratio = 0.18
    min_face_area_ratio = 0.025
    max_face_center_offset_ratio = 0.18
    min_face_aspect_ratio = 0.72
    max_face_aspect_ratio = 1.38
    occluded_face_aspect_ratio = 0.62
    edge_touch_ratio = 0.02

    def evaluate(self, image: Image.Image) -> Tuple[str, str]:
        width, height = image.size
        if width < 200 or height < 200:
            return QUALITY_STATUS_FAILED, "输出尺寸过小"
        if width < 350 or height < 450:
            return QUALITY_STATUS_WARNING, "输出清晰度一般，建议使用更高分辨率原图"
        return QUALITY_STATUS_PASSED, "质量检测通过"

    def is_image_too_blurry(self, blur_score: float) -> bool:
        return blur_score < self.min_blur_score

    def is_face_too_small(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        image_height, image_width = image_shape[:2]
        face_width = face_box['width']
        face_height = face_box['height']
        face_area_ratio = (face_width * face_height) / max(image_width * image_height, 1)
        face_height_ratio = face_height / max(image_height, 1)
        return face_area_ratio < self.min_face_area_ratio or face_height_ratio < self.min_face_height_ratio

    def is_pose_invalid(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        image_width = image_shape[1]
        face_width = face_box['width']
        face_height = max(face_box['height'], 1)
        face_center_x = face_box['x'] + face_width / 2
        center_offset_ratio = abs(face_center_x - image_width / 2) / max(image_width, 1)
        aspect_ratio = face_width / face_height
        return (
            center_offset_ratio > self.max_face_center_offset_ratio
            or aspect_ratio < self.min_face_aspect_ratio
            or aspect_ratio > self.max_face_aspect_ratio
        )

    def is_face_occluded(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        image_width = image_shape[1]
        face_width = face_box['width']
        face_height = max(face_box['height'], 1)
        face_center_x = face_box['x'] + face_width / 2
        center_offset_ratio = abs(face_center_x - image_width / 2) / max(image_width, 1)
        aspect_ratio = face_width / face_height
        return center_offset_ratio <= self.max_face_center_offset_ratio and (
            aspect_ratio < self.occluded_face_aspect_ratio
            or aspect_ratio > (1 / self.occluded_face_aspect_ratio)
        )

    def is_head_cropped(self, image_shape: tuple[int, ...], face_box: dict) -> bool:
        image_height, image_width = image_shape[:2]
        left = face_box['x'] / max(image_width, 1)
        top = face_box['y'] / max(image_height, 1)
        right = (face_box['x'] + face_box['width']) / max(image_width, 1)
        bottom = (face_box['y'] + face_box['height']) / max(image_height, 1)
        return (
            left <= self.edge_touch_ratio
            or top <= self.edge_touch_ratio
            or right >= 1 - self.edge_touch_ratio
            or bottom >= 1 - self.edge_touch_ratio
        )
