from __future__ import annotations

from PIL import Image

from constants.status import QUALITY_STATUS_FAILED, QUALITY_STATUS_PASSED, QUALITY_STATUS_WARNING
from utils.config import get_settings


class QualityService:
    recommended_source_min_width = 480
    recommended_source_min_height = 640
    upscale_risk_margin = 1.05

    def __init__(self) -> None:
        self.settings = get_settings()

    @staticmethod
    def _normalize_image_shape(width: int, height: int) -> tuple[int, int, int]:
        return height, width, 3

    def _resolve_source_size(
        self,
        image: Image.Image,
        source_image: Image.Image | None = None,
        source_size: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        if source_size:
            return source_size
        if source_image is not None:
            return source_image.size
        return image.size

    @staticmethod
    def _matches_expected_size(image_size: tuple[int, int], expected_output_size: tuple[int, int] | None) -> bool:
        if expected_output_size is None:
            return True
        return image_size == expected_output_size

    def evaluate_details(
        self,
        image: Image.Image,
        *,
        source_image: Image.Image | None = None,
        source_size: tuple[int, int] | None = None,
        expected_output_size: tuple[int, int] | None = None,
        face_box: dict | None = None,
        blur_score: float | None = None,
    ) -> dict:
        output_width, output_height = image.size
        source_width, source_height = self._resolve_source_size(image, source_image=source_image, source_size=source_size)

        source_resolution_too_low = (
            source_width < self.settings.min_image_width or source_height < self.settings.min_image_height
        )
        source_resolution_borderline = (
            source_width < self.recommended_source_min_width or source_height < self.recommended_source_min_height
        )
        output_size_is_standard = self._matches_expected_size(
            (output_width, output_height),
            expected_output_size,
        )

        likely_upscaled = False
        if expected_output_size is not None:
            expected_width, expected_height = expected_output_size
            likely_upscaled = (
                source_width < int(expected_width * self.upscale_risk_margin)
                or source_height < int(expected_height * self.upscale_risk_margin)
            )

        face_too_small_risk = False
        if face_box:
            face_too_small_risk = self.is_face_too_small(
                self._normalize_image_shape(source_width, source_height),
                face_box,
            )
            likely_upscaled = likely_upscaled or face_too_small_risk

        blur_risk = blur_score is not None and self.is_image_too_blurry(blur_score)
        clarity_insufficient = blur_risk or face_too_small_risk or likely_upscaled or source_resolution_borderline

        if source_resolution_too_low:
            status = QUALITY_STATUS_FAILED
            message = f'原图分辨率过低，建议至少达到 {self.settings.min_image_width}x{self.settings.min_image_height}'
        elif blur_risk:
            status = QUALITY_STATUS_WARNING
            message = '原图疑似偏模糊，建议更换更清晰的照片以降低出片风险'
        elif face_too_small_risk:
            status = QUALITY_STATUS_WARNING
            message = '人脸区域偏小，生成时可能放大，建议换用人物占比更高的原图'
        elif likely_upscaled:
            status = QUALITY_STATUS_WARNING
            message = '原图与目标规格过于接近，生成时可能存在放大风险'
        elif source_resolution_borderline and expected_output_size is None:
            status = QUALITY_STATUS_WARNING
            message = '原图分辨率偏低，建议使用更高分辨率原图'
        elif not output_size_is_standard:
            status = QUALITY_STATUS_WARNING
            message = '输出尺寸与目标模板不完全一致，请检查规格配置'
        else:
            status = QUALITY_STATUS_PASSED
            if expected_output_size is not None:
                message = '原图质量与目标规格匹配，适合直接用于证件照输出'
            else:
                message = '质量通过'

        suitable = status != QUALITY_STATUS_FAILED and output_size_is_standard and not likely_upscaled

        return {
            'qualityStatus': status,
            'qualityMessage': message,
            'resolutionTooLow': source_resolution_too_low,
            'clarityInsufficient': clarity_insufficient,
            'suitableForIdPhoto': suitable,
            'sourceResolutionTooLow': source_resolution_too_low,
            'outputSizeIsStandard': output_size_is_standard,
            'likelyUpscaled': likely_upscaled,
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
