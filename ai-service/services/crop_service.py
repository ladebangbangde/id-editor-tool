from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class CropResult:
    image: Image.Image
    cropBox: dict[str, int]
    targetWidth: int
    targetHeight: int
    headRatio: float
    method: str


class CropService:
    @staticmethod
    def _center_crop_box(width: int, height: int, target_ratio: float) -> tuple[int, int, int, int]:
        current_ratio = width / height
        if current_ratio > target_ratio:
            new_w = int(height * target_ratio)
            left = max((width - new_w) // 2, 0)
            return left, 0, left + new_w, height

        new_h = int(width / target_ratio)
        top = max((height - new_h) // 2, 0)
        return 0, top, width, top + new_h

    def _face_guided_crop_box(
        self,
        width: int,
        height: int,
        target_ratio: float,
        face_box: dict,
    ) -> tuple[tuple[int, int, int, int], float]:
        face_cx = face_box['x'] + face_box['width'] / 2
        face_cy = face_box['y'] + face_box['height'] / 2
        desired_head_ratio = 0.58
        desired_top_margin_ratio = 0.18

        crop_height = min(height, max(int(face_box['height'] / desired_head_ratio), face_box['height']))
        crop_width = min(width, max(int(crop_height * target_ratio), face_box['width']))
        crop_height = min(height, max(int(crop_width / target_ratio), crop_height))
        crop_width = min(width, max(int(crop_height * target_ratio), crop_width))

        top = int(face_box['y'] - crop_height * desired_top_margin_ratio)
        left = int(face_cx - crop_width / 2)

        left = min(max(left, 0), max(width - crop_width, 0))
        top = min(max(top, 0), max(height - crop_height, 0))
        right = min(left + crop_width, width)
        bottom = min(top + crop_height, height)

        if right - left <= 0 or bottom - top <= 0:
            return self._center_crop_box(width, height, target_ratio), 0.0

        crop_height = bottom - top
        head_ratio = round(face_box['height'] / max(crop_height, 1), 3)
        return (left, top, right, bottom), head_ratio

    def crop_to_size(
        self,
        img: Image.Image,
        pixel_width: int,
        pixel_height: int,
        face_box: dict | None = None,
    ) -> CropResult:
        width, height = img.size
        target_ratio = pixel_width / pixel_height
        method = 'center_crop_fallback'
        head_ratio = 0.0

        if face_box:
            crop_box, head_ratio = self._face_guided_crop_box(width, height, target_ratio, face_box)
            method = 'face_guided_crop'
        else:
            crop_box = self._center_crop_box(width, height, target_ratio)

        cropped = img.crop(crop_box).resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
        left, top, right, bottom = crop_box
        return CropResult(
            image=cropped,
            cropBox={'x': int(left), 'y': int(top), 'width': int(right - left), 'height': int(bottom - top)},
            targetWidth=pixel_width,
            targetHeight=pixel_height,
            headRatio=head_ratio,
            method=method,
        )
