from __future__ import annotations

from PIL import Image


class CropService:
    @staticmethod
    def _center_crop(img: Image.Image, target_ratio: float) -> Image.Image:
        w, h = img.size
        current_ratio = w / h
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            box = (left, 0, left + new_w, h)
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            box = (0, top, w, top + new_h)
        return img.crop(box)

    @staticmethod
    def _face_aware_crop_box(img: Image.Image, target_ratio: float, face_box: dict) -> tuple[int, int, int, int]:
        width, height = img.size
        fx = face_box['x']
        fy = face_box['y']
        fw = face_box['width']
        fh = face_box['height']

        desired_face_ratio = 0.38
        crop_h = max(int(fh / desired_face_ratio), fh + 40)
        crop_w = int(crop_h * target_ratio)

        if crop_w > width:
            crop_w = width
            crop_h = int(crop_w / target_ratio)
        if crop_h > height:
            crop_h = height
            crop_w = int(crop_h * target_ratio)

        face_center_x = fx + fw / 2
        desired_top_padding = max(int(fh * 0.9), 20)
        crop_left = int(round(face_center_x - crop_w / 2))
        crop_top = int(round(fy - desired_top_padding))

        crop_left = max(0, min(crop_left, width - crop_w))
        crop_top = max(0, min(crop_top, height - crop_h))

        if fy + fh > crop_top + crop_h:
            crop_top = max(0, min(fy + fh - crop_h, height - crop_h))

        return crop_left, crop_top, crop_left + crop_w, crop_top + crop_h

    def crop_to_size(
        self,
        img: Image.Image,
        pixel_width: int,
        pixel_height: int,
        face_box: dict | None = None,
    ) -> Image.Image:
        target_ratio = pixel_width / pixel_height
        if face_box:
            crop_box = self._face_aware_crop_box(img, target_ratio, face_box)
            cropped = img.crop(crop_box)
        else:
            cropped = self._center_crop(img, target_ratio)
        return cropped.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
