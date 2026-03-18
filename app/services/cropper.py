from PIL import Image

from app.services.specs import PhotoSpec


class CropperService:
    def crop(self, image: Image.Image, spec: PhotoSpec, face_box: dict[str, int] | None) -> Image.Image:
        rgba = image.convert('RGBA')
        width, height = rgba.size
        target_ratio = spec.width_px / spec.height_px

        if face_box:
            fx = face_box['x'] + face_box['width'] / 2
            fy = face_box['y'] + face_box['height'] / 2
            crop_height = min(height, max(face_box['height'] / 0.42, spec.height_px))
            crop_width = crop_height * target_ratio
            if crop_width > width:
                crop_width = width
                crop_height = crop_width / target_ratio
            left = max(0, min(width - crop_width, fx - crop_width / 2))
            top = max(0, min(height - crop_height, fy - crop_height * 0.38))
        else:
            crop_width = min(width, height * target_ratio)
            crop_height = crop_width / target_ratio
            left = (width - crop_width) / 2
            top = (height - crop_height) / 2

        box = (
            int(round(left)),
            int(round(top)),
            int(round(left + crop_width)),
            int(round(top + crop_height)),
        )
        cropped = rgba.crop(box)
        return cropped.resize((spec.width_px, spec.height_px), Image.Resampling.LANCZOS)
