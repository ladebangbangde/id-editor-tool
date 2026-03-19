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

    def crop_to_size(self, img: Image.Image, pixel_width: int, pixel_height: int) -> Image.Image:
        target_ratio = pixel_width / pixel_height
        cropped = self._center_crop(img, target_ratio)
        return cropped.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
