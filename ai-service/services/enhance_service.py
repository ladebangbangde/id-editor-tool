from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter


class EnhanceService:
    def enhance(self, image: Image.Image, beauty_enabled: bool) -> dict:
        if not beauty_enabled:
            return {'image': image, 'appliedOperations': ['keep_original']}

        sharp = image.filter(ImageFilter.SHARPEN)
        bright = ImageEnhance.Brightness(sharp).enhance(1.05)
        contrast = ImageEnhance.Contrast(bright).enhance(1.06)
        return {
            'image': contrast,
            'appliedOperations': ['sharpen', 'brightness_1.05', 'contrast_1.06'],
        }
