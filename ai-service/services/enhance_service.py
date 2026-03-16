from PIL import Image, ImageEnhance, ImageFilter


class EnhanceService:
    def enhance(self, image: Image.Image, beauty_enabled: bool) -> Image.Image:
        if not beauty_enabled:
            return image

        sharp = image.filter(ImageFilter.SHARPEN)
        bright = ImageEnhance.Brightness(sharp).enhance(1.05)
        contrast = ImageEnhance.Contrast(bright).enhance(1.06)
        return contrast
