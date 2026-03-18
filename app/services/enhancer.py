from PIL import Image, ImageEnhance, ImageFilter


class EnhancerService:
    def enhance(self, image: Image.Image) -> Image.Image:
        enhanced = ImageEnhance.Contrast(image).enhance(1.08)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.03)
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        return enhanced.filter(ImageFilter.UnsharpMask(radius=1.4, percent=125, threshold=2))
