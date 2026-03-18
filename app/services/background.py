from PIL import Image, ImageFilter

from app.services.specs import get_background_color


class BackgroundService:
    def apply(self, foreground_rgba: Image.Image, background_color: str) -> Image.Image:
        color = get_background_color(background_color)
        fg = foreground_rgba.convert('RGBA')
        alpha = fg.getchannel('A').filter(ImageFilter.GaussianBlur(radius=0.6))
        softened = fg.copy()
        softened.putalpha(alpha)
        bg = Image.new('RGBA', softened.size, color + (255,))
        return Image.alpha_composite(bg, softened).convert('RGB')
