from PIL import Image

from constants.colors import BACKGROUND_COLORS
from core.exceptions import AppException, ERROR_INVALID_ARGUMENT
from utils.image_utils import open_pil_image


class BackgroundService:
    def apply_background(self, transparent_png_path: str, background_color: str) -> Image.Image:
        if background_color not in BACKGROUND_COLORS:
            raise AppException(f'Unsupported background color: {background_color}', ERROR_INVALID_ARGUMENT, 400)

        fg = open_pil_image(transparent_png_path).convert('RGBA')
        bg = Image.new('RGBA', fg.size, BACKGROUND_COLORS[background_color] + (255,))
        merged = Image.alpha_composite(bg, fg)
        return merged.convert('RGB')
