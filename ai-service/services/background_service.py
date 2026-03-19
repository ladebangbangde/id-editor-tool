from PIL import Image

from constants.colors import BACKGROUND_COLORS


class BackgroundService:
    def apply_background(self, transparent_png_path: str, background_color: str) -> Image.Image:
        if background_color not in BACKGROUND_COLORS:
            raise ValueError(f"Unsupported background color: {background_color}")

        fg = Image.open(transparent_png_path).convert("RGBA")
        bg = Image.new("RGBA", fg.size, BACKGROUND_COLORS[background_color] + (255,))
        merged = Image.alpha_composite(bg, fg)
        return merged.convert("RGB")
