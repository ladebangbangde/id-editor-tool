import numpy as np
from PIL import Image, ImageFilter

from app.services.specs import get_background_color


class BackgroundService:
    LOWER_PROTECT_RATIO = 0.52
    UPPER_DILATE_KERNEL = 1
    LOWER_DILATE_KERNEL = 3
    ALPHA_GAMMA = 1.7

    def _build_conservative_alpha(self, foreground_rgba: Image.Image) -> Image.Image:
        """
        保守 matte：优先保持衣领/肩部/头发前景，不让底色向人物渗透。
        """
        alpha_np = np.array(foreground_rgba.getchannel('A'), dtype=np.uint8)
        h, _w = alpha_np.shape
        split = int(h * self.LOWER_PROTECT_RATIO)
        split = max(1, min(h - 1, split))

        upper = Image.fromarray(alpha_np[:split, :], mode='L')
        lower = Image.fromarray(alpha_np[split:, :], mode='L')

        if self.UPPER_DILATE_KERNEL > 1:
            upper = upper.filter(ImageFilter.MaxFilter(size=self.UPPER_DILATE_KERNEL * 2 + 1))
        if self.LOWER_DILATE_KERNEL > 1:
            lower = lower.filter(ImageFilter.MaxFilter(size=self.LOWER_DILATE_KERNEL * 2 + 1))

        merged = np.vstack(
            [
                np.array(upper, dtype=np.float32),
                np.array(lower, dtype=np.float32),
            ]
        ) / 255.0
        hardened = np.power(merged, self.ALPHA_GAMMA)
        original = alpha_np.astype(np.float32) / 255.0
        hardened[split:, :] = np.maximum(hardened[split:, :], original[split:, :])
        return Image.fromarray((hardened * 255.0).clip(0, 255).astype(np.uint8), mode='L')

    def apply(self, foreground_rgba: Image.Image, background_color: str) -> Image.Image:
        color = get_background_color(background_color)
        fg = foreground_rgba.convert('RGBA')
        alpha = self._build_conservative_alpha(fg)
        softened = fg.copy()
        softened.putalpha(alpha)
        bg = Image.new('RGBA', softened.size, color + (255,))
        return Image.alpha_composite(bg, softened).convert('RGB')
