import numpy as np
from PIL import Image, ImageFilter

from app.services.specs import get_background_color


class BackgroundService:
    LOWER_PROTECT_RATIO = 0.52
    UPPER_DILATE_KERNEL = 1
    LOWER_DILATE_KERNEL = 3
    ALPHA_GAMMA = 1.7
    CONFIDENT_ALPHA = 0.90
    PROTECT_ALPHA = 0.62
    EDGE_LOW = 0.08
    EDGE_HIGH = 0.90

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
        alpha_f = np.asarray(alpha, dtype=np.float32) / 255.0
        fg_np = np.asarray(fg, dtype=np.uint8)
        fg_rgb = fg_np[:, :, :3].astype(np.float32)

        h, w = alpha_f.shape
        yy, xx = np.ogrid[:h, :w]
        shoulder_region = (
            (yy >= h * 0.36)
            & (yy <= h * 0.72)
            & (xx >= w * 0.08)
            & (xx <= w * 0.92)
        )
        lower_region = yy >= h * self.LOWER_PROTECT_RATIO
        protected_zone = shoulder_region | lower_region

        confident_fg = alpha_f >= self.CONFIDENT_ALPHA
        protected_fg = (alpha_f >= self.PROTECT_ALPHA) & protected_zone
        edge_band = (alpha_f > self.EDGE_LOW) & (alpha_f < self.EDGE_HIGH)

        bg_rgb = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        composited = fg_rgb * alpha_f[:, :, None] + bg_rgb * (1.0 - alpha_f[:, :, None])

        preserve_strength = np.zeros_like(alpha_f, dtype=np.float32)
        preserve_strength[confident_fg] = 1.0
        preserve_strength[protected_fg] = np.maximum(preserve_strength[protected_fg], 0.88)
        preserve_strength[edge_band & protected_zone] = np.maximum(
            preserve_strength[edge_band & protected_zone],
            0.48 + 0.22 * alpha_f[edge_band & protected_zone],
        )
        preserve_strength = np.clip(preserve_strength, 0.0, 1.0)

        stabilized = composited * (1.0 - preserve_strength[:, :, None]) + fg_rgb * preserve_strength[:, :, None]
        return Image.fromarray(stabilized.clip(0, 255).astype(np.uint8), mode='RGB')
