import numpy as np
from PIL import Image, ImageFilter

from app.core.config import get_settings
from app.services.specs import get_background_color


class BackgroundService:
    LOWER_PROTECT_RATIO = 0.52
    UPPER_DILATE_KERNEL = 1
    LOWER_DILATE_KERNEL = 3
    ALPHA_GAMMA = 1.7
    HIGH_CONFIDENCE_ALPHA = 0.92
    EDGE_TRANSITION_LOW = 0.10
    EDGE_TRANSITION_HIGH = 0.92
    EDGE_BG_RATIO = 0.35

    def __init__(self) -> None:
        self.settings = get_settings()

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
        if self.settings.enable_safe_edge_background_compose:
            return self.apply_safe_idphoto_background(fg, color, background_color.lower())
        return self._legacy_apply(fg, color)

    def _legacy_apply(self, foreground_rgba: Image.Image, color: tuple[int, int, int]) -> Image.Image:
        fg = foreground_rgba.convert('RGBA')
        alpha = self._build_conservative_alpha(fg)
        softened = fg.copy()
        softened.putalpha(alpha)
        bg = Image.new('RGBA', softened.size, color + (255,))
        return Image.alpha_composite(bg, softened).convert('RGB')

    def apply_edge_aware(self, foreground_rgba: Image.Image, background_color: str) -> Image.Image:
        color = get_background_color(background_color)
        fg_np = np.asarray(foreground_rgba.convert('RGBA'), dtype=np.uint8).astype(np.float32)
        alpha = fg_np[:, :, 3] / 255.0
        fg_rgb = fg_np[:, :, :3]
        bg_rgb = np.zeros_like(fg_rgb) + np.array(color, dtype=np.float32)

        high_conf = alpha >= self.HIGH_CONFIDENCE_ALPHA
        edge_zone = (alpha >= self.EDGE_TRANSITION_LOW) & (alpha < self.EDGE_TRANSITION_HIGH)
        edge_alpha = np.clip((alpha - self.EDGE_TRANSITION_LOW) / (self.EDGE_TRANSITION_HIGH - self.EDGE_TRANSITION_LOW), 0.0, 1.0)
        conservative_edge_alpha = np.maximum(edge_alpha, alpha * self.EDGE_BG_RATIO)

        out_rgb = bg_rgb.copy()
        out_rgb[high_conf] = fg_rgb[high_conf]
        out_rgb[edge_zone] = (
            fg_rgb[edge_zone] * conservative_edge_alpha[edge_zone, None]
            + bg_rgb[edge_zone] * (1.0 - conservative_edge_alpha[edge_zone, None])
        )
        pure_bg = alpha < self.EDGE_TRANSITION_LOW
        out_rgb[pure_bg] = bg_rgb[pure_bg]
        return Image.fromarray(out_rgb.clip(0, 255).astype(np.uint8), mode='RGB')

    def apply_safe_idphoto_background(
        self,
        foreground_rgba: Image.Image,
        color: tuple[int, int, int],
        color_name: str,
    ) -> Image.Image:
        fg_np = np.asarray(foreground_rgba.convert('RGBA'), dtype=np.uint8).astype(np.float32)
        alpha = fg_np[:, :, 3] / 255.0
        fg_rgb = fg_np[:, :, :3]
        bg_rgb = np.zeros_like(fg_rgb) + np.array(color, dtype=np.float32)
        h, w = alpha.shape

        high_conf = alpha >= 0.95
        pure_bg = alpha <= 0.04
        edge_zone = ~(high_conf | pure_bg)
        edge_soft = np.clip((alpha - 0.04) / 0.91, 0.0, 1.0)
        edge_soft = np.power(edge_soft, 0.8)

        if color_name in {'red', 'blue'}:
            sat_boost = np.max(fg_rgb, axis=2) - np.min(fg_rgb, axis=2)
            anti_bleed = np.clip(1.0 - sat_boost / 255.0, 0.25, 0.95)
            edge_soft = np.clip(edge_soft * anti_bleed, 0.25, 0.95)

        # 面部核心区保护：上半区域中心椭圆不允许底色直接混入。
        yy, xx = np.ogrid[:h, :w]
        cx = w * 0.5
        cy = h * 0.36
        rx = max(1.0, w * 0.17)
        ry = max(1.0, h * 0.21)
        face_protect = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        face_protect = face_protect & (alpha > 0.20)
        edge_soft[face_protect] = np.maximum(edge_soft[face_protect], 0.92)

        # 下半部分保守保留，但仍去除明显污染色。
        lower_split = int(h * self.LOWER_PROTECT_RATIO)
        lower_split = max(1, min(h - 1, lower_split))
        lower_mask = np.zeros((h, w), dtype=bool)
        lower_mask[lower_split:, :] = True
        edge_soft[lower_mask] = np.maximum(edge_soft[lower_mask], alpha[lower_mask] * 0.92)

        out_rgb = bg_rgb.copy()
        out_rgb[high_conf] = fg_rgb[high_conf]
        out_rgb[pure_bg] = bg_rgb[pure_bg]
        out_rgb[edge_zone] = fg_rgb[edge_zone] * edge_soft[edge_zone, None] + bg_rgb[edge_zone] * (1.0 - edge_soft[edge_zone, None])

        return Image.fromarray(np.clip(out_rgb, 0, 255).astype(np.uint8), mode='RGB')
