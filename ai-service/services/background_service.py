from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from constants.colors import BACKGROUND_COLORS
from utils.config import get_settings
from utils.file_utils import public_url_for_path, to_url_like_path


class BackgroundService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _build_conservative_alpha(self, fg_rgba: Image.Image) -> np.ndarray:
        """收敛 matte：优先保护前景（特别是衣领/肩部），避免背景色向人物渗透。"""
        alpha = np.array(fg_rgba.getchannel('A'), dtype=np.uint8)
        h, w = alpha.shape
        split = int(h * max(0.0, min(1.0, self.settings.composite_lower_protect_ratio)))
        upper = alpha[:split, :]
        lower = alpha[split:, :]

        upper_kernel = max(1, int(self.settings.composite_dilate_kernel_upper))
        lower_kernel = max(1, int(self.settings.composite_dilate_kernel_lower))
        upper_img = Image.fromarray(upper, mode='L')
        lower_img = Image.fromarray(lower, mode='L')
        if upper_kernel > 1:
            upper_img = upper_img.filter(ImageFilter.MaxFilter(size=upper_kernel * 2 + 1))
        if lower_kernel > 1:
            lower_img = lower_img.filter(ImageFilter.MaxFilter(size=lower_kernel * 2 + 1))
        upper_closed = np.array(upper_img, dtype=np.uint8)
        lower_dilated = np.array(lower_img, dtype=np.uint8)

        merged = np.vstack([upper_closed, lower_dilated]).astype(np.float32) / 255.0
        gamma = max(self.settings.composite_alpha_gamma, 1.0)
        hardened = np.power(merged, gamma)  # 让软边更克制，减少衣服染色
        # 下半身（衣领/肩部）优先保真：不允许比原始 alpha 更薄，避免背景侵入衣服颜色。
        original_norm = alpha.astype(np.float32) / 255.0
        hardened[split:, :] = np.maximum(hardened[split:, :], original_norm[split:, :])
        return (hardened * 255.0).clip(0, 255).astype(np.uint8)

    def apply_background(
        self,
        transparent_png_path: str,
        background_color: str,
        preview_path: str | None = None,
    ) -> dict:
        if background_color not in BACKGROUND_COLORS:
            raise ValueError(f'Unsupported background color: {background_color}')

        fg = Image.open(transparent_png_path).convert('RGBA')
        alpha = self._build_conservative_alpha(fg)
        fg_arr = np.array(fg, dtype=np.uint8)
        fg_arr[:, :, 3] = alpha
        fg = Image.fromarray(fg_arr, mode='RGBA')
        bg = Image.new('RGBA', fg.size, BACKGROUND_COLORS[background_color] + (255,))
        merged = Image.alpha_composite(bg, fg).convert('RGB')
        return {
            'image': merged,
            'backgroundColor': background_color,
            'method': 'segmentation_composite',
            'outputPath': to_url_like_path(preview_path) if preview_path else None,
            'outputUrl': public_url_for_path(preview_path) if preview_path else None,
            'previewUrl': public_url_for_path(preview_path) if preview_path else None,
            'note': None,
        }

    def fallback_original(self, image_path: str, background_color: str, reason: str) -> dict:
        image = Image.open(image_path).convert('RGB')
        return {
            'image': image,
            'backgroundColor': background_color,
            'method': 'original_image_fallback',
            'outputPath': to_url_like_path(image_path),
            'outputUrl': public_url_for_path(image_path),
            'previewUrl': public_url_for_path(image_path),
            'note': reason,
        }
