from __future__ import annotations

from PIL import Image

from constants.colors import BACKGROUND_COLORS
from utils.file_utils import public_url_for_path, to_url_like_path
from utils.logger import get_logger


class BackgroundService:
    def __init__(self) -> None:
        self.logger = get_logger(component='background_service')

    def apply_background(
        self,
        transparent_png_path: str,
        background_color: str,
        preview_path: str | None = None,
    ) -> dict:
        background_logger = self.logger.bind(transparent_png_path=transparent_png_path, background_color=background_color)
        if background_color not in BACKGROUND_COLORS:
            background_logger.warning('unsupported background color requested')
            raise ValueError(f'Unsupported background color: {background_color}')

        background_logger.info('applying background color to segmented foreground')
        fg = Image.open(transparent_png_path).convert('RGBA')
        bg = Image.new('RGBA', fg.size, BACKGROUND_COLORS[background_color] + (255,))
        merged = Image.alpha_composite(bg, fg).convert('RGB')
        background_logger.bind(width=merged.size[0], height=merged.size[1]).info('background composition completed')
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
        self.logger.bind(image_path=image_path, background_color=background_color).warning('falling back to original image: {}', reason)
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
