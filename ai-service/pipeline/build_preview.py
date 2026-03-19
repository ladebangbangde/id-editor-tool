from __future__ import annotations

from PIL import Image

from utils.config import get_settings
from utils.file_utils import build_output_path, public_url_for_path, to_url_like_path
from utils.image_utils import save_pil_image


class PreviewBuilder:
    def __init__(self):
        self.settings = get_settings()

    def build_preview(self, image_id: str, hd_image: Image.Image) -> dict:
        preview = hd_image.copy()
        preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
        output_path = build_output_path('preview', f'{image_id}_preview.jpg')
        save_pil_image(preview, output_path, quality=self.settings.preview_quality, dpi=self.settings.jpeg_dpi)
        return {
            'previewPath': to_url_like_path(output_path),
            'previewUrl': public_url_for_path(output_path),
        }
