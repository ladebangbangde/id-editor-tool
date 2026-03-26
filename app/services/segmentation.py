import numpy as np
from PIL import Image, ImageFilter

from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    from rembg import remove  # type: ignore
except Exception:  # rembg is optional at runtime when native deps are missing
    remove = None


class SegmentationService:
    def remove_background(self, image: Image.Image) -> Image.Image:
        rgba = image.convert('RGBA')
        if remove is not None:
            try:
                output = remove(rgba)
                if isinstance(output, Image.Image):
                    return output.convert('RGBA')
                return Image.open(output).convert('RGBA')
            except Exception as exc:
                logger.warning('rembg failed, falling back to border-based matting: %s', exc)
        return self._fallback_remove_background(rgba)

    def _fallback_remove_background(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert('RGB')).astype(np.float32)
        h, w = rgb.shape[:2]
        border = max(6, min(h, w) // 20)
        samples = np.concatenate(
            [
                rgb[:border, :, :].reshape(-1, 3),
                rgb[-border:, :, :].reshape(-1, 3),
                rgb[:, :border, :].reshape(-1, 3),
                rgb[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(samples, axis=0)
        dist = np.linalg.norm(rgb - bg, axis=2)
        threshold = max(22.0, float(np.percentile(dist, 65)))
        alpha = np.clip((dist - threshold * 0.45) / max(threshold * 0.9, 1.0), 0, 1)
        alpha = (alpha * 255).astype(np.uint8)
        # 下半身(衣领/肩部)区域保守保留，避免背景替换后衣服被侵蚀。
        split = int(alpha.shape[0] * 0.52)
        split = max(1, min(alpha.shape[0] - 1, split))
        alpha[split:, :] = np.maximum(alpha[split:, :], 182)
        result = image.copy()
        result.putalpha(Image.fromarray(alpha, mode='L').filter(ImageFilter.GaussianBlur(radius=0.7)))
        return result
