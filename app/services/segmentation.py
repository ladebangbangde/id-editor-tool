import numpy as np
from PIL import Image, ImageFilter
import os

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    from rembg import new_session, remove  # type: ignore
except Exception:  # rembg is optional at runtime when native deps are missing
    new_session = None
    remove = None


class SegmentationService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._rembg_session = None
        self._rembg_model_path = None
        self._rembg_fallback_happened = False
        self._init_rembg_session()

    def _init_rembg_session(self) -> None:
        if remove is None or new_session is None:
            logger.warning('rembg unavailable, border-based matting fallback enabled')
            return
        model_root = os.getenv('U2NET_HOME', '/root/.u2net')
        self._rembg_model_path = f'{model_root}/u2net.onnx'
        if not os.path.exists(self._rembg_model_path):
            self._rembg_fallback_happened = True
            logger.warning(
                'rembg model missing path=%s; runtime download disabled, border-based fallback enabled',
                self._rembg_model_path,
            )
            return
        try:
            self._rembg_session = new_session(model_name='u2net')
            logger.info('rembg model loaded model=u2net path=%s', self._rembg_model_path)
        except Exception as exc:
            logger.warning('rembg session init failed, border-based matting fallback enabled: %s', exc)

    def remove_background(self, image: Image.Image) -> Image.Image:
        rgba = image.convert('RGBA')
        if remove is not None and self._rembg_session is not None:
            try:
                output = remove(rgba, session=self._rembg_session)
                logger.info(
                    'rembg remove_background active model=u2net path=%s fallback=%s',
                    self._rembg_model_path or 'unknown',
                    self._rembg_fallback_happened,
                )
                if isinstance(output, Image.Image):
                    return output.convert('RGBA')
                return Image.open(output).convert('RGBA')
            except Exception as exc:
                self._rembg_fallback_happened = True
                logger.warning('rembg failed, falling back to border-based matting: %s', exc)
        else:
            self._rembg_fallback_happened = True
            logger.warning('rembg session unavailable, falling back to border-based matting')
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
