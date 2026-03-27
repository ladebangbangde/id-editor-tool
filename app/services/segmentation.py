import os

import numpy as np
from PIL import Image, ImageFilter

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logger import get_logger
from app.services.baidu_human_segmentation_service import BaiduHumanSegmentationService

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
        self._last_debug_images: dict[str, Image.Image] = {}
        self._baidu_service = BaiduHumanSegmentationService() if self.settings.baidu_segmentation_enabled else None
        self._init_rembg_session()

    def _init_rembg_session(self) -> None:
        if remove is None or new_session is None:
            logger.warning('rembg unavailable')
            return
        model_root = os.getenv('U2NET_HOME', '/root/.u2net')
        self._rembg_model_path = f'{model_root}/u2net.onnx'
        if not os.path.exists(self._rembg_model_path):
            logger.warning('rembg model missing path=%s', self._rembg_model_path)
            return
        try:
            self._rembg_session = new_session(model_name='u2net')
            logger.info('rembg model loaded model=u2net path=%s', self._rembg_model_path)
        except Exception as exc:
            logger.warning('rembg session init failed: %s', exc)

    def consume_debug_images(self) -> dict[str, Image.Image]:
        snapshots = dict(self._last_debug_images)
        self._last_debug_images.clear()
        return snapshots

    def remove_background(self, image: Image.Image) -> Image.Image:
        rgba = image.convert('RGBA')
        self._last_debug_images = {}

        if self.settings.baidu_segmentation_enabled:
            if self._baidu_service is None:
                raise AppError(
                    code='BAIDU_SEGMENTATION_NOT_INITIALIZED',
                    message='Baidu segmentation backend is enabled but service is not initialized',
                    status_code=500,
                )
            logger.info('Segmentation backend=baidu start')
            result = self._baidu_service.segment_human(rgba)
            self._last_debug_images['baidu_foreground.png'] = result.foreground
            if result.labelmap is not None:
                self._last_debug_images['baidu_labelmap.png'] = result.labelmap
            if result.scoremap is not None:
                self._last_debug_images['baidu_scoremap.png'] = result.scoremap
            logger.info('Segmentation backend=baidu success')
            return result.foreground.convert('RGBA')

        logger.warning('Segmentation backend=rembg (BAIDU_SEGMENTATION_ENABLED=false)')
        return self._remove_background_with_rembg(rgba)

    def _remove_background_with_rembg(self, image: Image.Image) -> Image.Image:
        if remove is not None and self._rembg_session is not None:
            try:
                output = remove(image, session=self._rembg_session)
                logger.info('rembg remove_background active model=u2net path=%s', self._rembg_model_path or 'unknown')
                if isinstance(output, Image.Image):
                    return output.convert('RGBA')
                return Image.open(output).convert('RGBA')
            except Exception as exc:
                logger.warning('rembg failed, falling back to border-based matting: %s', exc)
        else:
            logger.warning('rembg session unavailable, falling back to border-based matting')
        return self._fallback_remove_background(image)

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
