from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage import filters, measure, morphology

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatteRefineResult:
    rgba: Image.Image
    alpha: Image.Image
    trimap: Image.Image


class MatteRefineService:
    # 边界精修阈值集中管理
    FG_THRESHOLD = 0.90
    BG_THRESHOLD = 0.08
    ERODE_SIZE = 2
    DILATE_SIZE = 4
    EDGE_BLUR_SIGMA = 0.8
    MIN_COMPONENT_AREA_RATIO = 0.00035

    def _cv2(self):
        try:
            import cv2

            return cv2
        except Exception:
            return None

    def _estimate_alpha_cf(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        try:
            from pymatting import estimate_alpha_cf

            return estimate_alpha_cf(rgb, trimap)
        except Exception as exc:
            logger.warning('pymatting unavailable/fail, fallback to original alpha: %s', exc)
            # 回退时 unknown 区域使用 trimap 中值，保证流程稳定。
            return np.where(trimap >= 0.99, 1.0, np.where(trimap <= 0.01, 0.0, 0.5)).astype(np.float32)

    def _build_trimap(self, alpha_u8: np.ndarray) -> np.ndarray:
        alpha = alpha_u8.astype(np.float32) / 255.0

        sure_fg = (alpha >= self.FG_THRESHOLD).astype(np.uint8)
        sure_bg = (alpha <= self.BG_THRESHOLD).astype(np.uint8)

        sure_fg = morphology.erosion(sure_fg.astype(bool), morphology.disk(self.ERODE_SIZE)).astype(np.uint8)
        sure_bg = morphology.dilation(sure_bg.astype(bool), morphology.disk(self.DILATE_SIZE)).astype(np.uint8)

        trimap = np.full(alpha.shape, 0.5, dtype=np.float32)
        trimap[sure_fg > 0] = 1.0
        trimap[sure_bg > 0] = 0.0
        return trimap

    def _postprocess_alpha(self, alpha: np.ndarray) -> np.ndarray:
        alpha_u8 = (alpha.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        blurred = filters.gaussian(alpha_u8.astype(np.float32), sigma=self.EDGE_BLUR_SIGMA, preserve_range=True).astype(np.uint8)
        hard_mask = blurred > 1

        labels = measure.label(hard_mask, connectivity=2)
        min_area = max(20, int(alpha_u8.size * self.MIN_COMPONENT_AREA_RATIO))
        cleaned = np.zeros_like(alpha_u8)
        for region in measure.regionprops(labels):
            if region.area >= min_area:
                cleaned[labels == region.label] = alpha_u8[labels == region.label]

        if not np.any(cleaned):
            cleaned = alpha_u8

        return cleaned.astype(np.float32) / 255.0

    def refine(self, source_image: Image.Image, rgba_foreground: Image.Image) -> MatteRefineResult:
        fg_rgba = rgba_foreground.convert('RGBA')
        rgb = np.asarray(source_image.convert('RGB')).astype(np.float32) / 255.0
        alpha_u8 = np.asarray(fg_rgba.getchannel('A'), dtype=np.uint8)

        trimap = self._build_trimap(alpha_u8)
        alpha_refined = self._estimate_alpha_cf(rgb, trimap)
        alpha_refined = self._postprocess_alpha(alpha_refined)

        refined_rgba = np.asarray(fg_rgba, dtype=np.uint8).copy()
        refined_rgba[:, :, 3] = (alpha_refined * 255.0).clip(0, 255).astype(np.uint8)

        return MatteRefineResult(
            rgba=Image.fromarray(refined_rgba, mode='RGBA'),
            alpha=Image.fromarray(refined_rgba[:, :, 3], mode='L'),
            trimap=Image.fromarray((trimap * 255.0).clip(0, 255).astype(np.uint8), mode='L'),
        )
