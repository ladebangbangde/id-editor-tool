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
    edge_band_mask: Image.Image | None = None
    decontaminated_foreground: Image.Image | None = None


class MatteRefineService:
    # 边界精修阈值集中管理
    FG_THRESHOLD = 0.90
    BG_THRESHOLD = 0.08
    ERODE_SIZE = 2
    DILATE_SIZE = 4
    EDGE_BLUR_SIGMA = 0.8
    MIN_COMPONENT_AREA_RATIO = 0.00035
    DECONTAM_EDGE_LOW = 0.08
    DECONTAM_EDGE_HIGH = 0.92
    DECONTAM_BASE_STRENGTH = 0.55
    DECONTAM_MAX_STRENGTH = 0.88

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

    def _estimate_foreground_ml(self, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray | None:
        try:
            from pymatting import estimate_foreground_ml

            return estimate_foreground_ml(rgb, alpha)
        except Exception as exc:
            logger.info('pymatting foreground estimate unavailable, fallback to lightweight decontaminate: %s', exc)
            return None

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

    def _decontaminate_foreground_rgb(self, rgb: np.ndarray, alpha_refined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edge_band = (alpha_refined > self.DECONTAM_EDGE_LOW) & (alpha_refined < self.DECONTAM_EDGE_HIGH)
        sure_fg = alpha_refined >= self.DECONTAM_EDGE_HIGH
        if not np.any(edge_band):
            return rgb.copy(), edge_band.astype(np.uint8) * 255

        ml_foreground = self._estimate_foreground_ml(rgb, alpha_refined)
        if ml_foreground is not None:
            base_foreground = ml_foreground.astype(np.float32)
        else:
            weight = sure_fg.astype(np.float32)
            if np.count_nonzero(weight) < 12:
                weight = (alpha_refined >= 0.75).astype(np.float32)
            weighted_rgb = rgb * weight[:, :, None]
            smooth_w = filters.gaussian(weight, sigma=2.0, preserve_range=True)
            smooth_r = filters.gaussian(weighted_rgb[:, :, 0], sigma=2.0, preserve_range=True)
            smooth_g = filters.gaussian(weighted_rgb[:, :, 1], sigma=2.0, preserve_range=True)
            smooth_b = filters.gaussian(weighted_rgb[:, :, 2], sigma=2.0, preserve_range=True)
            denom = np.maximum(smooth_w, 1e-4)
            propagated = np.stack([smooth_r / denom, smooth_g / denom, smooth_b / denom], axis=2)
            base_foreground = np.clip(propagated, 0.0, 1.0)

        edge_softness = np.clip((self.DECONTAM_EDGE_HIGH - alpha_refined) / max(self.DECONTAM_EDGE_HIGH - self.DECONTAM_EDGE_LOW, 1e-6), 0.0, 1.0)
        blend_strength = np.clip(
            self.DECONTAM_BASE_STRENGTH + 0.33 * edge_softness,
            self.DECONTAM_BASE_STRENGTH,
            self.DECONTAM_MAX_STRENGTH,
        )
        blend_strength = np.where(edge_band, blend_strength, 0.0).astype(np.float32)
        decontaminated = rgb * (1.0 - blend_strength[:, :, None]) + base_foreground * blend_strength[:, :, None]
        decontaminated = np.clip(decontaminated, 0.0, 1.0)
        return decontaminated, edge_band.astype(np.uint8) * 255

    def refine(self, source_image: Image.Image, rgba_foreground: Image.Image) -> MatteRefineResult:
        fg_rgba = rgba_foreground.convert('RGBA')
        source_rgb = np.asarray(source_image.convert('RGB')).astype(np.float32) / 255.0
        fg_rgb = np.asarray(fg_rgba.convert('RGB')).astype(np.float32) / 255.0
        alpha_u8 = np.asarray(fg_rgba.getchannel('A'), dtype=np.uint8)

        trimap = self._build_trimap(alpha_u8)
        alpha_refined = self._estimate_alpha_cf(source_rgb, trimap)
        alpha_refined = self._postprocess_alpha(alpha_refined)
        refined_rgb, edge_band_mask = self._decontaminate_foreground_rgb(fg_rgb, alpha_refined)

        refined_rgba = np.asarray(fg_rgba, dtype=np.uint8).copy()
        refined_rgba[:, :, :3] = (refined_rgb * 255.0).clip(0, 255).astype(np.uint8)
        refined_rgba[:, :, 3] = (alpha_refined * 255.0).clip(0, 255).astype(np.uint8)

        return MatteRefineResult(
            rgba=Image.fromarray(refined_rgba, mode='RGBA'),
            alpha=Image.fromarray(refined_rgba[:, :, 3], mode='L'),
            trimap=Image.fromarray((trimap * 255.0).clip(0, 255).astype(np.uint8), mode='L'),
            edge_band_mask=Image.fromarray(edge_band_mask, mode='L'),
            decontaminated_foreground=Image.fromarray(refined_rgba[:, :, :3], mode='RGB'),
        )
