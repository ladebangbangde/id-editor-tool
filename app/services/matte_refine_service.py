from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage import filters, measure, morphology

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatteRefineResult:
    rgba: Image.Image
    alpha: Image.Image
    trimap: Image.Image
    decontaminated_rgba: Image.Image | None = None
    edge_band_mask: Image.Image | None = None
    guided_alpha: Image.Image | None = None


class MatteRefineService:
    # 边界精修阈值集中管理
    FG_THRESHOLD = 0.90
    BG_THRESHOLD = 0.08
    ERODE_SIZE = 2
    DILATE_SIZE = 4
    EDGE_BLUR_SIGMA = 0.8
    MIN_COMPONENT_AREA_RATIO = 0.00035
    EDGE_BAND_ALPHA_LOW = 14
    EDGE_BAND_ALPHA_HIGH = 236
    UNKNOWN_TRIMAP_MIN = 0.08
    UNKNOWN_TRIMAP_MAX = 0.92
    EDGE_DECONTAMINATION_STRENGTH = 0.62

    def __init__(self) -> None:
        self.settings = get_settings()

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

    def _estimate_foreground_ml(self, source_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray | None:
        try:
            from pymatting import estimate_foreground_ml

            foreground = estimate_foreground_ml(source_rgb, alpha)
            return np.clip(foreground.astype(np.float32), 0.0, 1.0)
        except Exception as exc:
            logger.warning('pymatting foreground estimate unavailable/fail: %s', exc)
            return None

    def _guided_refine_alpha(self, source_rgb: np.ndarray, alpha_refined: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.settings.enable_guided_edge_refinement:
            return alpha_refined, None
        cv2 = self._cv2()
        if cv2 is None or not hasattr(cv2, 'ximgproc'):
            logger.warning('guided edge refinement skipped: cv2.ximgproc unavailable')
            return alpha_refined, None
        try:
            guide = (source_rgb * 255.0).clip(0, 255).astype(np.uint8)
            src = np.clip(alpha_refined.astype(np.float32), 0.0, 1.0)
            refined = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=8, eps=1e-3)
            refined = np.clip(refined.astype(np.float32), 0.0, 1.0)
            return refined, refined
        except Exception as exc:
            logger.warning('guided edge refinement failed, fallback to alpha_refined: %s', exc)
            return alpha_refined, None

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

    def _build_edge_band_mask(self, alpha_u8: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        edge_from_alpha = (alpha_u8 > self.EDGE_BAND_ALPHA_LOW) & (alpha_u8 < self.EDGE_BAND_ALPHA_HIGH)
        edge_from_trimap = (trimap > self.UNKNOWN_TRIMAP_MIN) & (trimap < self.UNKNOWN_TRIMAP_MAX)
        return (edge_from_alpha | edge_from_trimap).astype(np.float32)

    def _decontaminate_foreground_rgb(
        self,
        source_rgb: np.ndarray,
        fg_rgba: np.ndarray,
        alpha_refined: np.ndarray,
        edge_band_mask: np.ndarray,
    ) -> np.ndarray:
        fg_rgb = fg_rgba[:, :, :3].astype(np.float32) / 255.0
        alpha = np.clip(alpha_refined.astype(np.float32), 0.0, 1.0)
        edge = np.clip(edge_band_mask.astype(np.float32), 0.0, 1.0)

        safe_alpha = np.clip(alpha, 0.20, 0.98)
        estimated_fg = (fg_rgb - (1.0 - alpha)[..., None] * source_rgb) / safe_alpha[..., None]
        estimated_fg = np.clip(estimated_fg, 0.0, 1.0)
        pymatting_fg = self._estimate_foreground_ml(source_rgb=source_rgb, alpha=alpha)
        if pymatting_fg is not None:
            estimated_fg = np.clip(estimated_fg * 0.25 + pymatting_fg * 0.75, 0.0, 1.0)
        blend_strength = (edge * (1.0 - alpha) * self.EDGE_DECONTAMINATION_STRENGTH)[..., None]
        decontaminated = fg_rgb * (1.0 - blend_strength) + estimated_fg * blend_strength
        decontaminated = np.clip(decontaminated, 0.0, 1.0)

        out = fg_rgba.copy()
        out[:, :, :3] = (decontaminated * 255.0).astype(np.uint8)
        out[:, :, 3] = (alpha * 255.0).astype(np.uint8)
        return out

    def refine(self, source_image: Image.Image, rgba_foreground: Image.Image) -> MatteRefineResult:
        fg_rgba = rgba_foreground.convert('RGBA')
        rgb = np.asarray(source_image.convert('RGB')).astype(np.float32) / 255.0
        fg_rgba_np = np.asarray(fg_rgba, dtype=np.uint8)
        alpha_u8 = np.asarray(fg_rgba.getchannel('A'), dtype=np.uint8)

        trimap = self._build_trimap(alpha_u8)
        alpha_refined = self._estimate_alpha_cf(rgb, trimap)
        alpha_refined = self._postprocess_alpha(alpha_refined)
        alpha_refined, guided_alpha_debug = self._guided_refine_alpha(source_rgb=rgb, alpha_refined=alpha_refined)

        refined_rgba = fg_rgba_np.copy()
        refined_rgba[:, :, 3] = (alpha_refined * 255.0).clip(0, 255).astype(np.uint8)
        edge_band_mask = self._build_edge_band_mask(alpha_u8=alpha_u8, trimap=trimap)
        decontaminated_rgba = None
        if self.settings.enable_foreground_decontamination:
            decontaminated_rgba = self._decontaminate_foreground_rgb(
                source_rgb=rgb,
                fg_rgba=refined_rgba,
                alpha_refined=alpha_refined,
                edge_band_mask=edge_band_mask,
            )

        return MatteRefineResult(
            rgba=Image.fromarray(refined_rgba, mode='RGBA'),
            alpha=Image.fromarray(refined_rgba[:, :, 3], mode='L'),
            trimap=Image.fromarray((trimap * 255.0).clip(0, 255).astype(np.uint8), mode='L'),
            decontaminated_rgba=Image.fromarray(decontaminated_rgba, mode='RGBA') if decontaminated_rgba is not None else None,
            edge_band_mask=Image.fromarray((edge_band_mask * 255.0).astype(np.uint8), mode='L'),
            guided_alpha=(
                Image.fromarray((guided_alpha_debug * 255.0).clip(0, 255).astype(np.uint8), mode='L')
                if guided_alpha_debug is not None
                else None
            ),
        )
