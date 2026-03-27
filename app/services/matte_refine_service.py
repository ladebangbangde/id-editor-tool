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
    hair_internal_holes_mask: Image.Image | None = None
    hair_gap_filled_alpha: Image.Image | None = None
    border_residue_mask: Image.Image | None = None


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
    CORE_FOREGROUND_ALPHA = 0.96
    EDGE_PROPAGATION_BLEND = 0.45
    HAIR_HOLE_MAX_AREA_RATIO = 0.0018
    HAIR_HOLE_MIN_AREA = 3
    BORDER_RESIDUE_COLOR_DIST = 0.16
    BORDER_RESIDUE_MIN_AREA_RATIO = 0.0025

    def __init__(self) -> None:
        self.settings = get_settings()

    @staticmethod
    def _prepare_pymatting_rgb(rgb: np.ndarray) -> np.ndarray:
        rgb_prepared = np.ascontiguousarray(np.clip(rgb, 0.0, 1.0).astype(np.float64, copy=False))
        if rgb_prepared.ndim != 3 or rgb_prepared.shape[2] != 3:
            raise ValueError(f'Invalid rgb shape for pymatting: {rgb_prepared.shape}')
        return rgb_prepared

    @staticmethod
    def _prepare_pymatting_alpha(alpha_like: np.ndarray) -> np.ndarray:
        alpha_prepared = np.ascontiguousarray(np.clip(alpha_like, 0.0, 1.0).astype(np.float64, copy=False))
        if alpha_prepared.ndim != 2:
            raise ValueError(f'Invalid alpha/trimap shape for pymatting: {alpha_prepared.shape}')
        return alpha_prepared

    def _cv2(self):
        try:
            import cv2

            return cv2
        except Exception:
            return None

    def _estimate_alpha_cf(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        try:
            from pymatting import estimate_alpha_cf

            rgb_prepared = self._prepare_pymatting_rgb(rgb)
            trimap_prepared = self._prepare_pymatting_alpha(trimap)
            alpha_refined = estimate_alpha_cf(rgb_prepared, trimap_prepared)
            logger.info('pymatting alpha refine active')
            return np.clip(alpha_refined.astype(np.float32), 0.0, 1.0)
        except Exception as exc:
            logger.warning('pymatting unavailable/fail, fallback to original alpha: %s', exc)
            # 回退时 unknown 区域使用 trimap 中值，保证流程稳定。
            return np.where(trimap >= 0.99, 1.0, np.where(trimap <= 0.01, 0.0, 0.5)).astype(np.float32)

    def _estimate_foreground_ml(self, source_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray | None:
        try:
            from pymatting import estimate_foreground_ml

            rgb_prepared = self._prepare_pymatting_rgb(source_rgb)
            alpha_prepared = self._prepare_pymatting_alpha(alpha)
            foreground = estimate_foreground_ml(rgb_prepared, alpha_prepared)
            logger.info('pymatting foreground estimation active')
            return np.clip(foreground.astype(np.float32), 0.0, 1.0)
        except Exception as exc:
            logger.warning('pymatting foreground estimate unavailable/fail: %s', exc)
            return None

    def _guided_refine_alpha(self, source_rgb: np.ndarray, alpha_refined: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.settings.enable_guided_edge_refinement:
            logger.warning('guided edge refinement disabled by rollback switch ENABLE_GUIDED_EDGE_REFINEMENT=false')
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

    def _detect_hair_internal_holes(self, source_rgb: np.ndarray, alpha_refined: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha_refined.astype(np.float32), 0.0, 1.0)
        fg_core = alpha > 0.72
        if np.count_nonzero(fg_core) < 50:
            return np.zeros_like(alpha, dtype=np.float32)

        filled_fg = morphology.remove_small_holes(fg_core, area_threshold=300)
        hole_candidates = filled_fg & (~fg_core)
        if not np.any(hole_candidates):
            return np.zeros_like(alpha, dtype=np.float32)

        coords = np.argwhere(fg_core)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        h = max(1, y1 - y0 + 1)
        max_hole_area = max(self.HAIR_HOLE_MIN_AREA, int(alpha.size * self.HAIR_HOLE_MAX_AREA_RATIO))
        labels = measure.label(hole_candidates, connectivity=2)
        selected = np.zeros_like(alpha, dtype=np.float32)

        for region in measure.regionprops(labels):
            if region.area < self.HAIR_HOLE_MIN_AREA or region.area > max_hole_area:
                continue
            cy, cx = region.centroid
            if cy > y0 + h * 0.72:
                continue
            comp = labels == region.label
            if not np.any(comp):
                continue
            mean_luma = float(np.mean(source_rgb[comp]))
            if mean_luma < 0.68:
                continue

            ring = morphology.binary_dilation(comp, morphology.disk(2)) & (~comp)
            ring_luma = float(np.mean(source_rgb[ring])) if np.any(ring) else 1.0
            ring_fg_ratio = float(np.mean(alpha[ring] > 0.62)) if np.any(ring) else 0.0
            if ring_luma > 0.58 or ring_fg_ratio < 0.50:
                continue
            selected[comp] = 1.0

        return selected

    def _fill_hair_gap_background(self, alpha_refined: np.ndarray, hair_holes_mask: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha_refined.astype(np.float32), 0.0, 1.0).copy()
        holes = hair_holes_mask > 0.5
        if np.any(holes):
            alpha[holes] = 0.0
        return alpha

    def _detect_border_background_residue(
        self,
        source_rgb: np.ndarray,
        fg_rgb: np.ndarray,
        alpha_refined: np.ndarray,
    ) -> np.ndarray:
        h, w = alpha_refined.shape
        alpha = np.clip(alpha_refined.astype(np.float32), 0.0, 1.0)
        border_strip = max(3, min(h, w) // 18)
        border_samples = np.concatenate(
            [
                source_rgb[:border_strip, :, :].reshape(-1, 3),
                source_rgb[-border_strip:, :, :].reshape(-1, 3),
                source_rgb[:, :border_strip, :].reshape(-1, 3),
                source_rgb[:, -border_strip:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg_color = np.median(border_samples, axis=0)
        color_dist = np.linalg.norm(fg_rgb - bg_color[None, None, :], axis=2)
        candidate = color_dist < self.BORDER_RESIDUE_COLOR_DIST

        labels = measure.label(candidate, connectivity=2)
        min_area = max(18, int(h * w * self.BORDER_RESIDUE_MIN_AREA_RATIO))
        residue = np.zeros((h, w), dtype=np.float32)
        for region in measure.regionprops(labels):
            if region.area < min_area:
                continue
            minr, minc, maxr, maxc = region.bbox
            touches_border = minr == 0 or minc == 0 or maxr == h or maxc == w
            if not touches_border:
                continue
            comp = labels == region.label
            mean_alpha = float(np.mean(alpha[comp]))
            if mean_alpha > 0.985:
                continue
            residue[comp] = 1.0
        return residue

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

        core_mask = (alpha >= self.CORE_FOREGROUND_ALPHA).astype(np.float32)
        if np.any(core_mask):
            propagated_core = np.zeros_like(estimated_fg, dtype=np.float32)
            core_weight = filters.gaussian(core_mask, sigma=5.0, preserve_range=True).astype(np.float32)
            core_weight = np.clip(core_weight, 1e-4, None)
            for channel in range(3):
                weighted_channel = filters.gaussian(
                    fg_rgb[:, :, channel] * core_mask,
                    sigma=5.0,
                    preserve_range=True,
                ).astype(np.float32)
                propagated_core[:, :, channel] = weighted_channel / core_weight
            propagated_core = np.clip(propagated_core, 0.0, 1.0)
            edge_propagation_strength = (edge * (1.0 - alpha) * self.EDGE_PROPAGATION_BLEND)[..., None]
            estimated_fg = estimated_fg * (1.0 - edge_propagation_strength) + propagated_core * edge_propagation_strength

        bg_mask = alpha <= 0.03
        if np.any(bg_mask):
            bg_color = np.mean(source_rgb[bg_mask], axis=0)
            bg_norm = float(np.linalg.norm(bg_color))
            if bg_norm > 1e-6:
                bg_dir = bg_color / bg_norm
                flat_estimated = estimated_fg.reshape(-1, 3)
                projection = np.maximum(0.0, flat_estimated @ bg_dir)
                remove_strength = (edge * (1.0 - alpha) * 0.30).reshape(-1, 1)
                flat_estimated = np.clip(flat_estimated - remove_strength * projection[:, None] * bg_dir[None, :], 0.0, 1.0)
                estimated_fg = flat_estimated.reshape(estimated_fg.shape)

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

        fg_rgb = fg_rgba_np[:, :, :3].astype(np.float32) / 255.0
        border_residue_mask = self._detect_border_background_residue(
            source_rgb=rgb,
            fg_rgb=fg_rgb,
            alpha_refined=alpha_refined,
        )
        alpha_refined_border_cleaned = alpha_refined.copy()
        if np.any(border_residue_mask > 0.5):
            alpha_refined_border_cleaned[border_residue_mask > 0.5] = 0.0

        hair_holes_mask = self._detect_hair_internal_holes(source_rgb=rgb, alpha_refined=alpha_refined_border_cleaned)
        alpha_refined_filled = self._fill_hair_gap_background(alpha_refined=alpha_refined_border_cleaned, hair_holes_mask=hair_holes_mask)
        alpha_filled_u8 = (alpha_refined_filled * 255.0).clip(0, 255).astype(np.uint8)

        refined_rgba = fg_rgba_np.copy()
        refined_rgba[:, :, 3] = alpha_filled_u8
        edge_band_mask = self._build_edge_band_mask(alpha_u8=alpha_filled_u8, trimap=trimap)
        decontaminated_rgba = None
        if self.settings.enable_foreground_decontamination:
            decontaminated_rgba = self._decontaminate_foreground_rgb(
                source_rgb=rgb,
                fg_rgba=refined_rgba,
                alpha_refined=alpha_refined_filled,
                edge_band_mask=edge_band_mask,
            )
        else:
            logger.warning('foreground decontamination disabled by rollback switch ENABLE_FOREGROUND_DECONTAMINATION=false')

        return MatteRefineResult(
            rgba=Image.fromarray(refined_rgba, mode='RGBA'),
            alpha=Image.fromarray(alpha_filled_u8, mode='L'),
            trimap=Image.fromarray((trimap * 255.0).clip(0, 255).astype(np.uint8), mode='L'),
            decontaminated_rgba=Image.fromarray(decontaminated_rgba, mode='RGBA') if decontaminated_rgba is not None else None,
            edge_band_mask=Image.fromarray((edge_band_mask * 255.0).astype(np.uint8), mode='L'),
            guided_alpha=(
                Image.fromarray((guided_alpha_debug * 255.0).clip(0, 255).astype(np.uint8), mode='L')
                if guided_alpha_debug is not None
                else None
            ),
            hair_internal_holes_mask=Image.fromarray((hair_holes_mask * 255.0).astype(np.uint8), mode='L'),
            hair_gap_filled_alpha=Image.fromarray(alpha_filled_u8, mode='L'),
            border_residue_mask=Image.fromarray((border_residue_mask * 255.0).astype(np.uint8), mode='L'),
        )
