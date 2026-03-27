from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from app.services.photo_generation_engine import EngineInput, EngineResult, PhotoGenerationEngine


class HivisionPhotoGenerationEngine(PhotoGenerationEngine):
    name = 'hivision'

    def __init__(self, *, segmenter, matte_refiner, cropper, background, enhancer, preview_builder, settings, logger) -> None:
        self.segmenter = segmenter
        self.matte_refiner = matte_refiner
        self.cropper = cropper
        self.background = background
        self.enhancer = enhancer
        self.preview_builder = preview_builder
        self.settings = settings
        self.logger = logger

    def _try_import_hivision_backend(self):
        repo_path = (self.settings.hivision_repo_path or '').strip()
        if repo_path:
            repo = Path(repo_path)
            if repo.exists() and str(repo) not in sys.path:
                sys.path.insert(0, str(repo))

        for module_name in ('hivision', 'hivisionidphotos'):
            try:
                return importlib.import_module(module_name)
            except Exception:
                continue
        return None

    @staticmethod
    def _close_hair_holes(alpha: Image.Image) -> Image.Image:
        alpha_arr = np.asarray(alpha.convert('L'), dtype=np.uint8)
        fg = alpha_arr > 25
        try:
            from skimage import morphology

            fg = morphology.binary_closing(fg, morphology.disk(1))
            fg = morphology.remove_small_holes(fg, area_threshold=36)
            fg = morphology.remove_small_objects(fg, min_size=32)
        except Exception:
            pass
        return Image.fromarray((fg.astype(np.uint8) * 255), mode='L')

    def _compat_generate(self, payload: EngineInput) -> EngineResult:
        rgba_foreground = self.segmenter.remove_background(payload.source_image)
        refined = self.matte_refiner.refine(payload.source_image, rgba_foreground)
        alpha = self._close_hair_holes(refined.alpha)

        base_rgba = refined.decontaminated_rgba or refined.rgba
        rgba = base_rgba.convert('RGBA')
        rgba.putalpha(alpha)

        cropped_rgba = self.cropper.crop(rgba, payload.spec, payload.face_box)
        hd_image = self.background.apply_edge_aware(cropped_rgba, payload.background_color)
        if payload.enhance:
            hd_image = self.enhancer.enhance(hd_image)
        preview_image, preview_quality = self.preview_builder(hd_image)

        return EngineResult(
            engine_name=self.name,
            hd_image=hd_image,
            preview_image=preview_image,
            preview_quality=preview_quality,
            foreground_rgba=cropped_rgba,
            debug_images={
                'hivision_hd.png': hd_image,
                'hivision_preview.jpg': preview_image,
                'hivision_mask.png': alpha,
            },
            debug_info={
                'backend': 'compat',
                'reason': 'hivision backend unavailable; using compatibility path',
            },
        )

    def generate(self, payload: EngineInput) -> EngineResult:
        backend = self._try_import_hivision_backend()
        if backend is None:
            self.logger.warning('Hivision backend not importable, fallback to compatibility implementation')
            return self._compat_generate(payload)

        if hasattr(backend, 'generate_id_photo'):
            try:
                out = backend.generate_id_photo(
                    image=payload.source_image,
                    size=(payload.spec.width_px, payload.spec.height_px),
                    bg_color=payload.background_color,
                )
                hd_image = out if isinstance(out, Image.Image) else Image.open(out).convert('RGB')
                preview_image, preview_quality = self.preview_builder(hd_image)
                rgba = hd_image.convert('RGBA')
                return EngineResult(
                    engine_name=self.name,
                    hd_image=hd_image,
                    preview_image=preview_image,
                    preview_quality=preview_quality,
                    foreground_rgba=rgba,
                    debug_images={
                        'hivision_hd.png': hd_image,
                        'hivision_preview.jpg': preview_image,
                    },
                    debug_info={'backend': 'native'},
                )
            except Exception as exc:
                self.logger.warning('Native hivision backend failed: %s; fallback to compatibility mode', exc)

        return self._compat_generate(payload)
