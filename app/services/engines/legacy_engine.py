from __future__ import annotations

from typing import Callable

from PIL import Image

from app.services.photo_generation_engine import EngineInput, EngineResult, PhotoGenerationEngine


class LegacyPhotoGenerationEngine(PhotoGenerationEngine):
    name = 'legacy'

    def __init__(self, *, segmenter, matte_refiner, cropper, background, enhancer, preview_builder, settings) -> None:
        self.segmenter = segmenter
        self.matte_refiner = matte_refiner
        self.cropper = cropper
        self.background = background
        self.enhancer = enhancer
        self.preview_builder = preview_builder
        self.settings = settings

    def generate(
        self,
        payload: EngineInput,
        stage_reporter: Callable[[str], None] | None = None,
    ) -> EngineResult:
        if stage_reporter is not None:
            stage_reporter('adjusting')
        rgba_foreground = self.segmenter.remove_background(payload.source_image)
        segmentation_debug = self.segmenter.consume_debug_images() if hasattr(self.segmenter, 'consume_debug_images') else {}
        alpha_seed = segmentation_debug.get('baidu_alpha_seed.png', rgba_foreground.getchannel('A'))
        trimap_seed = segmentation_debug.get('baidu_trimap_seed.png')
        refined = self.matte_refiner.refine_from_alpha_seed(
            source_image=payload.source_image,
            alpha_seed=alpha_seed,
            foreground_hint=rgba_foreground,
            face_box=payload.face_box,
            trimap_seed=trimap_seed,
        )
        effective_rgba = refined.rgba
        if self.settings.enable_decontaminated_output_as_default and refined.decontaminated_rgba is not None:
            effective_rgba = refined.decontaminated_rgba
        cropped_rgba = self.cropper.crop(effective_rgba, payload.spec, payload.face_box)
        if stage_reporter is not None:
            stage_reporter('generating')
        hd_image = self.background.apply(cropped_rgba, payload.background_color)
        if payload.enhance:
            hd_image = self.enhancer.enhance(hd_image)

        preview_image, preview_quality = self.preview_builder(hd_image)

        debug_images: dict[str, Image.Image] = {
            'foreground.png': rgba_foreground,
            'refined_alpha.png': refined.alpha,
            'trimap.png': refined.trimap,
            'refined_foreground.png': refined.rgba,
            'cropped_rgba.png': cropped_rgba,
            'legacy_hd.png': hd_image,
            'legacy_preview.jpg': preview_image,
        }
        for name, debug_image in segmentation_debug.items():
            debug_images[name] = debug_image

        if refined.decontaminated_rgba is not None:
            debug_images['foreground_decontaminated.png'] = refined.decontaminated_rgba
        if refined.guided_alpha is not None:
            debug_images['guided_alpha.png'] = refined.guided_alpha
        if refined.hair_internal_holes_mask is not None:
            debug_images['hair_internal_holes_mask.png'] = refined.hair_internal_holes_mask
            debug_images['legacy_mask.png'] = refined.hair_internal_holes_mask
        if refined.hair_gap_filled_alpha is not None:
            debug_images['hair_gap_filled_alpha.png'] = refined.hair_gap_filled_alpha
        if refined.border_residue_mask is not None:
            debug_images['border_residue_mask.png'] = refined.border_residue_mask
        if refined.edge_band_mask is not None:
            debug_images['edge_band_mask.png'] = refined.edge_band_mask
        if refined.face_protected_alpha is not None:
            debug_images['face_protected_alpha.png'] = refined.face_protected_alpha
        if refined.final_refined_alpha is not None:
            debug_images['final_refined_alpha.png'] = refined.final_refined_alpha

        return EngineResult(
            engine_name=self.name,
            hd_image=hd_image,
            preview_image=preview_image,
            preview_quality=preview_quality,
            foreground_rgba=cropped_rgba,
            debug_images=debug_images,
        )
