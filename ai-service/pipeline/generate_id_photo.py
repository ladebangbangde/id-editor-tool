from pathlib import Path

from constants.photo_sizes import PHOTO_SIZE_TEMPLATES, build_custom_template
from pipeline.build_preview import PreviewBuilder
from services.background_service import BackgroundService
from services.crop_service import CropService
from services.detect_service import DetectService
from services.enhance_service import EnhanceService
from services.print_service import PrintService
from services.quality_service import QualityService
from services.segment_service import SegmentService
from utils.config import get_settings
from utils.file_utils import build_output_path, to_url_like_path
from utils.image_utils import save_pil_image


class GenerateIdPhotoPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.detect_service = DetectService()
        self.segment_service = SegmentService()
        self.background_service = BackgroundService()
        self.crop_service = CropService()
        self.enhance_service = EnhanceService()
        self.quality_service = QualityService()
        self.print_service = PrintService()
        self.preview_builder = PreviewBuilder()

    def _resolve_template(self, source_type: str, scene_key: str | None, custom_w: int | None, custom_h: int | None):
        if source_type == "scene":
            if scene_key not in PHOTO_SIZE_TEMPLATES:
                raise ValueError(f"Unknown sceneKey: {scene_key}")
            return PHOTO_SIZE_TEMPLATES[scene_key]
        return build_custom_template(custom_w, custom_h, dpi=self.settings.jpeg_dpi)

    def run(self, payload: dict) -> dict:
        image_id = payload["imageId"]
        original_image_path = payload["originalImagePath"]

        if not Path(original_image_path).exists():
            raise FileNotFoundError(f"original image not found: {original_image_path}")

        detect_result = self.detect_service.detect(image_id=image_id, image_path=original_image_path)
        if not detect_result.hasFace:
            raise ValueError("No face detected from source image")

        size_tpl = self._resolve_template(
            source_type=payload["sourceType"],
            scene_key=payload.get("sceneKey"),
            custom_w=payload.get("customWidthMm"),
            custom_h=payload.get("customHeightMm"),
        )

        seg_output_path = build_output_path("temp", f"{image_id}_segmented.png")
        self.segment_service.segment_person(original_image_path, seg_output_path)

        background_applied = self.background_service.apply_background(
            transparent_png_path=seg_output_path,
            background_color=payload["backgroundColor"],
        )

        cropped = self.crop_service.crop_to_size(background_applied, size_tpl.pixelWidth, size_tpl.pixelHeight)
        enhanced = self.enhance_service.enhance(cropped, bool(payload.get("beautyEnabled", False)))

        hd_output_path = build_output_path("hd", f"{image_id}_hd.jpg")
        save_pil_image(enhanced, hd_output_path, quality=self.settings.hd_quality, dpi=self.settings.jpeg_dpi)

        preview_url = self.preview_builder.build_preview(image_id=image_id, hd_image=enhanced)

        quality_status, _quality_message = self.quality_service.evaluate(enhanced)

        print_url = None
        layout_type = payload.get("printLayoutType")
        if layout_type:
            print_url = self.print_service.generate_layout(image_id=image_id, hd_image=enhanced, layout_type=layout_type)

        return {
            "imageId": image_id,
            "previewUrl": preview_url,
            "hdUrl": to_url_like_path(hd_output_path),
            "printUrl": print_url,
            "backgroundColor": payload["backgroundColor"],
            "widthMm": size_tpl.widthMm,
            "heightMm": size_tpl.heightMm,
            "pixelWidth": size_tpl.pixelWidth,
            "pixelHeight": size_tpl.pixelHeight,
            "qualityStatus": quality_status,
        }
