from __future__ import annotations

from PIL import Image

from core.exceptions import AppException
from services.detect_service import DetectService
from services.print_service import PrintService
from services.validation_service import ValidationService


class GeneratePrintLayoutPipeline:
    def __init__(self):
        self.detect_service = DetectService()
        self.validation_service = ValidationService()
        self.print_service = PrintService()

    def run(self, image_id: str, hd_image_path: str, layout_type: str) -> dict:
        detect_result = self.detect_service.detect(image_id=image_id, image_path=hd_image_path)
        if hasattr(detect_result, 'passed') and not detect_result.passed:
            error_code, message = self.validation_service.build_generate_error(list(getattr(detect_result, 'reasons', [])))
            raise AppException(message, error_code, 400)

        hd_image = Image.open(hd_image_path).convert('RGB')
        print_url = self.print_service.generate_layout(image_id=image_id, hd_image=hd_image, layout_type=layout_type)
        return {
            'imageId': image_id,
            'layoutType': layout_type,
            'printUrl': print_url,
        }
