from pathlib import Path

from PIL import Image, UnidentifiedImageError

from core.exceptions import AppException, ERROR_FILE_NOT_FOUND, ERROR_INVALID_IMAGE
from services.detect_service import DetectService
from services.print_service import PrintService
from services.validation_service import ValidationService


class GeneratePrintLayoutPipeline:
    def __init__(self):
        self.print_service = PrintService()
        self.detect_service = DetectService()
        self.validation_service = ValidationService()

    def run(self, image_id: str, hd_image_path: str, layout_type: str) -> dict:
        normalized_path = Path(hd_image_path)
        relaxed_min_face_size = (40, 40) if normalized_path.name.endswith('_hd.jpg') else (60, 60)
        try:
            hd_image = Image.open(hd_image_path).convert('RGB')
        except FileNotFoundError as exc:
            raise AppException(f'HD image not found: {hd_image_path}', ERROR_FILE_NOT_FOUND, 404) from exc
        except UnidentifiedImageError as exc:
            raise AppException('Invalid HD image content', ERROR_INVALID_IMAGE, 400) from exc
        detect_result = self.detect_service.detect(
            image_id=image_id,
            image_path=hd_image_path,
            min_face_size=relaxed_min_face_size,
        )
        if not detect_result.passed:
            error_code, message = self.validation_service.build_generate_error(detect_result.reasons)
            raise AppException(message, error_code, 400, data=detect_result.to_dict())
        print_url = self.print_service.generate_layout(image_id=image_id, hd_image=hd_image, layout_type=layout_type)
        return {
            'imageId': image_id,
            'layoutType': layout_type,
            'printUrl': print_url,
        }
