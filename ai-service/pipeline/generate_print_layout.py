from PIL import Image, UnidentifiedImageError

from core.exceptions import AppException, ERROR_FILE_NOT_FOUND, ERROR_INVALID_IMAGE
from services.print_service import PrintService


class GeneratePrintLayoutPipeline:
    def __init__(self):
        self.print_service = PrintService()

    def run(self, image_id: str, hd_image_path: str, layout_type: str) -> dict:
        try:
            hd_image = Image.open(hd_image_path).convert('RGB')
        except FileNotFoundError as exc:
            raise AppException(f'HD image not found: {hd_image_path}', ERROR_FILE_NOT_FOUND, 404) from exc
        except UnidentifiedImageError as exc:
            raise AppException('Invalid HD image content', ERROR_INVALID_IMAGE, 400) from exc
        print_url = self.print_service.generate_layout(image_id=image_id, hd_image=hd_image, layout_type=layout_type)
        return {
            'imageId': image_id,
            'layoutType': layout_type,
            'printUrl': print_url,
        }
