from PIL import Image

from services.print_service import PrintService


class GeneratePrintLayoutPipeline:
    def __init__(self):
        self.print_service = PrintService()

    def run(self, image_id: str, hd_image_path: str, layout_type: str) -> dict:
        hd_image = Image.open(hd_image_path).convert("RGB")
        print_url = self.print_service.generate_layout(image_id=image_id, hd_image=hd_image, layout_type=layout_type)
        return {
            "imageId": image_id,
            "layoutType": layout_type,
            "printUrl": print_url,
        }
