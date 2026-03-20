from __future__ import annotations

from math import ceil

from PIL import Image, ImageOps

from constants.status import LAYOUT_TYPES
from utils.file_utils import build_output_path, public_url_for_path, to_url_like_path
from utils.image_utils import save_pil_image
from utils.logger import get_logger


class PrintService:
    paper_size = (1800, 1200)
    margin = 60
    spacing = 25

    def __init__(self) -> None:
        self.logger = get_logger(component='print_service')

    def _grid(self, count: int) -> tuple[int, int]:
        cols = ceil(count**0.5)
        rows = ceil(count / cols)
        return rows, cols

    def generate_layout(self, image_id: str, hd_image: Image.Image, layout_type: str) -> dict:
        print_logger = self.logger.bind(image_id=image_id, layout_type=layout_type)
        if layout_type not in LAYOUT_TYPES:
            print_logger.warning('unsupported print layout requested')
            raise ValueError(f'Unsupported layout type: {layout_type}')

        count = LAYOUT_TYPES[layout_type]
        rows, cols = self._grid(count)
        print_logger.bind(photo_count=count, rows=rows, cols=cols).info('generating print layout')
        canvas = Image.new('RGB', self.paper_size, (255, 255, 255))

        max_w = (self.paper_size[0] - self.margin * 2 - self.spacing * (cols - 1)) // cols
        max_h = (self.paper_size[1] - self.margin * 2 - self.spacing * (rows - 1)) // rows

        tile = ImageOps.contain(hd_image, (max_w, max_h), Image.Resampling.LANCZOS)
        for index in range(count):
            row = index // cols
            col = index % cols
            x = self.margin + col * (max_w + self.spacing) + (max_w - tile.size[0]) // 2
            y = self.margin + row * (max_h + self.spacing) + (max_h - tile.size[1]) // 2
            canvas.paste(tile, (x, y))

        suffix = {'six': '6', 'eight': '8', 'twelve': '12'}[layout_type]
        output_path = build_output_path('print', f'{image_id}_print_{suffix}.jpg')
        save_pil_image(canvas, output_path, quality=95)
        print_logger.bind(output_path=output_path).info('print layout generated successfully')
        return {
            'printPath': to_url_like_path(output_path),
            'printUrl': public_url_for_path(output_path),
            'paperType': '6inch',
            'photoCount': count,
            'layoutType': layout_type,
        }
