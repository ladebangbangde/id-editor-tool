from math import floor

from PIL import Image, ImageOps

from app.core.exceptions import ProcessFailedError
from app.services.specs import PhotoSpec, get_layout_paper


class LayoutService:
    def build(self, photo: Image.Image, spec: PhotoSpec, paper: str = '6inch') -> tuple[Image.Image, int]:
        paper_spec = get_layout_paper(paper)
        canvas = Image.new('RGB', (paper_spec['width_px'], paper_spec['height_px']), (255, 255, 255))

        margin = paper_spec['margin_px']
        spacing = paper_spec['spacing_px']
        usable_width = paper_spec['width_px'] - margin * 2
        usable_height = paper_spec['height_px'] - margin * 2

        cols = max(1, floor((usable_width + spacing) / (spec.width_px + spacing)))
        rows = max(1, floor((usable_height + spacing) / (spec.height_px + spacing)))
        count = cols * rows
        if count == 0:
            raise ProcessFailedError('Photo size does not fit selected paper')

        tile = ImageOps.contain(photo, (spec.width_px, spec.height_px), Image.Resampling.LANCZOS)
        total_grid_width = cols * tile.width + (cols - 1) * spacing
        total_grid_height = rows * tile.height + (rows - 1) * spacing
        origin_x = (paper_spec['width_px'] - total_grid_width) // 2
        origin_y = (paper_spec['height_px'] - total_grid_height) // 2

        index = 0
        for row in range(rows):
            for col in range(cols):
                x = origin_x + col * (tile.width + spacing)
                y = origin_y + row * (tile.height + spacing)
                canvas.paste(tile, (x, y))
                index += 1
        return canvas, count
