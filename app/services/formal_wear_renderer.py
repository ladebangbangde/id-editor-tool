from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

from app.services.formal_wear_geometry import FormalWearAnchors


@dataclass(frozen=True)
class RenderPalette:
    jacket: tuple[int, int, int, int]
    jacket_shadow: tuple[int, int, int, int]
    lapel: tuple[int, int, int, int]
    shirt: tuple[int, int, int, int]
    shirt_shadow: tuple[int, int, int, int]
    tie: tuple[int, int, int, int]
    highlight: tuple[int, int, int, int]


class FormalWearRenderer:
    PALETTES: dict[str, RenderPalette] = {
        'black': RenderPalette(
            jacket=(34, 38, 46, 255),
            jacket_shadow=(15, 18, 24, 160),
            lapel=(49, 54, 64, 230),
            shirt=(240, 243, 247, 245),
            shirt_shadow=(202, 208, 217, 115),
            tie=(42, 49, 64, 235),
            highlight=(255, 255, 255, 26),
        ),
        'navy': RenderPalette(
            jacket=(35, 54, 88, 255),
            jacket_shadow=(18, 28, 49, 165),
            lapel=(44, 67, 108, 230),
            shirt=(241, 244, 248, 245),
            shirt_shadow=(198, 207, 220, 120),
            tie=(58, 78, 126, 235),
            highlight=(255, 255, 255, 28),
        ),
        'gray': RenderPalette(
            jacket=(85, 90, 99, 255),
            jacket_shadow=(51, 55, 63, 150),
            lapel=(104, 110, 121, 228),
            shirt=(241, 243, 245, 245),
            shirt_shadow=(205, 211, 218, 120),
            tie=(86, 94, 112, 230),
            highlight=(255, 255, 255, 30),
        ),
    }

    def render(
        self,
        image_size: tuple[int, int],
        anchors: FormalWearAnchors,
        gender: str,
        style: str,
        color: str,
    ) -> Image.Image:
        canvas = Image.new('RGBA', image_size, (0, 0, 0, 0))
        palette = self.PALETTES[color]

        self._draw_base_jacket(canvas, anchors, palette, style)
        self._draw_shirt(canvas, anchors, palette, gender, style)
        self._draw_lapels(canvas, anchors, palette, gender, style)
        if gender == 'male' and style in {'standard', 'business'}:
            self._draw_tie(canvas, anchors, palette, style)
        if gender == 'female':
            self._draw_female_opening(canvas, anchors, palette, style)
        self._draw_depth(canvas, anchors, palette)
        return canvas.filter(ImageFilter.GaussianBlur(radius=0.6))

    def _draw_base_jacket(
        self,
        canvas: Image.Image,
        anchors: FormalWearAnchors,
        palette: RenderPalette,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(canvas, 'RGBA')
        width, height = canvas.size
        hem_y = min(height - 1, int(anchors.chest_bottom_y + (anchors.face_box['height'] * 0.22)))
        waist_inset = anchors.face_box['width'] * (0.20 if style == 'simple' else 0.12)
        jacket = [
            (max(0, anchors.left_shoulder_x - anchors.face_box['width'] * 0.08), anchors.shoulder_y),
            (max(0, anchors.left_shoulder_x - anchors.face_box['width'] * 0.22), hem_y),
            (max(0, anchors.neck_center_x - anchors.face_box['width'] * 0.55 + waist_inset), hem_y),
            (anchors.neck_center_x, anchors.chest_top_y + anchors.face_box['height'] * 0.12),
            (min(width - 1, anchors.neck_center_x + anchors.face_box['width'] * 0.55 - waist_inset), hem_y),
            (min(width - 1, anchors.right_shoulder_x + anchors.face_box['width'] * 0.22), hem_y),
            (min(width - 1, anchors.right_shoulder_x + anchors.face_box['width'] * 0.08), anchors.shoulder_y),
            (anchors.neck_center_x + anchors.neck_width * 0.64, anchors.jacket_top_y),
            (anchors.neck_center_x - anchors.neck_width * 0.64, anchors.jacket_top_y),
        ]
        draw.polygon(jacket, fill=palette.jacket)

        shoulder_shadow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shoulder_shadow, 'RGBA')
        sdraw.rounded_rectangle(
            [
                anchors.left_shoulder_x - anchors.face_box['width'] * 0.35,
                anchors.shoulder_y - anchors.face_box['height'] * 0.04,
                anchors.right_shoulder_x + anchors.face_box['width'] * 0.35,
                anchors.shoulder_y + anchors.face_box['height'] * 0.22,
            ],
            radius=max(8, int(anchors.face_box['width'] * 0.12)),
            fill=palette.jacket_shadow,
        )
        canvas.alpha_composite(shoulder_shadow.filter(ImageFilter.GaussianBlur(radius=14)))

    def _draw_shirt(
        self,
        canvas: Image.Image,
        anchors: FormalWearAnchors,
        palette: RenderPalette,
        gender: str,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(canvas, 'RGBA')
        shirt_width = anchors.face_box['width'] * (0.44 if gender == 'male' else 0.54)
        shirt_bottom = anchors.chest_bottom_y - anchors.face_box['height'] * (0.08 if style == 'simple' else 0.03)
        draw.polygon(
            [
                (anchors.neck_center_x - shirt_width * 0.28, anchors.neck_bottom_y - anchors.face_box['height'] * 0.02),
                (anchors.neck_center_x - shirt_width * 0.48, shirt_bottom),
                (anchors.neck_center_x + shirt_width * 0.48, shirt_bottom),
                (anchors.neck_center_x + shirt_width * 0.28, anchors.neck_bottom_y - anchors.face_box['height'] * 0.02),
            ],
            fill=palette.shirt,
        )

        shirt_shadow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shirt_shadow, 'RGBA')
        sdraw.ellipse(
            [
                anchors.neck_center_x - shirt_width * 0.62,
                anchors.neck_top_y,
                anchors.neck_center_x + shirt_width * 0.62,
                anchors.neck_bottom_y + anchors.face_box['height'] * 0.25,
            ],
            fill=palette.shirt_shadow,
        )
        canvas.alpha_composite(shirt_shadow.filter(ImageFilter.GaussianBlur(radius=7)))

    def _draw_lapels(
        self,
        canvas: Image.Image,
        anchors: FormalWearAnchors,
        palette: RenderPalette,
        gender: str,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(canvas, 'RGBA')
        lapel_drop = anchors.face_box['height'] * (0.56 if style == 'business' else 0.48)
        inner_gap = anchors.lapel_inner_gap * (1.08 if gender == 'female' else 1.0)
        outer = anchors.lapel_outer_span

        left_lapel = [
            (anchors.neck_center_x - inner_gap, anchors.jacket_top_y),
            (anchors.neck_center_x - outer, anchors.chest_top_y + anchors.face_box['height'] * 0.22),
            (anchors.neck_center_x - anchors.face_box['width'] * 0.16, anchors.chest_top_y + lapel_drop),
            (anchors.neck_center_x - anchors.face_box['width'] * 0.02, anchors.neck_bottom_y + anchors.face_box['height'] * 0.05),
        ]
        right_lapel = [
            (anchors.neck_center_x + inner_gap, anchors.jacket_top_y),
            (anchors.neck_center_x + outer, anchors.chest_top_y + anchors.face_box['height'] * 0.22),
            (anchors.neck_center_x + anchors.face_box['width'] * 0.16, anchors.chest_top_y + lapel_drop),
            (anchors.neck_center_x + anchors.face_box['width'] * 0.02, anchors.neck_bottom_y + anchors.face_box['height'] * 0.05),
        ]
        draw.polygon(left_lapel, fill=palette.lapel)
        draw.polygon(right_lapel, fill=palette.lapel)

        highlight = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        hdraw = ImageDraw.Draw(highlight, 'RGBA')
        hdraw.line(left_lapel[:3], fill=palette.highlight, width=max(2, int(anchors.face_box['width'] * 0.03)))
        hdraw.line(right_lapel[:3], fill=palette.highlight, width=max(2, int(anchors.face_box['width'] * 0.03)))
        canvas.alpha_composite(highlight.filter(ImageFilter.GaussianBlur(radius=2)))

    def _draw_tie(
        self,
        canvas: Image.Image,
        anchors: FormalWearAnchors,
        palette: RenderPalette,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(canvas, 'RGBA')
        knot_half = anchors.face_box['width'] * (0.09 if style == 'business' else 0.08)
        tie_half = anchors.face_box['width'] * (0.08 if style == 'business' else 0.06)
        knot = [
            (anchors.neck_center_x, anchors.tie_top_y),
            (anchors.neck_center_x - knot_half, anchors.tie_top_y + anchors.face_box['height'] * 0.08),
            (anchors.neck_center_x, anchors.tie_top_y + anchors.face_box['height'] * 0.16),
            (anchors.neck_center_x + knot_half, anchors.tie_top_y + anchors.face_box['height'] * 0.08),
        ]
        blade = [
            (anchors.neck_center_x - tie_half, anchors.tie_top_y + anchors.face_box['height'] * 0.14),
            (anchors.neck_center_x - tie_half * 0.62, anchors.tie_bottom_y),
            (anchors.neck_center_x, anchors.tie_bottom_y + anchors.face_box['height'] * 0.12),
            (anchors.neck_center_x + tie_half * 0.62, anchors.tie_bottom_y),
            (anchors.neck_center_x + tie_half, anchors.tie_top_y + anchors.face_box['height'] * 0.14),
        ]
        draw.polygon(knot, fill=palette.tie)
        draw.polygon(blade, fill=palette.tie)

    def _draw_female_opening(
        self,
        canvas: Image.Image,
        anchors: FormalWearAnchors,
        palette: RenderPalette,
        style: str,
    ) -> None:
        overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        mask = Image.new('L', canvas.size, 0)
        draw = ImageDraw.Draw(mask)
        scoop_depth = anchors.face_box['height'] * (0.34 if style == 'simple' else 0.28)
        draw.ellipse(
            [
                anchors.neck_center_x - anchors.face_box['width'] * 0.40,
                anchors.neck_top_y + anchors.face_box['height'] * 0.02,
                anchors.neck_center_x + anchors.face_box['width'] * 0.40,
                anchors.neck_bottom_y + scoop_depth,
            ],
            fill=220,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=6))
        overlay.paste(Image.new('RGBA', canvas.size, palette.shirt), (0, 0), mask)
        canvas.alpha_composite(overlay)

    def _draw_depth(self, canvas: Image.Image, anchors: FormalWearAnchors, _palette: RenderPalette) -> None:
        depth = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(depth, 'RGBA')
        draw.ellipse(
            [
                anchors.neck_center_x - anchors.face_box['width'] * 0.52,
                anchors.neck_top_y,
                anchors.neck_center_x + anchors.face_box['width'] * 0.52,
                anchors.neck_bottom_y + anchors.face_box['height'] * 0.16,
            ],
            fill=(0, 0, 0, 52),
        )
        draw.rectangle(
            [
                anchors.left_shoulder_x - anchors.face_box['width'] * 0.18,
                anchors.shoulder_y,
                anchors.right_shoulder_x + anchors.face_box['width'] * 0.18,
                anchors.chest_bottom_y + anchors.face_box['height'] * 0.10,
            ],
            fill=(255, 255, 255, 16),
        )
        canvas.alpha_composite(depth.filter(ImageFilter.GaussianBlur(radius=10)))
