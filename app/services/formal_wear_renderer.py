from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter

from app.services.formal_wear_geometry import FormalWearGeometry, estimate_formal_wear_geometry


@dataclass(frozen=True)
class FormalWearPalette:
    suit: tuple[int, int, int]
    suit_shadow: tuple[int, int, int]
    shirt: tuple[int, int, int]
    tie: tuple[int, int, int]
    accent: tuple[int, int, int]


class FormalWearRenderer:
    def render(
        self,
        foreground_rgba: Image.Image,
        *,
        face_box: dict[str, int] | None,
        gender: str | None,
        style: str,
        color: str,
    ) -> tuple[Image.Image, list[str]]:
        canvas = foreground_rgba.convert('RGBA')
        geometry = estimate_formal_wear_geometry(canvas.width, canvas.height, face_box, gender, style)
        palette = self._palette(color)

        clothing_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        self._draw_body_base(clothing_layer, geometry, palette, style)
        if gender == 'female':
            self._draw_female_neckline(clothing_layer, geometry, palette, style)
        else:
            self._draw_male_shirt_and_tie(clothing_layer, geometry, palette, style)
        self._draw_lapels(clothing_layer, geometry, palette, gender, style)
        softened_clothing = clothing_layer.filter(ImageFilter.GaussianBlur(radius=0.7))
        composed = Image.alpha_composite(softened_clothing, canvas)

        warnings = [
            f'Applied lightweight formal-wear overlay gender={gender or "male"} style={style} color={color}'
        ]
        return composed, warnings

    def _palette(self, color: str) -> FormalWearPalette:
        normalized = (color or 'black').strip().lower()
        mapping = {
            'black': FormalWearPalette((39, 43, 50), (18, 21, 28), (246, 248, 252), (43, 49, 68), (112, 124, 150)),
            'navy': FormalWearPalette((41, 58, 96), (23, 34, 62), (247, 248, 252), (25, 36, 78), (122, 143, 188)),
            'gray': FormalWearPalette((104, 108, 118), (73, 76, 86), (245, 246, 249), (90, 76, 104), (154, 157, 164)),
        }
        return mapping.get(normalized, mapping['black'])

    def _draw_body_base(
        self,
        layer: Image.Image,
        geometry: FormalWearGeometry,
        palette: FormalWearPalette,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        shoulder_drop = 8 if style == 'simple' else 0
        draw.polygon(
            [
                (geometry.shoulder_left_x, geometry.shoulder_y + shoulder_drop),
                (geometry.neck_left_x - 14, geometry.chest_y),
                (geometry.head_center_x - 58, geometry.waist_y),
                (geometry.head_center_x - 44, geometry.torso_bottom_y),
                (geometry.shoulder_left_x - 16, geometry.torso_bottom_y),
            ],
            fill=palette.suit + (235,),
        )
        draw.polygon(
            [
                (geometry.shoulder_right_x, geometry.shoulder_y + shoulder_drop),
                (geometry.neck_right_x + 14, geometry.chest_y),
                (geometry.head_center_x + 58, geometry.waist_y),
                (geometry.head_center_x + 44, geometry.torso_bottom_y),
                (geometry.shoulder_right_x + 16, geometry.torso_bottom_y),
            ],
            fill=palette.suit + (235,),
        )
        draw.ellipse(
            [
                geometry.head_center_x - (geometry.shoulder_right_x - geometry.shoulder_left_x) * 0.42,
                geometry.shoulder_y - 12,
                geometry.head_center_x + (geometry.shoulder_right_x - geometry.shoulder_left_x) * 0.42,
                geometry.waist_y + 44,
            ],
            fill=palette.suit_shadow + (68,),
        )

    def _draw_male_shirt_and_tie(
        self,
        layer: Image.Image,
        geometry: FormalWearGeometry,
        palette: FormalWearPalette,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        shirt_width = (geometry.neck_right_x - geometry.neck_left_x) * 1.22
        shirt_bottom = geometry.waist_y + (24 if style == 'business' else 8)
        draw.polygon(
            [
                (geometry.head_center_x, geometry.chin_y - 3),
                (geometry.head_center_x - shirt_width * 0.52, geometry.chest_y),
                (geometry.head_center_x - shirt_width * 0.22, shirt_bottom),
                (geometry.head_center_x + shirt_width * 0.22, shirt_bottom),
                (geometry.head_center_x + shirt_width * 0.52, geometry.chest_y),
            ],
            fill=palette.shirt + (246,),
        )

        tie_top = geometry.chin_y + 10
        tie_width = 18 if style == 'simple' else 24
        tie_bottom = geometry.waist_y + (18 if style == 'business' else -4)
        draw.polygon(
            [
                (geometry.head_center_x, tie_top),
                (geometry.head_center_x - tie_width, geometry.chest_y + 28),
                (geometry.head_center_x - tie_width * 0.66, tie_bottom - 18),
                (geometry.head_center_x, tie_bottom),
                (geometry.head_center_x + tie_width * 0.66, tie_bottom - 18),
                (geometry.head_center_x + tie_width, geometry.chest_y + 28),
            ],
            fill=palette.tie + (220,),
        )
        draw.polygon(
            [
                (geometry.head_center_x, tie_top - 8),
                (geometry.head_center_x - tie_width * 0.86, tie_top + 12),
                (geometry.head_center_x, tie_top + 28),
                (geometry.head_center_x + tie_width * 0.86, tie_top + 12),
            ],
            fill=palette.tie + (236,),
        )

    def _draw_female_neckline(
        self,
        layer: Image.Image,
        geometry: FormalWearGeometry,
        palette: FormalWearPalette,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        blouse_width = (geometry.neck_right_x - geometry.neck_left_x) * (1.95 if style == 'business' else 1.75)
        neckline_depth = 44 if style == 'simple' else 64
        draw.rounded_rectangle(
            [
                geometry.head_center_x - blouse_width / 2,
                geometry.chin_y + 10,
                geometry.head_center_x + blouse_width / 2,
                geometry.waist_y + 16,
            ],
            radius=28,
            fill=palette.shirt + (244,),
        )
        draw.polygon(
            [
                (geometry.head_center_x - blouse_width * 0.24, geometry.chin_y + 10),
                (geometry.head_center_x, geometry.chin_y + neckline_depth),
                (geometry.head_center_x + blouse_width * 0.24, geometry.chin_y + 10),
            ],
            fill=(0, 0, 0, 0),
        )
        draw.ellipse(
            [
                geometry.head_center_x - blouse_width * 0.18,
                geometry.chin_y + 16,
                geometry.head_center_x + blouse_width * 0.18,
                geometry.chin_y + neckline_depth + 10,
            ],
            fill=(0, 0, 0, 0),
        )

    def _draw_lapels(
        self,
        layer: Image.Image,
        geometry: FormalWearGeometry,
        palette: FormalWearPalette,
        gender: str | None,
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        lapel_alpha = 232 if style == 'business' else 214
        lapel_top = geometry.chin_y + (6 if gender == 'female' else 2)
        lapel_bottom = geometry.chest_y + (42 if style == 'business' else 26)
        inner_spread = 14 if style == 'simple' else 22
        outer_spread = 54 if gender == 'female' else 62

        draw.polygon(
            [
                (geometry.neck_left_x - 4, lapel_top),
                (geometry.head_center_x - inner_spread, lapel_bottom),
                (geometry.head_center_x - outer_spread, lapel_top + 26),
                (geometry.neck_left_x - 18, lapel_top - 3),
            ],
            fill=palette.suit_shadow + (lapel_alpha,),
        )
        draw.polygon(
            [
                (geometry.neck_right_x + 4, lapel_top),
                (geometry.head_center_x + inner_spread, lapel_bottom),
                (geometry.head_center_x + outer_spread, lapel_top + 26),
                (geometry.neck_right_x + 18, lapel_top - 3),
            ],
            fill=palette.suit_shadow + (lapel_alpha,),
        )
        draw.line(
            [
                (geometry.head_center_x - inner_spread, lapel_bottom),
                (geometry.head_center_x - outer_spread + 8, lapel_top + 28),
            ],
            fill=palette.accent + (186,),
            width=2,
        )
        draw.line(
            [
                (geometry.head_center_x + inner_spread, lapel_bottom),
                (geometry.head_center_x + outer_spread - 8, lapel_top + 28),
            ],
            fill=palette.accent + (186,),
            width=2,
        )
