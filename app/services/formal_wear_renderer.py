from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.services.formal_wear_geometry import FormalWearGeometry, estimate_formal_wear_geometry


@dataclass(frozen=True)
class FormalWearPalette:
    suit: tuple[int, int, int]
    suit_dark: tuple[int, int, int]
    suit_light: tuple[int, int, int]
    shirt: tuple[int, int, int]
    tie: tuple[int, int, int]


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
        geometry = estimate_formal_wear_geometry(canvas, face_box, gender, style)
        palette = self._palette(color)
        skin_tone = self._estimate_skin_tone(canvas, geometry.face_box)

        clothing_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        underlay_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        self._draw_neck_bridge(underlay_layer, geometry, skin_tone)
        if gender == 'female':
            if style == 'standard' and color == 'black':
                self._draw_female_structured_suit(clothing_layer, geometry, palette)
            else:
                self._draw_female_jacket(clothing_layer, geometry, palette, style)
                self._draw_female_inner_top(clothing_layer, geometry, palette, skin_tone, style)
        else:
            self._draw_male_jacket(clothing_layer, geometry, palette, style)
            self._draw_male_shirt_and_tie(clothing_layer, geometry, palette, style)
        if not (gender == 'female' and style == 'standard' and color == 'black'):
            self._draw_lapels(clothing_layer, geometry, palette, gender, style)
            self._draw_seam_shadows(clothing_layer, geometry, palette)

        softened_clothing = clothing_layer.filter(ImageFilter.GaussianBlur(radius=0.9))
        front_portrait = self._extract_front_portrait(canvas, geometry)
        composed = Image.alpha_composite(underlay_layer, softened_clothing)
        composed = Image.alpha_composite(composed, front_portrait)

        warnings = [
            f'Applied refined formal-wear overlay gender={gender or "male"} style={style} color={color}'
        ]
        return composed, warnings

    def _palette(self, color: str) -> FormalWearPalette:
        normalized = (color or 'black').strip().lower()
        mapping = {
            'black': FormalWearPalette((44, 47, 54), (23, 26, 32), (73, 77, 88), (247, 248, 252), (54, 62, 86)),
            'navy': FormalWearPalette((38, 56, 96), (21, 33, 66), (70, 91, 136), (248, 249, 252), (32, 45, 91)),
            'gray': FormalWearPalette((108, 112, 122), (72, 76, 88), (145, 149, 160), (247, 248, 251), (96, 82, 110)),
        }
        return mapping.get(normalized, mapping['black'])

    def _estimate_skin_tone(self, image: Image.Image, face_box: dict[str, int]) -> tuple[int, int, int]:
        rgb = np.asarray(image.convert('RGB'))
        x0 = max(face_box['x'] + face_box['width'] // 4, 0)
        x1 = min(face_box['x'] + face_box['width'] * 3 // 4, rgb.shape[1])
        y0 = max(face_box['y'] + face_box['height'] // 2, 0)
        y1 = min(face_box['y'] + face_box['height'] - face_box['height'] // 8, rgb.shape[0])
        if x1 <= x0 or y1 <= y0:
            return 220, 188, 166
        patch = rgb[y0:y1, x0:x1]
        if patch.size == 0:
            return 220, 188, 166
        median = np.median(patch.reshape(-1, 3), axis=0)
        return int(median[0]), int(median[1]), int(median[2])

    def _extract_front_portrait(self, image: Image.Image, geometry: FormalWearGeometry) -> Image.Image:
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        face = geometry.face_box

        draw.rectangle((0, 0, image.width, geometry.neck_top_y + 8), fill=255)
        draw.polygon(
            [
                (face['x'] - face['width'] * 0.16, face['y'] + face['height'] * 0.06),
                (face['x'] - face['width'] * 0.10, geometry.chin_y + 6),
                (geometry.neck_left_x - 16, geometry.neck_base_y + 10),
                (geometry.head_center_x - 18, geometry.chest_y + 10),
                (geometry.head_center_x + 18, geometry.chest_y + 10),
                (geometry.neck_right_x + 16, geometry.neck_base_y + 10),
                (face['x'] + face['width'] * 1.10, geometry.chin_y + 6),
                (face['x'] + face['width'] * 1.16, face['y'] + face['height'] * 0.06),
            ],
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.4))
        front = image.copy()
        front.putalpha(ImageChops.multiply(image.getchannel('A'), mask))
        return front

    def _draw_neck_bridge(self, layer: Image.Image, geometry: FormalWearGeometry, skin_tone: tuple[int, int, int]) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        bridge_width = (geometry.neck_right_x - geometry.neck_left_x) * 0.88
        draw.rounded_rectangle(
            [
                geometry.head_center_x - bridge_width / 2,
                geometry.neck_top_y + 2,
                geometry.head_center_x + bridge_width / 2,
                geometry.neck_base_y + 10,
            ],
            radius=14,
            fill=skin_tone + (212,),
        )

    def _curve(self, p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], steps: int = 12) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for index in range(steps + 1):
            t = index / steps
            mt = 1 - t
            x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
            y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
            points.append((x, y))
        return points

    def _draw_female_structured_suit(self, layer: Image.Image, geometry: FormalWearGeometry, palette: FormalWearPalette) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        shoulder_span = geometry.shoulder_right_x - geometry.shoulder_left_x
        body_left = geometry.head_center_x - shoulder_span * 0.34
        body_right = geometry.head_center_x + shoulder_span * 0.34
        body_top = min(geometry.shoulder_left_y, geometry.shoulder_right_y) + 8
        body_bottom = geometry.torso_bottom_y

        draw.rounded_rectangle(
            [body_left, body_top, body_right, body_bottom],
            radius=34,
            fill=palette.suit + (242,),
        )
        shoulder_radius = shoulder_span * 0.17
        draw.ellipse(
            [
                geometry.shoulder_left_x - shoulder_radius * 0.20,
                geometry.shoulder_left_y - shoulder_radius * 0.62,
                geometry.shoulder_left_x + shoulder_radius * 1.65,
                geometry.shoulder_left_y + shoulder_radius * 0.95,
            ],
            fill=palette.suit + (242,),
        )
        draw.ellipse(
            [
                geometry.shoulder_right_x - shoulder_radius * 1.65,
                geometry.shoulder_right_y - shoulder_radius * 0.62,
                geometry.shoulder_right_x + shoulder_radius * 0.20,
                geometry.shoulder_right_y + shoulder_radius * 0.95,
            ],
            fill=palette.suit + (242,),
        )

        blouse_top = geometry.neck_base_y + 2
        blouse_bottom = geometry.waist_y + 18
        blouse_width = (geometry.collar_right_x - geometry.collar_left_x) * 1.75
        blouse_layer = Image.new('RGBA', layer.size, (0, 0, 0, 0))
        blouse_draw = ImageDraw.Draw(blouse_layer, 'RGBA')
        blouse_draw.rounded_rectangle(
            [
                geometry.head_center_x - blouse_width / 2,
                blouse_top,
                geometry.head_center_x + blouse_width / 2,
                blouse_bottom,
            ],
            radius=28,
            fill=palette.shirt + (246,),
        )
        neckline = self._curve(
            (geometry.collar_left_x + 2, blouse_top + 4),
            (geometry.head_center_x, blouse_top + 46),
            (geometry.collar_right_x - 2, blouse_top + 4),
            steps=18,
        )
        neckline_mask = Image.new('L', layer.size, 0)
        neckline_draw = ImageDraw.Draw(neckline_mask)
        neckline_draw.polygon(
            neckline
            + [
                (geometry.collar_right_x - 8, geometry.neck_top_y - 6),
                (geometry.collar_left_x + 8, geometry.neck_top_y - 6),
            ],
            fill=255,
        )
        blouse_alpha = blouse_layer.getchannel('A')
        blouse_alpha = ImageChops.subtract(blouse_alpha, neckline_mask)
        blouse_layer.putalpha(blouse_alpha)
        layer.alpha_composite(blouse_layer)

        left_panel = self._curve(
            (geometry.shoulder_left_x + 10, geometry.shoulder_left_y + 4),
            (geometry.head_center_x - shoulder_span * 0.35, geometry.chest_y + 6),
            (geometry.head_center_x - shoulder_span * 0.24, body_bottom),
            steps=12,
        )
        left_inner = self._curve(
            (geometry.head_center_x - 20, body_bottom),
            (geometry.head_center_x - 30, geometry.chest_y + 54),
            (geometry.collar_left_x + 2, geometry.neck_base_y + 4),
            steps=10,
        )
        draw.polygon(left_panel + left_inner + [(geometry.shoulder_left_x + 4, geometry.shoulder_left_y + 4)], fill=palette.suit_dark + (244,))

        right_panel = self._curve(
            (geometry.shoulder_right_x - 10, geometry.shoulder_right_y + 4),
            (geometry.head_center_x + shoulder_span * 0.35, geometry.chest_y + 6),
            (geometry.head_center_x + shoulder_span * 0.24, body_bottom),
            steps=12,
        )
        right_inner = self._curve(
            (geometry.head_center_x + 20, body_bottom),
            (geometry.head_center_x + 30, geometry.chest_y + 54),
            (geometry.collar_right_x - 2, geometry.neck_base_y + 4),
            steps=10,
        )
        draw.polygon(right_panel + right_inner + [(geometry.shoulder_right_x - 4, geometry.shoulder_right_y + 4)], fill=palette.suit_dark + (244,))

        left_lapel = [
            (geometry.collar_left_x + 2, geometry.neck_base_y + 4),
            (geometry.head_center_x - 16, geometry.chest_y + 20),
            (geometry.head_center_x - 34, geometry.chest_y + 4),
            (geometry.collar_left_x - 12, geometry.neck_base_y + 10),
        ]
        right_lapel = [
            (geometry.collar_right_x - 2, geometry.neck_base_y + 4),
            (geometry.head_center_x + 16, geometry.chest_y + 20),
            (geometry.head_center_x + 34, geometry.chest_y + 4),
            (geometry.collar_right_x + 12, geometry.neck_base_y + 10),
        ]
        draw.polygon(left_lapel, fill=palette.suit + (236,))
        draw.polygon(right_lapel, fill=palette.suit + (236,))
        draw.line(
            [(geometry.collar_left_x + 2, geometry.neck_base_y + 4), (geometry.head_center_x - 18, geometry.chest_y + 20)],
            fill=palette.suit_light + (150,),
            width=2,
        )
        draw.line(
            [(geometry.collar_right_x - 2, geometry.neck_base_y + 4), (geometry.head_center_x + 18, geometry.chest_y + 20)],
            fill=palette.suit_light + (150,),
            width=2,
        )
        draw.arc(
            [
                geometry.head_center_x - blouse_width * 0.24,
                blouse_top + 6,
                geometry.head_center_x + blouse_width * 0.24,
                blouse_top + 54,
            ],
            start=195,
            end=345,
            fill=(226, 228, 233, 180),
            width=2,
        )

    def _draw_male_jacket(self, layer: Image.Image, geometry: FormalWearGeometry, palette: FormalWearPalette, style: str) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        body_mid_left = geometry.head_center_x - 38
        body_mid_right = geometry.head_center_x + 38
        front_open_y = geometry.chest_y + (10 if style == 'simple' else 18)

        draw.polygon(
            [
                (geometry.shoulder_left_x, geometry.shoulder_left_y + 4),
                (geometry.collar_left_x - 28, geometry.neck_base_y),
                (body_mid_left, front_open_y),
                (geometry.head_center_x - 62, geometry.waist_y),
                (geometry.head_center_x - 44, geometry.torso_bottom_y),
                (geometry.shoulder_left_x - 8, geometry.torso_bottom_y),
            ],
            fill=palette.suit + (244,),
        )
        draw.polygon(
            [
                (geometry.shoulder_right_x, geometry.shoulder_right_y + 4),
                (geometry.collar_right_x + 28, geometry.neck_base_y),
                (body_mid_right, front_open_y),
                (geometry.head_center_x + 62, geometry.waist_y),
                (geometry.head_center_x + 44, geometry.torso_bottom_y),
                (geometry.shoulder_right_x + 8, geometry.torso_bottom_y),
            ],
            fill=palette.suit + (244,),
        )

    def _draw_female_jacket(self, layer: Image.Image, geometry: FormalWearGeometry, palette: FormalWearPalette, style: str) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        inward = 26 if style == 'simple' else 18
        draw.polygon(
            [
                (geometry.shoulder_left_x + 8, geometry.shoulder_left_y + 2),
                (geometry.collar_left_x - 18, geometry.neck_base_y - 2),
                (geometry.head_center_x - inward, geometry.chest_y + 8),
                (geometry.head_center_x - 42, geometry.waist_y),
                (geometry.head_center_x - 28, geometry.torso_bottom_y),
                (geometry.shoulder_left_x - 6, geometry.torso_bottom_y),
            ],
            fill=palette.suit + (240,),
        )
        draw.polygon(
            [
                (geometry.shoulder_right_x - 8, geometry.shoulder_right_y + 2),
                (geometry.collar_right_x + 18, geometry.neck_base_y - 2),
                (geometry.head_center_x + inward, geometry.chest_y + 8),
                (geometry.head_center_x + 42, geometry.waist_y),
                (geometry.head_center_x + 28, geometry.torso_bottom_y),
                (geometry.shoulder_right_x + 6, geometry.torso_bottom_y),
            ],
            fill=palette.suit + (240,),
        )

    def _draw_male_shirt_and_tie(self, layer: Image.Image, geometry: FormalWearGeometry, palette: FormalWearPalette, style: str) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        shirt_bottom = geometry.waist_y + (20 if style == 'business' else 4)
        shirt_open = 30 if style == 'business' else 24
        draw.polygon(
            [
                (geometry.collar_left_x - 4, geometry.neck_base_y),
                (geometry.head_center_x, geometry.chest_y + 12),
                (geometry.collar_left_x + 20, shirt_bottom),
                (geometry.head_center_x - shirt_open, shirt_bottom),
            ],
            fill=palette.shirt + (248,),
        )
        draw.polygon(
            [
                (geometry.collar_right_x + 4, geometry.neck_base_y),
                (geometry.head_center_x, geometry.chest_y + 12),
                (geometry.collar_right_x - 20, shirt_bottom),
                (geometry.head_center_x + shirt_open, shirt_bottom),
            ],
            fill=palette.shirt + (248,),
        )
        if style == 'business':
            tie_top = geometry.neck_base_y + 10
            tie_bottom = geometry.waist_y + 20
            tie_half = 15
            draw.polygon(
                [
                    (geometry.head_center_x, tie_top),
                    (geometry.head_center_x - tie_half, geometry.chest_y + 34),
                    (geometry.head_center_x - tie_half * 0.75, tie_bottom - 18),
                    (geometry.head_center_x, tie_bottom),
                    (geometry.head_center_x + tie_half * 0.75, tie_bottom - 18),
                    (geometry.head_center_x + tie_half, geometry.chest_y + 34),
                ],
                fill=palette.tie + (232,),
            )
            draw.polygon(
                [
                    (geometry.head_center_x, tie_top - 7),
                    (geometry.head_center_x - 14, tie_top + 12),
                    (geometry.head_center_x, tie_top + 28),
                    (geometry.head_center_x + 14, tie_top + 12),
                ],
                fill=palette.tie + (244,),
            )

    def _draw_female_inner_top(
        self,
        layer: Image.Image,
        geometry: FormalWearGeometry,
        palette: FormalWearPalette,
        skin_tone: tuple[int, int, int],
        style: str,
    ) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        neckline_depth = 34 if style == 'simple' else 48
        inner_width = (geometry.collar_right_x - geometry.collar_left_x) * 1.45
        draw.rounded_rectangle(
            [
                geometry.head_center_x - inner_width / 2,
                geometry.neck_base_y - 2,
                geometry.head_center_x + inner_width / 2,
                geometry.waist_y + 16,
            ],
            radius=26,
            fill=palette.shirt + (244,),
        )
        draw.polygon(
            [
                (geometry.head_center_x - inner_width * 0.18, geometry.neck_base_y + 4),
                (geometry.head_center_x, geometry.neck_base_y + neckline_depth),
                (geometry.head_center_x + inner_width * 0.18, geometry.neck_base_y + 4),
            ],
            fill=skin_tone + (0,),
        )
        draw.ellipse(
            [
                geometry.head_center_x - inner_width * 0.14,
                geometry.neck_base_y + 8,
                geometry.head_center_x + inner_width * 0.14,
                geometry.neck_base_y + neckline_depth + 10,
            ],
            fill=skin_tone + (0,),
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
        lapel_inner = 16 if style == 'simple' else 22
        lapel_outer = 48 if gender == 'female' else 60
        lapel_bottom = geometry.chest_y + (28 if gender == 'female' else 38)

        draw.polygon(
            [
                (geometry.collar_left_x - 14, geometry.neck_base_y - 2),
                (geometry.head_center_x - lapel_inner, lapel_bottom),
                (geometry.head_center_x - lapel_outer, geometry.neck_base_y + 32),
                (geometry.collar_left_x - 30, geometry.neck_base_y + 10),
            ],
            fill=palette.suit_dark + (242,),
        )
        draw.polygon(
            [
                (geometry.collar_right_x + 14, geometry.neck_base_y - 2),
                (geometry.head_center_x + lapel_inner, lapel_bottom),
                (geometry.head_center_x + lapel_outer, geometry.neck_base_y + 32),
                (geometry.collar_right_x + 30, geometry.neck_base_y + 10),
            ],
            fill=palette.suit_dark + (242,),
        )
        draw.line(
            [
                (geometry.head_center_x - lapel_inner, lapel_bottom),
                (geometry.head_center_x - lapel_outer + 6, geometry.neck_base_y + 30),
            ],
            fill=palette.suit_light + (190,),
            width=2,
        )
        draw.line(
            [
                (geometry.head_center_x + lapel_inner, lapel_bottom),
                (geometry.head_center_x + lapel_outer - 6, geometry.neck_base_y + 30),
            ],
            fill=palette.suit_light + (190,),
            width=2,
        )

    def _draw_seam_shadows(self, layer: Image.Image, geometry: FormalWearGeometry, palette: FormalWearPalette) -> None:
        draw = ImageDraw.Draw(layer, 'RGBA')
        draw.line(
            [
                (geometry.shoulder_left_x + 10, geometry.shoulder_left_y + 4),
                (geometry.head_center_x - 60, geometry.waist_y),
            ],
            fill=palette.suit_dark + (86,),
            width=4,
        )
        draw.line(
            [
                (geometry.shoulder_right_x - 10, geometry.shoulder_right_y + 4),
                (geometry.head_center_x + 60, geometry.waist_y),
            ],
            fill=palette.suit_dark + (86,),
            width=4,
        )
