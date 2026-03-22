from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.services.specs import PhotoSpec


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class FormalWearGeometry:
    face_box: dict[str, int]
    head_center_x: float
    chin_y: float
    neck_top_y: float
    neck_base_y: float
    neck_left_x: float
    neck_right_x: float
    shoulder_left_x: float
    shoulder_right_x: float
    shoulder_left_y: float
    shoulder_right_y: float
    chest_y: float
    waist_y: float
    torso_bottom_y: float
    collar_left_x: float
    collar_right_x: float


def compute_crop_box(image_width: int, image_height: int, spec: PhotoSpec, face_box: dict[str, int] | None) -> CropBox:
    target_ratio = spec.width_px / spec.height_px

    if face_box:
        fx = face_box['x'] + face_box['width'] / 2
        fy = face_box['y'] + face_box['height'] / 2
        crop_height = min(image_height, max(face_box['height'] / 0.42, spec.height_px))
        crop_width = crop_height * target_ratio
        if crop_width > image_width:
            crop_width = image_width
            crop_height = crop_width / target_ratio
        left = max(0, min(image_width - crop_width, fx - crop_width / 2))
        top = max(0, min(image_height - crop_height, fy - crop_height * 0.38))
    else:
        crop_width = min(image_width, image_height * target_ratio)
        crop_height = crop_width / target_ratio
        left = (image_width - crop_width) / 2
        top = (image_height - crop_height) / 2

    return CropBox(
        left=int(round(left)),
        top=int(round(top)),
        right=int(round(left + crop_width)),
        bottom=int(round(top + crop_height)),
    )


def project_face_box(face_box: dict[str, int] | None, crop_box: CropBox, target_width: int, target_height: int) -> dict[str, int] | None:
    if face_box is None:
        return None

    scale_x = target_width / crop_box.width
    scale_y = target_height / crop_box.height
    return {
        'x': int(round((face_box['x'] - crop_box.left) * scale_x)),
        'y': int(round((face_box['y'] - crop_box.top) * scale_y)),
        'width': int(round(face_box['width'] * scale_x)),
        'height': int(round(face_box['height'] * scale_y)),
    }


def _row_bounds(alpha: np.ndarray, row: int, center_x: int, threshold: int = 12) -> tuple[int, int] | None:
    row_data = alpha[row]
    if row_data[center_x] <= threshold:
        return None

    left = center_x
    while left > 0 and row_data[left] > threshold:
        left -= 1
    right = center_x
    max_x = row_data.shape[0] - 1
    while right < max_x and row_data[right] > threshold:
        right += 1
    return left, right


def estimate_formal_wear_geometry(
    foreground_rgba: Image.Image,
    face_box: dict[str, int] | None,
    gender: str | None,
    style: str,
) -> FormalWearGeometry:
    canvas_width, canvas_height = foreground_rgba.size
    if face_box is None:
        face_width = canvas_width * 0.30
        face_height = canvas_height * 0.28
        face_x = (canvas_width - face_width) / 2
        face_y = canvas_height * 0.10
        face_box = {
            'x': int(round(face_x)),
            'y': int(round(face_y)),
            'width': int(round(face_width)),
            'height': int(round(face_height)),
        }

    alpha = np.asarray(foreground_rgba.getchannel('A'))
    face_center_x = int(round(face_box['x'] + face_box['width'] / 2))
    face_bottom = face_box['y'] + face_box['height']
    search_top = max(0, int(face_box['y'] + face_box['height'] * 0.72))
    search_bottom = min(canvas_height - 1, int(face_bottom + face_box['height'] * 0.65))

    measured_rows: list[tuple[int, int, int]] = []
    for row in range(search_top, search_bottom + 1):
        bounds = _row_bounds(alpha, row, face_center_x)
        if bounds is None:
            continue
        left, right = bounds
        width = right - left
        measured_rows.append((row, left, right))

    if measured_rows:
        neck_row, neck_left, neck_right = min(measured_rows, key=lambda item: item[2] - item[1])
    else:
        neck_row = int(round(face_bottom + face_box['height'] * 0.08))
        neck_left = int(round(face_center_x - face_box['width'] * 0.22))
        neck_right = int(round(face_center_x + face_box['width'] * 0.22))

    shoulder_candidates = [
        item
        for item in measured_rows
        if item[0] >= neck_row + face_box['height'] * 0.10
    ]
    if shoulder_candidates:
        shoulder_row, shoulder_left, shoulder_right = max(shoulder_candidates, key=lambda item: item[2] - item[1])
    else:
        style_scale = {'simple': 1.75, 'business': 2.25, 'standard': 2.0}.get(style, 2.0)
        shoulder_row = int(round(neck_row + face_box['height'] * 0.36))
        shoulder_left = int(round(face_center_x - face_box['width'] * style_scale / 2))
        shoulder_right = int(round(face_center_x + face_box['width'] * style_scale / 2))

    gender_expand = 0.14 if gender == 'female' else 0.18
    side_margin_left = shoulder_left
    side_margin_right = canvas_width - shoulder_right
    shoulder_expand = face_box['width'] * gender_expand
    shoulder_left_x = max(0.0, shoulder_left - min(side_margin_left * 0.45, shoulder_expand))
    shoulder_right_x = min(float(canvas_width), shoulder_right + min(side_margin_right * 0.45, shoulder_expand))

    shoulder_slope = face_box['height'] * (0.10 if gender == 'female' else 0.13)
    shoulder_left_y = min(canvas_height * 0.86, shoulder_row + shoulder_slope * 0.25)
    shoulder_right_y = min(canvas_height * 0.86, shoulder_row + shoulder_slope * 0.25)
    chest_y = min(canvas_height * 0.84, shoulder_row + face_box['height'] * (0.36 if gender == 'female' else 0.32))
    waist_y = min(canvas_height * 0.94, shoulder_row + face_box['height'] * 0.95)

    neck_width = max(neck_right - neck_left, face_box['width'] * (0.28 if gender == 'female' else 0.32))
    collar_spread = neck_width * (0.64 if gender == 'female' else 0.75)
    chin_y = neck_row - face_box['height'] * 0.10
    neck_top_y = chin_y + face_box['height'] * 0.06
    neck_base_y = neck_row + face_box['height'] * (0.18 if gender == 'female' else 0.14)

    return FormalWearGeometry(
        face_box=face_box,
        head_center_x=float(face_center_x),
        chin_y=float(chin_y),
        neck_top_y=float(neck_top_y),
        neck_base_y=float(neck_base_y),
        neck_left_x=float(face_center_x - neck_width / 2),
        neck_right_x=float(face_center_x + neck_width / 2),
        shoulder_left_x=float(shoulder_left_x),
        shoulder_right_x=float(shoulder_right_x),
        shoulder_left_y=float(shoulder_left_y),
        shoulder_right_y=float(shoulder_right_y),
        chest_y=float(chest_y),
        waist_y=float(waist_y),
        torso_bottom_y=float(canvas_height + canvas_height * 0.08),
        collar_left_x=float(face_center_x - collar_spread / 2),
        collar_right_x=float(face_center_x + collar_spread / 2),
    )
