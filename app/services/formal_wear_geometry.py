from dataclasses import dataclass

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
    neck_left_x: float
    neck_right_x: float
    shoulder_left_x: float
    shoulder_right_x: float
    shoulder_y: float
    chest_y: float
    waist_y: float
    torso_bottom_y: float


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


def estimate_formal_wear_geometry(canvas_width: int, canvas_height: int, face_box: dict[str, int] | None, gender: str | None, style: str) -> FormalWearGeometry:
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

    face_center_x = face_box['x'] + face_box['width'] / 2
    chin_y = face_box['y'] + face_box['height'] * 1.02

    gender_ratio = 0.46 if gender == 'female' else 0.52
    style_ratio = {
        'simple': 1.90,
        'business': 2.15,
        'standard': 2.05,
        'formal': 2.05,
    }.get(style, 2.0)
    shoulder_span = face_box['width'] * style_ratio
    neck_width = face_box['width'] * gender_ratio
    shoulder_y = min(canvas_height * 0.72, chin_y + face_box['height'] * 0.38)
    chest_y = min(canvas_height * 0.82, shoulder_y + face_box['height'] * 0.34)
    waist_y = min(canvas_height * 0.92, shoulder_y + face_box['height'] * 0.95)

    return FormalWearGeometry(
        face_box=face_box,
        head_center_x=face_center_x,
        chin_y=chin_y,
        neck_left_x=face_center_x - neck_width / 2,
        neck_right_x=face_center_x + neck_width / 2,
        shoulder_left_x=max(-canvas_width * 0.05, face_center_x - shoulder_span / 2),
        shoulder_right_x=min(canvas_width * 1.05, face_center_x + shoulder_span / 2),
        shoulder_y=shoulder_y,
        chest_y=chest_y,
        waist_y=waist_y,
        torso_bottom_y=canvas_height + canvas_height * 0.08,
    )
