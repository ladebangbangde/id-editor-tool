from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage import color, data, transform
from skimage.feature import Cascade

from app.core.config import get_settings
from app.core.exceptions import ImageTooSmallError


@dataclass
class FaceDetectionResult:
    width: int
    height: int
    face_count: int
    has_face: bool
    recommended: bool
    reasons: list[str]
    face_boxes: list[dict[str, int]]
    primary_face: dict[str, int] | None


class FaceDetectionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.detector = Cascade(data.lbp_frontal_face_cascade_filename())

    def detect(self, image: Image.Image) -> FaceDetectionResult:
        width, height = image.size
        reasons: list[str] = []
        if width < self.settings.min_image_width or height < self.settings.min_image_height:
            raise ImageTooSmallError(
                f'Image is too small: minimum is '
                f'{self.settings.min_image_width}x{self.settings.min_image_height}px'
            )

        rgb = np.asarray(image.convert('RGB'))
        gray = color.rgb2gray(rgb)
        scale = 1.0
        max_side = max(width, height)
        if max_side > 1200:
            scale = 1200.0 / max_side
            gray = transform.rescale(gray, scale, anti_aliasing=True)

        detections = self.detector.detect_multi_scale(
            img=gray,
            scale_factor=1.2,
            step_ratio=1,
            min_size=(60, 60),
            max_size=(int(gray.shape[0] * 0.9), int(gray.shape[1] * 0.9)),
        )
        face_boxes = []
        for item in detections:
            x = int(item['c'] / scale)
            y = int(item['r'] / scale)
            w = int(item['width'] / scale)
            h = int(item['height'] / scale)
            face_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})

        face_count = len(face_boxes)
        if face_count == 0:
            reasons.append('No face detected')
        elif face_count > 1:
            reasons.append('Multiple faces detected')

        recommended = face_count == 1
        primary_face = max(face_boxes, key=lambda box: box['width'] * box['height']) if face_boxes else None
        return FaceDetectionResult(
            width=width,
            height=height,
            face_count=face_count,
            has_face=face_count > 0,
            recommended=recommended,
            reasons=reasons,
            face_boxes=face_boxes,
            primary_face=primary_face,
        )
