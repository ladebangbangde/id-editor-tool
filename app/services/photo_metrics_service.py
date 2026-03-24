from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class FaceBox:
    x: int
    y: int
    width: int
    height: int


class PhotoMetricsService:
    """CPU-friendly image quality and composition metrics."""

    @staticmethod
    def _cv2():
        import cv2

        return cv2

    @staticmethod
    def _to_gray_np(image: Image.Image) -> np.ndarray:
        cv2 = PhotoMetricsService._cv2()
        rgb = np.asarray(image.convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def calculate(
        self,
        image: Image.Image,
        face_box: FaceBox | None,
    ) -> dict[str, float]:
        width, height = image.size
        cv2 = self._cv2()
        gray = self._to_gray_np(image)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        edges = cv2.Canny(gray, 80, 180)
        edge_density = float(np.count_nonzero(edges) / max(edges.size, 1))

        metrics: dict[str, float] = {
            'blur_score': blur_score,
            'brightness': brightness,
            'edge_density': edge_density,
            'image_width': float(width),
            'image_height': float(height),
            'face_count': 0.0,
            'face_width_ratio': 0.0,
            'face_height_ratio': 0.0,
            'face_center_x': 0.0,
            'face_center_y': 0.0,
        }

        if face_box is None:
            return metrics

        center_x = face_box.x + face_box.width / 2
        center_y = face_box.y + face_box.height / 2
        metrics.update(
            {
                'face_count': 1.0,
                'face_width_ratio': face_box.width / max(width, 1),
                'face_height_ratio': face_box.height / max(height, 1),
                'face_center_x': center_x / max(width, 1),
                'face_center_y': center_y / max(height, 1),
            }
        )
        return metrics
