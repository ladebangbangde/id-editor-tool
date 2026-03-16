from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image_cv(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def save_jpeg_cv(image: np.ndarray, output_path: str, quality: int = 95) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def save_pil_image(image: Image.Image, output_path: str, quality: int = 95, dpi: int = 300) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="JPEG", quality=quality, dpi=(dpi, dpi))


def cv_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    if image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(rgba)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
