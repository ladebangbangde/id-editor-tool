from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from core.exceptions import AppException, ERROR_FILE_NOT_FOUND, ERROR_INVALID_IMAGE


def open_pil_image(path: str) -> Image.Image:
    try:
        return Image.open(path)
    except FileNotFoundError as exc:
        raise AppException(f'Image not found: {path}', ERROR_FILE_NOT_FOUND, 404) from exc
    except UnidentifiedImageError as exc:
        raise AppException(f'Invalid image content: {path}', ERROR_INVALID_IMAGE, 400) from exc


def read_image_array(path: str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise AppException(f'Image not found: {path}', ERROR_FILE_NOT_FOUND, 404)
    try:
        image = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError as exc:
        raise AppException(f'Image not readable: {path}', ERROR_INVALID_IMAGE, 400) from exc
    return np.array(image)


def save_pil_image(image: Image.Image, output_path: str, quality: int = 95, dpi: int = 300) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.convert('RGB').save(output_path, format='JPEG', quality=quality, dpi=(dpi, dpi))
