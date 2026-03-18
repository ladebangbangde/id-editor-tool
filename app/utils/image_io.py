from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.exceptions import InvalidImageError


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_image_from_bytes(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.load()
        return image
    except Exception as exc:
        raise InvalidImageError('Uploaded file is not a valid image') from exc


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('RGB'))


def rgba_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def save_image(image: Image.Image, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **kwargs)


def relative_url(path: Path, output_root: Path) -> str:
    rel = path.resolve().relative_to(output_root.resolve())
    return f'/outputs/{rel.as_posix()}'
