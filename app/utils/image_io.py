from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import InvalidArgumentError, InvalidImageError


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_image_from_bytes(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.load()
        return image
    except Exception as exc:
        raise InvalidImageError('Uploaded file is not a valid image') from exc


def load_image_from_path(path: Path) -> Image.Image:
    try:
        image = Image.open(path)
        image.load()
        return image
    except Exception as exc:
        raise InvalidImageError(f'Input file is not a valid image: {path}') from exc


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('RGB'))


def rgba_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def save_image(image: Image.Image, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, **kwargs)


def resolve_input_path(path_or_url: str) -> Path:
    settings = get_settings()
    normalized = (path_or_url or '').replace('\\', '/').strip()
    if not normalized:
        raise InvalidArgumentError('Image path is required')

    candidate = Path(normalized)
    upload_root = settings.upload_root_path
    static_prefix = settings.normalized_static_mount_path.rstrip('/')

    if candidate.is_absolute() and not normalized.startswith(static_prefix + '/'):
        resolved = candidate.resolve()
        if not resolved.exists():
            raise InvalidArgumentError(f'Image path does not exist: {path_or_url}')
        try:
            resolved.relative_to(upload_root)
        except ValueError as exc:
            raise InvalidArgumentError(f'Image path must be under upload root: {upload_root}') from exc
        return resolved

    public_relative = normalized.lstrip('/')
    if normalized.startswith(static_prefix + '/'):
        public_relative = normalized[len(static_prefix) + 1 :]
    elif normalized.startswith(static_prefix.lstrip('/') + '/'):
        public_relative = normalized[len(static_prefix):].lstrip('/')

    resolved = (upload_root / public_relative).resolve()
    try:
        resolved.relative_to(upload_root)
    except ValueError as exc:
        raise InvalidArgumentError(f'Image path must be under upload root: {upload_root}') from exc

    if not resolved.exists():
        raise InvalidArgumentError(f'Image path does not exist: {path_or_url}')
    return resolved


def relative_url(path: Path, upload_root: Path | None = None, static_mount_path: str | None = None) -> str:
    settings = get_settings()
    root = (upload_root or settings.upload_root_path).resolve()
    mount_path = static_mount_path or settings.normalized_static_mount_path
    rel = path.resolve().relative_to(root)
    return f"{mount_path.rstrip('/')}/{rel.as_posix()}"
