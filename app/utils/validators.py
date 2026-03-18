from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import InvalidArgumentError, InvalidImageError
from app.utils.image_io import SUPPORTED_EXTENSIONS


def validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise InvalidArgumentError('Uploaded file must have a filename')
    suffix = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if suffix not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(f'Unsupported image format: {suffix or "unknown"}')


def validate_content_size(content: bytes) -> None:
    settings = get_settings()
    if len(content) > settings.max_upload_size_bytes:
        raise InvalidArgumentError(
            f'Uploaded file exceeds max size of {settings.max_upload_size_mb} MB'
        )
