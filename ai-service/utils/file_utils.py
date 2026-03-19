from __future__ import annotations

from pathlib import Path

from core.exceptions import AppException, ERROR_FILE_NOT_FOUND
from utils.config import get_settings


CATEGORY_TO_SETTING = {
    'original': 'original',
    'preview': 'preview',
    'hd': 'hd',
    'print': 'print',
    'temp': 'temp',
}


def ensure_upload_dirs() -> None:
    settings = get_settings()
    for directory in settings.upload_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def build_output_path(category: str, filename: str) -> str:
    settings = get_settings()
    if category not in CATEGORY_TO_SETTING:
        raise ValueError(f'Unsupported category: {category}')

    abs_path = settings.upload_dirs[category] / filename
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return str(abs_path)


def resolve_input_path(path_or_url: str) -> str:
    settings = get_settings()
    normalized = (path_or_url or '').replace('\\', '/').strip()
    if not normalized:
        raise AppException('image path is required', ERROR_FILE_NOT_FOUND, 400)

    raw = Path(normalized)
    if raw.is_absolute():
        if raw.exists():
            return str(raw)
        raise AppException(f'file not found: {normalized}', ERROR_FILE_NOT_FOUND, 404)

    public_prefix = settings.upload_public_prefix.strip('/')
    static_prefix = settings.static_mount_path.rstrip('/')

    if normalized.startswith(static_prefix + '/'):
        normalized = normalized[len(static_prefix) + 1 :]
    elif normalized.startswith(public_prefix + '/'):
        normalized = normalized[len(public_prefix) + 1 :]

    candidate = settings.upload_base_path / normalized
    if candidate.exists():
        return str(candidate)

    alt_candidate = settings.upload_base_path / Path(normalized).name if '/' not in normalized else None
    if alt_candidate and alt_candidate.exists():
        return str(alt_candidate)

    raise AppException(f'file not found: {path_or_url}', ERROR_FILE_NOT_FOUND, 404)


def to_url_like_path(path: str | Path) -> str:
    settings = get_settings()
    path_obj = Path(path)
    try:
        relative = path_obj.resolve().relative_to(settings.upload_base_path.resolve())
    except Exception:
        return str(path).replace('\\', '/')

    public_prefix = settings.upload_public_prefix.strip('/')
    rel = str(relative).replace('\\', '/')
    return f'{public_prefix}/{rel}'


def public_url_for_path(path: str | Path) -> str:
    settings = get_settings()
    relative = to_url_like_path(path)
    return f"{settings.static_mount_path}/{relative.split('/', 1)[1]}"
