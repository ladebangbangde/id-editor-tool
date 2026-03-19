from pathlib import Path

from utils.config import get_settings


def ensure_upload_dirs() -> None:
    settings = get_settings()
    for sub in ["preview", "hd", "print", "temp", "original"]:
        (settings.upload_base_path / sub).mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def build_output_path(category: str, filename: str) -> str:
    settings = get_settings()
    if category not in {"preview", "hd", "print", "temp", "original"}:
        raise ValueError(f"Unsupported category: {category}")

    abs_path = settings.upload_base_path / category / filename
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    return str(abs_path)


def resolve_input_path(path_or_url: str) -> str:
    """
    Resolve request path to an absolute local path under shared uploads mount.
    Supports:
    - absolute path: /data/uploads/original/xxx.jpg
    - relative uploads prefix: uploads/original/xxx.jpg
    - relative bare path: original/xxx.jpg
    """
    settings = get_settings()
    raw = Path(path_or_url)
    if raw.is_absolute():
        return str(raw)

    prefix = f"{settings.upload_public_prefix}/"
    normalized = path_or_url.replace("\\", "/")
    if normalized.startswith(prefix):
        suffix = normalized[len(prefix):]
        return str(settings.upload_base_path / suffix)

    return str(settings.upload_base_path / normalized)


def to_url_like_path(path: str | Path) -> str:
    """Convert absolute file path to server-compatible uploads/* path."""
    settings = get_settings()
    path_obj = Path(path)
    try:
        relative = path_obj.resolve().relative_to(settings.upload_base_path.resolve())
    except Exception:
        normalized = str(path).replace("\\", "/")
        return normalized

    public_prefix = settings.upload_public_prefix.strip("/")
    rel = str(relative).replace("\\", "/")
    return f"{public_prefix}/{rel}"


def public_url_for_path(path: str | Path) -> str:
    return to_url_like_path(path)
