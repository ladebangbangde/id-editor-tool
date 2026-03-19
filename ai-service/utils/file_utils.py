from pathlib import Path

from core.config import get_settings


settings = get_settings()


def ensure_upload_dirs() -> None:
    for path in settings.upload_dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def build_output_path(category: str, filename: str) -> str:
    dir_path = settings.upload_dirs[category]
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return str(Path(dir_path) / filename)


def public_url_for_path(path: str | Path) -> str:
    normalized_path = Path(path)
    try:
        relative_path = normalized_path.resolve().relative_to(settings.upload_root_path.resolve())
        return f"{settings.static_mount_path}/{relative_path.as_posix()}"
    except Exception:
        try:
            relative_path = normalized_path.relative_to(settings.upload_root_path)
            return f"{settings.static_mount_path}/{relative_path.as_posix()}"
        except Exception:
            return str(normalized_path).replace('\\', '/')


def to_url_like_path(path: str | Path) -> str:
    return public_url_for_path(path)
