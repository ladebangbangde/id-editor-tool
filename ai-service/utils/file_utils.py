from pathlib import Path

from utils.config import get_settings


def ensure_upload_dirs() -> None:
    settings = get_settings()
    for path in settings.upload_dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def build_output_path(category: str, filename: str) -> str:
    settings = get_settings()
    dir_path = settings.upload_dirs[category]
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return str(Path(dir_path) / filename)


def to_url_like_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    return normalized
