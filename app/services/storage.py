from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.utils.file_naming import date_slug
from app.utils.image_io import relative_url


@dataclass
class StoredFile:
    path: Path
    url: str


class StorageService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.output_root = self.settings.resolved_output_dir

    def task_dir(self, task_id: str) -> Path:
        path = self.output_root / date_slug() / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stored_file(self, path: Path) -> StoredFile:
        return StoredFile(path=path, url=relative_url(path, self.output_root))
