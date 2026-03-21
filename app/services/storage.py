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
        self.upload_root = self.settings.upload_root_path
        self.category_roots = self.settings.upload_dirs

    def category_task_dir(self, category: str, task_id: str) -> Path:
        path = self.category_roots[category] / date_slug() / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def preview_path(self, task_id: str, filename: str) -> Path:
        return self.category_task_dir('preview', task_id) / filename

    def hd_path(self, task_id: str, filename: str) -> Path:
        return self.category_task_dir('hd', task_id) / filename

    def print_path(self, task_id: str, filename: str) -> Path:
        return self.category_task_dir('print', task_id) / filename

    def temp_path(self, task_id: str, filename: str) -> Path:
        return self.category_task_dir('temp', task_id) / filename

    def stored_file(self, path: Path) -> StoredFile:
        return StoredFile(
            path=path,
            url=relative_url(
                path,
                upload_root=self.upload_root,
                static_mount_path=self.settings.normalized_static_mount_path,
            ),
        )
