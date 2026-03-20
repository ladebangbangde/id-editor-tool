from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import UploadFile

from core.exceptions import AppException, ERROR_INVALID_ARGUMENT
from utils.config import get_settings
from utils.file_utils import ensure_upload_dirs, public_url_for_path, to_url_like_path
from utils.logger import get_logger


class StorageService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger(component='storage_service')

    @staticmethod
    def _sanitize_name(filename: str | None) -> str:
        if not filename:
            return 'upload.jpg'
        return Path(filename).name.replace(' ', '_')

    def save_upload(self, upload_file: UploadFile, image_id: str | None = None) -> dict[str, str]:
        ensure_upload_dirs()
        safe_name = self._sanitize_name(upload_file.filename)
        suffix = Path(safe_name).suffix.lower() or '.jpg'
        if suffix not in set(self.settings.allowed_image_extensions):
            self.logger.bind(filename=upload_file.filename, suffix=suffix).warning('rejected upload due to unsupported file type')
            raise AppException('Unsupported upload file type', ERROR_INVALID_ARGUMENT, 400)

        generated_id = image_id or Path(safe_name).stem or secrets.token_hex(8)
        target_name = f'{generated_id}_{secrets.token_hex(4)}{suffix}'
        target_path = self.settings.upload_dirs['original'] / target_name
        storage_logger = self.logger.bind(image_id=generated_id, filename=safe_name, target_path=target_path)
        storage_logger.info('saving uploaded file')

        size_limit = self.settings.max_upload_mb * 1024 * 1024
        total = 0
        with target_path.open('wb') as output:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > size_limit:
                    target_path.unlink(missing_ok=True)
                    storage_logger.bind(total_bytes=total, size_limit=size_limit).warning('upload exceeded size limit')
                    raise AppException(
                        f'Upload file exceeds {self.settings.max_upload_mb} MB limit',
                        ERROR_INVALID_ARGUMENT,
                        400,
                    )
                output.write(chunk)

        storage_logger.bind(total_bytes=total).info('uploaded file saved successfully')
        return {
            'imageId': generated_id,
            'originalImagePath': str(target_path),
            'originalImageStoragePath': to_url_like_path(target_path),
            'originalImageUrl': public_url_for_path(target_path),
            'filename': target_name,
        }
