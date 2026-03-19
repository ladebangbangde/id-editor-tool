from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ERROR_INVALID_IMAGE = 'INVALID_IMAGE'
ERROR_NO_FACE_DETECTED = 'NO_FACE_DETECTED'
ERROR_MULTIPLE_FACES_DETECTED = 'MULTIPLE_FACES_DETECTED'
ERROR_IMAGE_TOO_SMALL = 'IMAGE_TOO_SMALL'
ERROR_FILE_NOT_FOUND = 'FILE_NOT_FOUND'
ERROR_INVALID_ARGUMENT = 'INVALID_ARGUMENT'
ERROR_PROCESS_FAILED = 'PROCESS_FAILED'


@dataclass
class AppException(Exception):
    message: str
    error_code: str = ERROR_PROCESS_FAILED
    status_code: int = 400
    data: Any = None

    def __str__(self) -> str:
        return self.message
