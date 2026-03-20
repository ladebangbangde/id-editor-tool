from __future__ import annotations

from typing import Any

from core.exceptions import (
    ERROR_FACE_OCCLUDED,
    ERROR_FACE_TOO_SMALL,
    ERROR_FILE_NOT_FOUND,
    ERROR_HEAD_CROPPED,
    ERROR_IMAGE_TOO_BLURRY,
    ERROR_IMAGE_TOO_SMALL,
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_IMAGE,
    ERROR_MULTIPLE_FACES_DETECTED,
    ERROR_NO_FACE_DETECTED,
    ERROR_POSE_INVALID,
    ERROR_PROCESS_FAILED,
)

TOOL_ERROR_CODE_MAP = {
    ERROR_INVALID_ARGUMENT: 1006,
    ERROR_INVALID_IMAGE: 1002,
    ERROR_IMAGE_TOO_SMALL: 1003,
    ERROR_NO_FACE_DETECTED: 1004,
    ERROR_MULTIPLE_FACES_DETECTED: 1004,
    ERROR_FACE_TOO_SMALL: 1005,
    ERROR_IMAGE_TOO_BLURRY: 1005,
    ERROR_POSE_INVALID: 1005,
    ERROR_FACE_OCCLUDED: 1005,
    ERROR_HEAD_CROPPED: 1005,
    ERROR_FILE_NOT_FOUND: 1006,
    ERROR_PROCESS_FAILED: 2001,
}


def success_response(data: Any, message: str = 'OK') -> dict:
    return {'success': True, 'message': message, 'errorCode': None, 'data': data}


def error_response(message: str, error_code: str | None = None, data: Any = None) -> dict:
    return {'success': False, 'message': message, 'errorCode': error_code, 'data': data}


def tool_success_response(data: Any, message: str = 'success') -> dict:
    return {'code': 0, 'message': message, 'data': data}


def map_tool_error_code(error_code: str | None) -> int:
    if not error_code:
        return 9001
    return TOOL_ERROR_CODE_MAP.get(error_code, 9001)


def tool_error_response(message: str, error_code: str | None = None, data: Any = None) -> dict:
    return {'code': map_tool_error_code(error_code), 'message': message, 'data': data}


def is_tool_v1_path(path: str) -> bool:
    return (path or '').startswith('/api/v1/')
