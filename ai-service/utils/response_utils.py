from __future__ import annotations

from typing import Any


def success_response(data: Any, message: str = 'OK') -> dict:
    return {'success': True, 'message': message, 'errorCode': None, 'data': data}


def error_response(message: str, error_code: str | None = None, data: Any = None) -> dict:
    return {'success': False, 'message': message, 'errorCode': error_code, 'data': data}
