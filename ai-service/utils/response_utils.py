from typing import Any, Optional


def success_response(data: Any, message: str = "OK") -> dict:
    return {"success": True, "message": message, "data": data}


def error_response(message: str, data: Optional[Any] = None) -> dict:
    return {"success": False, "message": message, "data": data}
