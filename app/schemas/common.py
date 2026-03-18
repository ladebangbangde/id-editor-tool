from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar('T')


class ErrorBody(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    message: str = 'ok'
    data: Optional[T] = None
    error: Optional[ErrorBody] = None


class SizeInfo(BaseModel):
    key: str
    name: str
    widthPx: int
    heightPx: int
    widthMm: float
    heightMm: float


class FileInfo(BaseModel):
    path: str
    url: str


class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
