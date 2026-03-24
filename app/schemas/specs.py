from typing import List

from pydantic import BaseModel


class PhotoSizeSpec(BaseModel):
    sizeKey: str
    name: str
    widthMm: float
    heightMm: float
    pixelWidth: int
    pixelHeight: int
    aliases: List[str]
    category: str | None = None
    featured: bool = False
    canonical: bool = True


class PhotoSpecsData(BaseModel):
    supportedSizeKeys: List[str]
    specs: List[PhotoSizeSpec]
    customSizeSupported: bool = False
    customSizeHint: str
