from typing import Literal

from pydantic import BaseModel, Field


FormalWearGender = Literal['male', 'female']
FormalWearStyle = Literal['standard', 'business', 'simple']
FormalWearColor = Literal['black', 'navy', 'gray']


class FormalWearData(BaseModel):
    taskId: str
    previewUrl: str
    hdUrl: str
    gender: FormalWearGender
    style: FormalWearStyle
    color: FormalWearColor
    warnings: list[str] = Field(default_factory=list)
