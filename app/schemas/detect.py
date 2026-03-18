from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import FaceBox


class DetectData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hasFace: bool
    faceCount: int
    width: int
    height: int
    pass_: bool = Field(alias='pass')
    reasons: List[str]
    faceBoxes: List[FaceBox]
    recommended: bool
    warning: Optional[str] = None
