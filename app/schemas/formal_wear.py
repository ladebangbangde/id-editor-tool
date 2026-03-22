from typing import List, Optional

from pydantic import BaseModel


class FormalWearData(BaseModel):
    taskId: str
    previewUrl: str
    hdUrl: str
    gender: Optional[str] = None
    style: Optional[str] = None
    color: str
    warnings: List[str]
    previewPath: str = ''
    hdPath: str = ''
