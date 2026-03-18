from typing import List, Optional

from pydantic import BaseModel

from app.schemas.common import FileInfo, SizeInfo


class LayoutData(BaseModel):
    taskId: str
    layoutPath: str
    layoutUrl: str
    paper: str
    count: int
    photoSize: SizeInfo
    warnings: List[str]
    sourceHd: Optional[FileInfo] = None
