from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class DetectRequest(BaseModel):
    imageId: str = Field(min_length=1)
    originalImagePath: str = Field(min_length=1)


class GenerateIdPhotoRequest(BaseModel):
    imageId: str = Field(min_length=1)
    sourceType: Literal["scene", "custom"]
    sceneKey: Optional[str] = None
    customWidthMm: Optional[int] = None
    customHeightMm: Optional[int] = None
    backgroundColor: Literal["white", "blue", "red"]
    beautyEnabled: Optional[bool] = False
    printLayoutType: Optional[Literal["six", "eight", "twelve"]] = None
    originalImagePath: str = Field(min_length=1)
    outputBaseDir: Optional[str] = None

    @model_validator(mode="after")
    def validate_source_mode(self):
        if self.sourceType == "scene" and not self.sceneKey:
            raise ValueError("scene mode requires sceneKey")
        if self.sourceType == "custom":
            if not self.customWidthMm or not self.customHeightMm:
                raise ValueError("custom mode requires customWidthMm and customHeightMm")
        return self


class GeneratePrintLayoutRequest(BaseModel):
    imageId: str = Field(min_length=1)
    hdImagePath: str = Field(min_length=1)
    layoutType: Literal["six", "eight", "twelve"]
