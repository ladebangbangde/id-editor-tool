from __future__ import annotations

from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator


class DetectRequest(BaseModel):
    imageId: str = Field(min_length=1)
    originalImagePath: str = Field(min_length=1, validation_alias=AliasChoices('originalImagePath', 'imagePath'))


class GenerateIdPhotoRequest(BaseModel):
    imageId: str = Field(min_length=1)
    sourceType: Literal['scene', 'custom'] = 'scene'
    sceneKey: Optional[str] = Field(default='passport', validation_alias=AliasChoices('sceneKey', 'sizeName'))
    customWidthMm: Optional[int] = None
    customHeightMm: Optional[int] = None
    backgroundColor: Literal['white', 'blue', 'red'] = 'white'
    beautyEnabled: Optional[bool] = False
    printLayoutType: Optional[Literal['six', 'eight', 'twelve']] = None
    originalImagePath: str = Field(min_length=1)
    outputBaseDir: Optional[str] = None

    @model_validator(mode='after')
    def validate_source_mode(self):
        if self.sourceType == 'scene' and not self.sceneKey:
            raise ValueError('scene mode requires sceneKey or sizeName')
        if self.sourceType == 'custom':
            if not self.customWidthMm or not self.customHeightMm:
                raise ValueError('custom mode requires customWidthMm and customHeightMm')
        return self


class GeneratePrintLayoutRequest(BaseModel):
    imageId: str = Field(min_length=1)
    hdImagePath: str = Field(min_length=1, validation_alias=AliasChoices('hdImagePath', 'imagePath'))
    layoutType: Literal['six', 'eight', 'twelve']
