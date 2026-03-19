from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class PhotoSizeTemplate:
    sceneKey: str
    sceneName: str
    sceneNameEn: str
    widthMm: int
    heightMm: int
    pixelWidth: int
    pixelHeight: int
    unit: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


PHOTO_SIZE_TEMPLATES: Dict[str, PhotoSizeTemplate] = {
    'one_inch': PhotoSizeTemplate(
        sceneKey='one_inch',
        sceneName='一寸照',
        sceneNameEn='One Inch',
        widthMm=25,
        heightMm=35,
        pixelWidth=295,
        pixelHeight=413,
        unit='mm',
        description='常见报名、工牌与基础档案使用的一寸证件照。',
    ),
    'two_inch': PhotoSizeTemplate(
        sceneKey='two_inch',
        sceneName='二寸照',
        sceneNameEn='Two Inch',
        widthMm=35,
        heightMm=49,
        pixelWidth=413,
        pixelHeight=579,
        unit='mm',
        description='常见考试报名、简历与个人材料使用的二寸证件照。',
    ),
    'small_two_inch': PhotoSizeTemplate(
        sceneKey='small_two_inch',
        sceneName='小二寸',
        sceneNameEn='Small Two Inch',
        widthMm=35,
        heightMm=45,
        pixelWidth=413,
        pixelHeight=531,
        unit='mm',
        description='签证、考试报名等常见规格。',
    ),
    'passport': PhotoSizeTemplate(
        sceneKey='passport',
        sceneName='护照照',
        sceneNameEn='Passport',
        widthMm=33,
        heightMm=48,
        pixelWidth=390,
        pixelHeight=567,
        unit='mm',
        description='中国护照常见规格，可作为后续裁切参考。',
    ),
    'visa': PhotoSizeTemplate(
        sceneKey='visa',
        sceneName='签证照',
        sceneNameEn='Visa',
        widthMm=35,
        heightMm=45,
        pixelWidth=413,
        pixelHeight=531,
        unit='mm',
        description='签证申请与国际出行材料常见尺寸。',
    ),
    'driver_license': PhotoSizeTemplate(
        sceneKey='driver_license',
        sceneName='驾驶证照',
        sceneNameEn='Driver License',
        widthMm=22,
        heightMm=32,
        pixelWidth=260,
        pixelHeight=378,
        unit='mm',
        description='驾驶证等小规格证件使用尺寸。',
    ),
}


def build_custom_template(width_mm: int, height_mm: int, dpi: int = 300) -> PhotoSizeTemplate:
    pixel_width = int(round(width_mm / 25.4 * dpi))
    pixel_height = int(round(height_mm / 25.4 * dpi))
    return PhotoSizeTemplate(
        sceneKey='custom',
        sceneName='自定义尺寸',
        sceneNameEn='Custom Size',
        widthMm=width_mm,
        heightMm=height_mm,
        pixelWidth=max(pixel_width, 1),
        pixelHeight=max(pixel_height, 1),
        unit='mm',
        description='按传入毫米尺寸换算得到的自定义规格。',
    )


def list_photo_sizes() -> list[dict]:
    return [template.to_dict() for template in PHOTO_SIZE_TEMPLATES.values()]
