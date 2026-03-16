from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PhotoSizeTemplate:
    sceneKey: str
    sceneName: str
    widthMm: int
    heightMm: int
    pixelWidth: int
    pixelHeight: int
    description: str


PHOTO_SIZE_TEMPLATES: Dict[str, PhotoSizeTemplate] = {
    "one_inch": PhotoSizeTemplate(
        sceneKey="one_inch",
        sceneName="一寸照",
        widthMm=25,
        heightMm=35,
        pixelWidth=295,
        pixelHeight=413,
        description="常见报名与工卡使用的一寸证件照",
    ),
    "two_inch": PhotoSizeTemplate(
        sceneKey="two_inch",
        sceneName="二寸照",
        widthMm=35,
        heightMm=49,
        pixelWidth=413,
        pixelHeight=579,
        description="常见考试报名和简历使用的二寸证件照",
    ),
    "passport": PhotoSizeTemplate(
        sceneKey="passport",
        sceneName="护照照",
        widthMm=33,
        heightMm=48,
        pixelWidth=413,
        pixelHeight=579,
        description="中国护照标准尺寸",
    ),
    "visa": PhotoSizeTemplate(
        sceneKey="visa",
        sceneName="签证照",
        widthMm=35,
        heightMm=45,
        pixelWidth=413,
        pixelHeight=531,
        description="签证常用尺寸",
    ),
    "driver_license": PhotoSizeTemplate(
        sceneKey="driver_license",
        sceneName="驾驶证照",
        widthMm=22,
        heightMm=32,
        pixelWidth=260,
        pixelHeight=378,
        description="驾驶证证件照尺寸",
    ),
    "resume": PhotoSizeTemplate(
        sceneKey="resume",
        sceneName="简历照",
        widthMm=35,
        heightMm=49,
        pixelWidth=413,
        pixelHeight=579,
        description="简历常用证件照规格",
    ),
    "exam": PhotoSizeTemplate(
        sceneKey="exam",
        sceneName="考试报名照",
        widthMm=30,
        heightMm=40,
        pixelWidth=354,
        pixelHeight=472,
        description="考试报名系统常见规格",
    ),
}


def build_custom_template(width_mm: int, height_mm: int, dpi: int = 300) -> PhotoSizeTemplate:
    pixel_width = int(round(width_mm / 25.4 * dpi))
    pixel_height = int(round(height_mm / 25.4 * dpi))
    return PhotoSizeTemplate(
        sceneKey="custom",
        sceneName="自定义尺寸",
        widthMm=width_mm,
        heightMm=height_mm,
        pixelWidth=max(pixel_width, 1),
        pixelHeight=max(pixel_height, 1),
        description="由主业务服务传入的自定义尺寸",
    )
