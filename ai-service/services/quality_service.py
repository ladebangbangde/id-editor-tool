from typing import Tuple

from PIL import Image

from constants.status import QUALITY_STATUS_FAILED, QUALITY_STATUS_PASSED, QUALITY_STATUS_WARNING


class QualityService:
    def evaluate(self, image: Image.Image) -> Tuple[str, str]:
        width, height = image.size
        if width < 200 or height < 200:
            return QUALITY_STATUS_FAILED, "输出尺寸过小"
        if width < 350 or height < 450:
            return QUALITY_STATUS_WARNING, "输出清晰度一般，建议使用更高分辨率原图"
        return QUALITY_STATUS_PASSED, "质量检测通过"
