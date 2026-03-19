from __future__ import annotations

from pathlib import Path

from core.exceptions import AppException, ERROR_PROCESS_FAILED
from utils.config import get_settings


class SegmentService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def segment_person(self, input_path: str, output_path: str) -> str:
        if not self.settings.segmentation_enabled:
            raise AppException('当前版本未启用高级抠图能力', ERROR_PROCESS_FAILED, 503)

        try:
            from rembg import remove
        except Exception as exc:
            raise AppException('抠图依赖未安装或初始化失败', ERROR_PROCESS_FAILED, 503) from exc

        input_bytes = Path(input_path).read_bytes()
        output_bytes = remove(input_bytes)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(output_bytes)
        return str(output_file)
