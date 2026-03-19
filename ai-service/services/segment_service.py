from pathlib import Path

import numpy as np
from PIL import Image

from core.exceptions import AppException, ERROR_PROCESS_FAILED


class SegmentService:
    @staticmethod
    def _fallback_segment(input_path: str, output_path: str) -> str:
        image = Image.open(input_path).convert('RGBA')
        rgba = np.array(image)
        rgb = rgba[:, :, :3].astype(np.int16)

        border_pixels = np.concatenate(
            [
                rgb[0, :, :],
                rgb[-1, :, :],
                rgb[:, 0, :],
                rgb[:, -1, :],
            ],
            axis=0,
        )
        bg_color = np.median(border_pixels, axis=0)
        distance = np.linalg.norm(rgb - bg_color, axis=2)
        alpha = np.where(distance > 35, 255, 0).astype(np.uint8)
        rgba[:, :, 3] = alpha

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode='RGBA').save(output_file)
        return str(output_file)

    def segment_person(self, input_path: str, output_path: str) -> str:
        try:
            from rembg import remove

            input_bytes = Path(input_path).read_bytes()
            output_bytes = remove(input_bytes)
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(output_bytes)
            return str(output_file)
        except ImportError:
            return self._fallback_segment(input_path, output_path)
        except Exception as exc:
            try:
                return self._fallback_segment(input_path, output_path)
            except Exception as fallback_exc:
                raise AppException('Background segmentation failed', ERROR_PROCESS_FAILED, 500) from fallback_exc
