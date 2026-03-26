import numpy as np
from PIL import Image

from app.services.output_quality_service import OutputQualityService


def _solid_rgba(width: int, height: int, alpha: int = 255) -> Image.Image:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, :3] = 180
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode='RGBA')


def test_output_quality_detects_face_color_pollution() -> None:
    service = OutputQualityService()
    src = Image.new('RGB', (200, 260), (190, 170, 160))
    out = np.zeros((260, 200, 3), dtype=np.uint8)
    out[:, :] = (190, 170, 160)
    out[70:180, 60:140] = (255, 120, 120)
    output = Image.fromarray(out, mode='RGB')
    fg = _solid_rgba(200, 260, alpha=255)

    result = service.evaluate(
        source_image=src,
        output_image=output,
        foreground_rgba=fg,
        face_box={'x': 55, 'y': 55, 'width': 90, 'height': 130},
        background_color='red',
    )

    assert result.status == 'FAIL'
    assert 'FACE_COLOR_POLLUTION' in result.reason_codes


def test_output_quality_pass_for_clean_output() -> None:
    service = OutputQualityService()
    src = Image.new('RGB', (200, 260), (190, 170, 160))
    output = Image.new('RGB', (200, 260), (191, 171, 161))
    fg = _solid_rgba(200, 260, alpha=255)

    result = service.evaluate(
        source_image=src,
        output_image=output,
        foreground_rgba=fg,
        face_box={'x': 55, 'y': 55, 'width': 90, 'height': 130},
        background_color='white',
    )

    assert result.status == 'PASS'
    assert result.reason_codes == []
