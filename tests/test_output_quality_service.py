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


def test_output_quality_cloth_pollution_low_risk_can_pass() -> None:
    service = OutputQualityService()
    service.settings.enable_cloth_pollution_check = True
    src = Image.new('RGB', (220, 280), (210, 205, 200))
    out = np.zeros((280, 220, 3), dtype=np.uint8)
    out[:, :] = (210, 205, 200)
    out[180:230, 85:135] = (150, 160, 235)
    output = Image.fromarray(out, mode='RGB')
    fg = _solid_rgba(220, 280, alpha=255)

    result = service.evaluate(
        source_image=src,
        output_image=output,
        foreground_rgba=fg,
        face_box={'x': 70, 'y': 45, 'width': 80, 'height': 105},
        background_color='blue',
    )

    assert result.status == 'PASS'
    assert 'CLOTH_COLOR_POLLUTION' not in result.warnings


def test_output_quality_detects_cloth_color_pollution_fail() -> None:
    service = OutputQualityService()
    service.settings.enable_cloth_pollution_check = True
    src = Image.new('RGB', (220, 280), (220, 215, 210))
    out = np.zeros((280, 220, 3), dtype=np.uint8)
    out[:, :] = (220, 215, 210)
    out[130:275, 20:210] = (230, 40, 40)
    output = Image.fromarray(out, mode='RGB')
    fg = _solid_rgba(220, 280, alpha=255)

    result = service.evaluate(
        source_image=src,
        output_image=output,
        foreground_rgba=fg,
        face_box={'x': 70, 'y': 45, 'width': 80, 'height': 105},
        background_color='red',
    )

    assert result.status == 'FAIL'
    assert 'CLOTH_COLOR_POLLUTION' in result.reason_codes


def test_output_quality_detects_hair_gap_background_residue() -> None:
    service = OutputQualityService()
    src = Image.new('RGB', (220, 280), (200, 195, 190))
    out = np.zeros((280, 220, 3), dtype=np.uint8)
    out[:, :] = (200, 195, 190)
    out[20:120, 55:165] = (70, 65, 60)
    for y in range(40, 95, 10):
        for x in range(75, 150, 12):
            out[y:y + 2, x:x + 2] = (245, 245, 245)
    output = Image.fromarray(out, mode='RGB')

    fg = np.zeros((280, 220, 4), dtype=np.uint8)
    fg[:, :, :3] = 120
    fg[:, :, 3] = 255
    foreground = Image.fromarray(fg, mode='RGBA')

    result = service.evaluate(
        source_image=src,
        output_image=output,
        foreground_rgba=foreground,
        face_box={'x': 70, 'y': 65, 'width': 80, 'height': 110},
        background_color='red',
    )

    assert 'HAIR_GAP_BACKGROUND_RESIDUE' in (result.reason_codes + result.warnings)


def test_output_quality_detects_border_background_residue() -> None:
    service = OutputQualityService()
    src = Image.new('RGB', (220, 280), (240, 240, 240))
    out = np.zeros((280, 220, 3), dtype=np.uint8)
    out[:, :] = (67, 142, 219)
    out[:, :20] = (245, 245, 245)
    out[:, -20:] = (245, 245, 245)
    output = Image.fromarray(out, mode='RGB')
    fg = _solid_rgba(220, 280, alpha=255)

    result = service.evaluate(
        source_image=src,
        output_image=output,
        foreground_rgba=fg,
        face_box={'x': 70, 'y': 65, 'width': 80, 'height': 110},
        background_color='blue',
    )

    assert 'BORDER_BACKGROUND_RESIDUE' in result.reason_codes
