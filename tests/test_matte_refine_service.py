import numpy as np
from PIL import Image

from app.services.matte_refine_service import MatteRefineService


def test_matte_refine_generates_refined_rgba_and_trimap() -> None:
    service = MatteRefineService()
    src = Image.new('RGB', (64, 64), 'white')
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[12:54, 20:44, :3] = 160
    rgba[12:54, 20:44, 3] = 255
    rgba[10:56, 18:46, 3] = np.maximum(rgba[10:56, 18:46, 3], 128)
    fg = Image.fromarray(rgba, mode='RGBA')

    service._estimate_alpha_cf = lambda rgb, trimap: trimap
    result = service.refine(src, fg)

    assert result.rgba.mode == 'RGBA'
    assert result.alpha.mode == 'L'
    assert result.trimap.mode == 'L'
    alpha_np = np.array(result.alpha)
    assert alpha_np.max() > 0
    assert alpha_np[32, 32] >= 120


def test_matte_refine_decontamination_respects_flag() -> None:
    service = MatteRefineService()
    service.settings.enable_foreground_decontamination = False
    src = Image.new('RGB', (32, 32), 'white')
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[6:28, 8:24, :3] = 150
    rgba[6:28, 8:24, 3] = 255
    fg = Image.fromarray(rgba, mode='RGBA')

    result = service.refine(src, fg)
    assert result.decontaminated_rgba is None
