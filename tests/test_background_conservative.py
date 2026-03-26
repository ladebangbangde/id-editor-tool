import numpy as np
from PIL import Image

from app.services.background import BackgroundService


def test_background_alpha_keeps_lower_foreground_density():
    service = BackgroundService()
    rgba = np.zeros((120, 80, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[:50, :, 3] = 255
    rgba[50:, :, 3] = 128
    fg = Image.fromarray(rgba, mode='RGBA')

    alpha = np.array(service._build_conservative_alpha(fg))
    assert int(alpha[95, 40]) >= 128
