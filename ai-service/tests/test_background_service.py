import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.background_service import BackgroundService


class ConservativeCompositeTests(unittest.TestCase):
    def test_lower_body_alpha_is_not_reduced_after_protection(self):
        service = BackgroundService()
        rgba = np.zeros((120, 80, 4), dtype=np.uint8)
        rgba[:, :, :3] = 120
        rgba[:50, :, 3] = 255
        rgba[50:, :, 3] = 128
        fg = Image.fromarray(rgba, mode='RGBA')

        hardened = service._build_conservative_alpha(fg)
        self.assertGreaterEqual(int(hardened[90, 40]), int(rgba[90, 40, 3]))


if __name__ == '__main__':
    unittest.main()
