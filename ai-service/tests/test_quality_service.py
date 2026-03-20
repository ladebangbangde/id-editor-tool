import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants.status import QUALITY_STATUS_PASSED, QUALITY_STATUS_WARNING
from services.quality_service import QualityService


class QualityServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = QualityService()

    def test_standard_passport_output_can_pass_when_source_is_good(self):
        source = Image.new('RGB', (1200, 1600), color='white')
        output = Image.new('RGB', (390, 567), color='white')

        result = self.service.evaluate_details(
            output,
            source_image=source,
            expected_output_size=(390, 567),
            face_box={'x': 300, 'y': 220, 'width': 520, 'height': 760},
            blur_score=0.92,
        )

        self.assertEqual(result['qualityStatus'], QUALITY_STATUS_PASSED)
        self.assertTrue(result['outputSizeIsStandard'])
        self.assertFalse(result['likelyUpscaled'])
        self.assertTrue(result['suitableForIdPhoto'])

    def test_warning_comes_from_small_source_and_upscale_risk_not_small_standard_output(self):
        output = Image.new('RGB', (390, 567), color='white')

        result = self.service.evaluate_details(
            output,
            source_size=(410, 590),
            expected_output_size=(390, 567),
            face_box={'x': 130, 'y': 110, 'width': 120, 'height': 150},
            blur_score=0.8,
        )

        self.assertEqual(result['qualityStatus'], QUALITY_STATUS_WARNING)
        self.assertTrue(result['clarityInsufficient'])
        self.assertTrue(result['likelyUpscaled'])
        self.assertFalse(result['suitableForIdPhoto'])

    def test_small_original_without_output_context_is_warning_not_failed(self):
        original = Image.new('RGB', (390, 567), color='white')

        result = self.service.evaluate_details(original)

        self.assertEqual(result['qualityStatus'], QUALITY_STATUS_WARNING)
        self.assertFalse(result['resolutionTooLow'])
        self.assertTrue(result['clarityInsufficient'])


if __name__ == '__main__':
    unittest.main()
