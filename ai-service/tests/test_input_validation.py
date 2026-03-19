import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.exceptions import (
    AppException,
    ERROR_FACE_TOO_SMALL,
    ERROR_IMAGE_TOO_BLURRY,
    ERROR_MULTIPLE_FACES_DETECTED,
    ERROR_NO_FACE_DETECTED,
    ERROR_POSE_INVALID,
)
from pipeline.generate_id_photo import GenerateIdPhotoPipeline
from pipeline.generate_print_layout import GeneratePrintLayoutPipeline
from services.validation_service import ValidationService


class ValidationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ValidationService()
        self.image_shape = (1200, 900, 3)
        self.good_face = {'c': 300, 'r': 180, 'width': 320, 'height': 420}

    def test_single_clear_face_passes(self):
        outcome = self.service.validate(self.image_shape, [self.good_face], blur_score=0.9)
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.reasons, [])
        self.assertEqual(outcome.message, '图片符合证件照制作要求')
        self.assertEqual(outcome.imageWidth, 900)
        self.assertEqual(outcome.imageHeight, 1200)
        self.assertEqual(outcome.primaryFaceBox['width'], 320)

    def test_multiple_faces_rejected(self):
        outcome = self.service.validate(self.image_shape, [self.good_face, self.good_face], blur_score=0.9)
        self.assertFalse(outcome.passed)
        self.assertTrue(outcome.hasFace)
        self.assertEqual(outcome.faceCount, 2)
        self.assertIn(ERROR_MULTIPLE_FACES_DETECTED, outcome.reasons)

    def test_no_face_rejected(self):
        outcome = self.service.validate(self.image_shape, [], blur_score=0.9)
        self.assertFalse(outcome.passed)
        self.assertFalse(outcome.hasFace)
        self.assertIn(ERROR_NO_FACE_DETECTED, outcome.reasons)

    def test_blurry_face_rejected(self):
        outcome = self.service.validate(self.image_shape, [self.good_face], blur_score=0.1)
        self.assertFalse(outcome.passed)
        self.assertIn(ERROR_IMAGE_TOO_BLURRY, outcome.reasons)

    def test_small_face_rejected(self):
        small_face = {'c': 360, 'r': 260, 'width': 80, 'height': 90}
        outcome = self.service.validate(self.image_shape, [small_face], blur_score=0.9)
        self.assertFalse(outcome.passed)
        self.assertIn(ERROR_FACE_TOO_SMALL, outcome.reasons)

    def test_pose_invalid_rejected(self):
        side_face = {'c': 20, 'r': 180, 'width': 500, 'height': 420}
        outcome = self.service.validate(self.image_shape, [side_face], blur_score=0.9)
        self.assertFalse(outcome.passed)
        self.assertIn(ERROR_POSE_INVALID, outcome.reasons)


class PipelineValidationTests(unittest.TestCase):
    def _write_temp_image(self, path: Path):
        Image.new('RGB', (600, 800), color='white').save(path)

    def test_generate_rejects_failed_validation(self):
        pipeline = GenerateIdPhotoPipeline()
        failed = SimpleNamespace(passed=False, reasons=[ERROR_MULTIPLE_FACES_DETECTED], primaryFaceBox=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            original = Path(tmpdir) / 'input.jpg'
            self._write_temp_image(original)
            with patch.object(pipeline.detect_service, 'detect', return_value=failed):
                with self.assertRaises(AppException) as ctx:
                    pipeline.run(
                        {
                            'imageId': 'multi-face',
                            'originalImagePath': str(original),
                            'sourceType': 'scene',
                            'sceneKey': 'passport',
                            'backgroundColor': 'white',
                            'beautyEnabled': False,
                            'printLayoutType': None,
                        }
                    )
        self.assertEqual(ctx.exception.error_code, ERROR_MULTIPLE_FACES_DETECTED)
        self.assertEqual(ctx.exception.data, None)

    def test_generate_accepts_passed_validation(self):
        pipeline = GenerateIdPhotoPipeline()
        passed = SimpleNamespace(passed=True, reasons=[], primaryFaceBox={'x': 10, 'y': 10, 'width': 200, 'height': 260})
        with tempfile.TemporaryDirectory() as tmpdir:
            original = Path(tmpdir) / 'input.jpg'
            hd_output = Path(tmpdir) / 'out_hd.jpg'
            self._write_temp_image(original)
            with (
                patch.object(pipeline.detect_service, 'detect', return_value=passed),
                patch.object(pipeline.segment_service, 'segment_person', return_value=None),
                patch.object(pipeline.background_service, 'apply_background', return_value={'image': Image.new('RGB', (600, 800), color='white'), 'method': 'segmentation_composite', 'note': None}),
                patch.object(pipeline.crop_service, 'crop_to_size', return_value=SimpleNamespace(image=Image.new('RGB', (600, 800), color='white'), cropBox={'x': 0, 'y': 0, 'width': 600, 'height': 800}, targetWidth=390, targetHeight=567, headRatio=0.6, method='face_guided_crop')),
                patch.object(pipeline.enhance_service, 'enhance', return_value={'image': Image.new('RGB', (600, 800), color='white'), 'appliedOperations': ['keep_original']}),
                patch.object(pipeline.preview_builder, 'build_preview', return_value={'previewPath': 'uploads/preview/pass_preview.jpg', 'previewUrl': '/uploads/preview/pass_preview.jpg'}),
                patch.object(pipeline.quality_service, 'evaluate_details', return_value={'qualityStatus': 'passed', 'qualityMessage': '质量检测通过', 'resolutionTooLow': False, 'clarityInsufficient': False, 'suitableForIdPhoto': True}),
                patch.object(pipeline.print_service, 'generate_layout', return_value={'printPath': 'uploads/print/pass_print.jpg', 'printUrl': '/uploads/print/pass_print.jpg', 'layoutType': 'six', 'paperType': '6inch', 'photoCount': 6}),
                patch('pipeline.generate_id_photo.build_output_path', return_value=str(hd_output)),
                patch('pipeline.generate_id_photo.save_pil_image', return_value=None),
                patch('pipeline.generate_id_photo.to_url_like_path', side_effect=lambda value: 'uploads/hd/pass_hd.jpg' if 'out_hd' in str(value) else 'uploads/original/input.jpg'),
                patch('pipeline.generate_id_photo.public_url_for_path', side_effect=lambda value: '/uploads/hd/pass_hd.jpg' if 'out_hd' in str(value) else '/uploads/original/input.jpg'),
            ):
                result = pipeline.run(
                    {
                        'imageId': 'passed',
                        'originalImagePath': str(original),
                        'sourceType': 'scene',
                        'sceneKey': 'passport',
                        'backgroundColor': 'white',
                        'beautyEnabled': False,
                        'printLayoutType': 'six',
                    }
                )
        self.assertEqual(result['imageId'], 'passed')
        self.assertEqual(result['hdUrl'], '/uploads/hd/pass_hd.jpg')
        self.assertEqual(result['printUrl'], '/uploads/print/pass_print.jpg')
        self.assertEqual(result['previewPath'], 'uploads/preview/pass_preview.jpg')

    def test_print_rejects_failed_validation(self):
        pipeline = GeneratePrintLayoutPipeline()
        failed = SimpleNamespace(passed=False, reasons=[ERROR_NO_FACE_DETECTED])
        with tempfile.TemporaryDirectory() as tmpdir:
            hd_image = Path(tmpdir) / 'input_hd.jpg'
            self._write_temp_image(hd_image)
            with patch.object(pipeline.detect_service, 'detect', return_value=failed):
                with self.assertRaises(AppException) as ctx:
                    pipeline.run('no-face', str(hd_image), 'six')
        self.assertEqual(ctx.exception.error_code, ERROR_NO_FACE_DETECTED)
        self.assertEqual(ctx.exception.data, None)

    def test_print_accepts_passed_validation(self):
        pipeline = GeneratePrintLayoutPipeline()
        passed = SimpleNamespace(passed=True, reasons=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            hd_image = Path(tmpdir) / 'input_hd.jpg'
            self._write_temp_image(hd_image)
            with (
                patch.object(pipeline.detect_service, 'detect', return_value=passed),
                patch.object(pipeline.print_service, 'generate_layout', return_value={'printPath': 'uploads/print/pass_print.jpg', 'printUrl': '/uploads/print/pass_print.jpg', 'layoutType': 'six', 'paperType': '6inch', 'photoCount': 6}),
            ):
                result = pipeline.run('passed', str(hd_image), 'six')
        self.assertEqual(result['printUrl'], '/uploads/print/pass_print.jpg')


if __name__ == '__main__':
    unittest.main()
