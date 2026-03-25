import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from core.exceptions import ERROR_FACE_OCCLUDED, ERROR_MULTIPLE_FACES_DETECTED, ERROR_NO_FACE_DETECTED
from services.face_postprocess_service import FacePostprocessService
from services.validation_service import ValidationService


class FaceValidationTestCase(unittest.TestCase):
    def setUp(self):
        self.validation_service = ValidationService()
        self.postprocess_service = FacePostprocessService()
        self.image_shape = (1200, 900, 3)

    def test_single_face_with_duplicate_and_small_false_positive_keeps_one_valid_face(self):
        faces = [
            {'c': 240, 'r': 180, 'width': 260, 'height': 260},
            {'c': 250, 'r': 190, 'width': 240, 'height': 240},
            {'c': 540, 'r': 250, 'width': 62, 'height': 62},
        ]

        outcome = self.validation_service.validate(self.image_shape, faces, blur_score=0.9)

        self.assertTrue(outcome.hasFace)
        self.assertEqual(outcome.faceCount, 1)
        self.assertNotIn(ERROR_MULTIPLE_FACES_DETECTED, outcome.reasons)
        self.assertEqual(outcome.primaryFaceBox, {'x': 240, 'y': 180, 'width': 260, 'height': 260})
        self.assertEqual(outcome.rawFaceCount, 3)
        self.assertIn('overlapped_duplicate', [item['reason'] for item in outcome.filteredOutReasons])
        self.assertIn('too_small_relative_to_primary', [item['reason'] for item in outcome.filteredOutReasons])

    def test_real_multi_face_scene_still_returns_multiple_faces(self):
        faces = [
            {'c': 120, 'r': 150, 'width': 220, 'height': 220},
            {'c': 470, 'r': 170, 'width': 210, 'height': 210},
        ]

        outcome = self.validation_service.validate(self.image_shape, faces, blur_score=0.95)

        self.assertTrue(outcome.hasFace)
        self.assertEqual(outcome.faceCount, 2)
        self.assertIn(ERROR_MULTIPLE_FACES_DETECTED, outcome.reasons)
        self.assertEqual(outcome.primaryFaceBox, {'x': 120, 'y': 150, 'width': 220, 'height': 220})

    def test_all_tiny_boxes_are_treated_as_no_face(self):
        faces = [
            {'c': 50, 'r': 60, 'width': 30, 'height': 30},
            {'c': 140, 'r': 80, 'width': 40, 'height': 45},
        ]

        outcome = self.validation_service.validate(self.image_shape, faces, blur_score=0.9)

        self.assertFalse(outcome.hasFace)
        self.assertEqual(outcome.faceCount, 0)
        self.assertIn(ERROR_NO_FACE_DETECTED, outcome.reasons)
        self.assertIsNone(outcome.primaryFaceBox)

    def test_postprocess_deduplicates_high_overlap_boxes(self):
        faces = [
            {'c': 200, 'r': 120, 'width': 280, 'height': 280},
            {'c': 215, 'r': 135, 'width': 270, 'height': 270},
        ]

        result = self.postprocess_service.face_box_postprocess(faces)

        self.assertEqual(result.rawFaceCount, 2)
        self.assertEqual(len(result.validFaces), 1)
        self.assertEqual(result.primaryFaceBox, {'x': 200, 'y': 120, 'width': 280, 'height': 280})
        self.assertEqual(result.filteredOutReasons[0]['reason'], 'overlapped_duplicate')

    def test_validation_returns_audit_result_structure(self):
        self.validation_service.compliance_service.evaluate = lambda **_: {
            'status': 'failed',
            'code': 'FACE_OCCLUDED',
            'message': '人脸存在明显遮挡，请露出双眼和完整面部后重试',
            'details': [
                {
                    'code': 'FACE_OCCLUDED',
                    'message': '人脸存在明显遮挡，请露出双眼和完整面部后重试',
                }
            ],
            'keypointConfidences': {'leftEye': 0.2, 'rightEye': 0.1, 'noseTip': 0.1, 'mouthCorners': 0.1},
        }
        outcome = self.validation_service.validate(
            self.image_shape,
            [{'x': 240, 'y': 180, 'width': 260, 'height': 260}],
            blur_score=0.95,
            image_bgr=self._dummy_image(self.image_shape),
            gray_image=self._dummy_image(self.image_shape[:2]),
        )
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.auditStatus, 'failed')
        self.assertEqual(outcome.auditCode, 'FACE_OCCLUDED')
        self.assertGreaterEqual(len(outcome.auditDetails), 1)
        self.assertIn('leftEye', outcome.keypointConfidences)

    def test_generate_error_message_prefers_hand_or_hat_copy(self):
        code, message = self.validation_service.build_generate_error(['HAND_OCCLUSION'])
        self.assertEqual(code, 'HAND_OCCLUSION')
        self.assertEqual(message, '检测到帽子或手部遮挡，不符合证件照要求')

    def test_regression_case_previously_warning_now_not_failed(self):
        self.validation_service.compliance_service.evaluate = lambda **_: {
            'status': 'warning',
            'code': 'COMPLIANCE_WARNING',
            'message': '合规审核通过（存在提醒项）',
            'details': [
                {
                    'code': 'COMPLIANCE_WARNING',
                    'message': '关键点低置信度(非关键阻断): nose,mouth',
                    'status': 'warning',
                    'stage': 'keypoint_detection',
                }
            ],
            'warnings': ['关键点分类器缺失: nose,mouth'],
            'keypointConfidences': {'eyes': 0.81, 'nose': 0.0, 'mouth': 0.0},
        }
        outcome = self.validation_service.validate(
            self.image_shape,
            [{'x': 240, 'y': 180, 'width': 260, 'height': 260}],
            blur_score=0.95,
            image_bgr=self._dummy_image(self.image_shape),
            gray_image=self._dummy_image(self.image_shape[:2]),
        )
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.auditStatus, 'passed')
        warning_details = [item for item in outcome.auditDetails if item.get('status') == 'warning']
        self.assertGreaterEqual(len(warning_details), 1)

    def test_regression_case_still_blocks_clear_occlusion(self):
        self.validation_service.compliance_service.evaluate = lambda **_: {
            'status': 'failed',
            'code': ERROR_FACE_OCCLUDED,
            'message': '人脸存在明显遮挡，请露出双眼和完整面部后重试',
            'details': [
                {
                    'code': ERROR_FACE_OCCLUDED,
                    'message': '人脸存在明显遮挡，请露出双眼和完整面部后重试',
                    'status': 'failed',
                    'stage': 'compliance_occlusion',
                }
            ],
            'warnings': [],
            'keypointConfidences': {'eyes': 0.21, 'nose': 0.2, 'mouth': 0.18},
        }
        outcome = self.validation_service.validate(
            self.image_shape,
            [{'x': 240, 'y': 180, 'width': 260, 'height': 260}],
            blur_score=0.95,
            image_bgr=self._dummy_image(self.image_shape),
            gray_image=self._dummy_image(self.image_shape[:2]),
        )
        self.assertFalse(outcome.passed)
        self.assertIn(ERROR_FACE_OCCLUDED, outcome.reasons)

    @staticmethod
    def _dummy_image(shape):
        import numpy as np

        return np.zeros(shape, dtype=np.uint8)


if __name__ == '__main__':
    unittest.main()
