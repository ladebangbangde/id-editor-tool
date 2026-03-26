import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.validation_service import ValidationService


class CompliancePriorityTests(unittest.TestCase):
    def setUp(self):
        self.validation_service = ValidationService()
        self.image_shape = (1200, 900, 3)

    def test_failed_compliance_reason_has_higher_priority_than_blur(self):
        self.validation_service.compliance_service.evaluate = lambda **_: {
            'status': 'failed',
            'code': 'EYE_OCCLUDED',
            'message': '检测到单眼闭合或严重眯眼，证件照审核高概率不通过',
            'details': [
                {
                    'code': 'EYE_OCCLUDED',
                    'message': '检测到单眼闭合或严重眯眼，证件照审核高概率不通过',
                    'status': 'failed',
                    'stage': 'eye_state',
                }
            ],
            'warnings': [],
            'keypointConfidences': {'eyes': 0.9},
        }
        outcome = self.validation_service.validate(
            self.image_shape,
            [{'x': 240, 'y': 180, 'width': 260, 'height': 260}],
            blur_score=0.1,
            image_bgr=np.zeros(self.image_shape, dtype=np.uint8),
            gray_image=np.zeros(self.image_shape[:2], dtype=np.uint8),
        )
        self.assertEqual(outcome.auditCode, 'EYE_OCCLUDED')
        self.assertEqual(outcome.reasons[0], 'EYE_OCCLUDED')

    def test_warning_compliance_marks_audit_warning(self):
        self.validation_service.compliance_service.evaluate = lambda **_: {
            'status': 'warning',
            'code': 'POSE_INVALID',
            'message': '头部姿态不够端正，建议微调到更标准的正脸角度',
            'details': [
                {
                    'code': 'POSE_INVALID',
                    'message': '头部姿态不够端正，建议微调到更标准的正脸角度',
                    'status': 'warning',
                    'stage': 'pose_estimation',
                }
            ],
            'warnings': [],
            'keypointConfidences': {'eyes': 0.9},
        }
        outcome = self.validation_service.validate(
            self.image_shape,
            [{'x': 240, 'y': 180, 'width': 260, 'height': 260}],
            blur_score=0.95,
            image_bgr=np.zeros(self.image_shape, dtype=np.uint8),
            gray_image=np.zeros(self.image_shape[:2], dtype=np.uint8),
        )
        self.assertEqual(outcome.auditStatus, 'warning')
        self.assertEqual(outcome.auditCode, 'COMPLIANCE_WARNING')


if __name__ == '__main__':
    unittest.main()
