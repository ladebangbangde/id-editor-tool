from __future__ import annotations

from PIL import Image

from core.exceptions import AppException, ERROR_NO_FACE_DETECTED
from services.detect_service import DetectService
from services.print_service import PrintService
from services.validation_service import ValidationService
from utils.file_utils import public_url_for_path, to_url_like_path
from utils.logger import get_logger


class GeneratePrintLayoutPipeline:
    def __init__(self):
        self.detect_service = DetectService()
        self.validation_service = ValidationService()
        self.print_service = PrintService()
        self.logger = get_logger(component='generate_print_layout_pipeline')

    def run(self, image_id: str, hd_image_path: str, layout_type: str) -> dict:
        pipeline_logger = self.logger.bind(image_id=image_id, hd_image_path=hd_image_path, layout_type=layout_type)
        pipeline_logger.info('starting print layout pipeline')
        detect_result = self.detect_service.detect(image_id=image_id, image_path=hd_image_path)
        validation_passed = getattr(detect_result, 'validationPassed', getattr(detect_result, 'passed', False))
        default_face_detected = validation_passed or bool(getattr(detect_result, 'primaryFaceBox', None))
        reasons = list(getattr(detect_result, 'reasons', []))
        if 'NO_FACE_DETECTED' not in reasons and reasons:
            default_face_detected = True
        face_detected = getattr(detect_result, 'faceDetected', getattr(detect_result, 'hasFace', default_face_detected))
        detect_payload = detect_result.to_dict() if hasattr(detect_result, 'to_dict') else None
        pipeline_logger.bind(face_detected=face_detected, validation_passed=validation_passed).info('validation stage finished for print layout')
        if not face_detected:
            pipeline_logger.warning('print layout pipeline aborted: no face detected')
            raise AppException('No face detected from HD image', ERROR_NO_FACE_DETECTED, 400, data=detect_payload)
        if not validation_passed:
            error_code, message = self.validation_service.build_generate_error(reasons)
            pipeline_logger.bind(error_code=error_code, reasons=','.join(reasons) if reasons else 'none').warning('print layout pipeline aborted by validation failure')
            raise AppException(message, error_code, 400, data=detect_payload)

        hd_image = Image.open(hd_image_path).convert('RGB')
        print_result = self.print_service.generate_layout(image_id=image_id, hd_image=hd_image, layout_type=layout_type)
        pipeline_logger.bind(print_path=print_result['printPath'], photo_count=print_result['photoCount']).info('print layout pipeline completed')
        return {
            'imageId': image_id,
            'hdPath': to_url_like_path(hd_image_path),
            'hdUrl': public_url_for_path(hd_image_path),
            'layoutType': print_result['layoutType'],
            'printPath': print_result['printPath'],
            'printUrl': print_result['printUrl'],
            'paperType': print_result['paperType'],
            'photoCount': print_result['photoCount'],
        }
