from __future__ import annotations

from pathlib import Path

from core.exceptions import AppException, ERROR_NO_FACE_DETECTED
from constants.photo_sizes import PHOTO_SIZE_TEMPLATES, build_custom_template
from pipeline.build_preview import PreviewBuilder
from services.background_service import BackgroundService
from services.crop_service import CropService
from services.detect_service import DetectService
from services.enhance_service import EnhanceService
from services.print_service import PrintService
from services.quality_service import QualityService
from services.segment_service import SegmentService
from services.validation_service import ValidationService
from utils.config import get_settings
from utils.file_utils import build_output_path, public_url_for_path, to_url_like_path
from utils.image_utils import save_pil_image
from utils.logger import get_logger


class GenerateIdPhotoPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.detect_service = DetectService()
        self.segment_service = SegmentService()
        self.background_service = BackgroundService()
        self.crop_service = CropService()
        self.enhance_service = EnhanceService()
        self.quality_service = QualityService()
        self.validation_service = ValidationService()
        self.print_service = PrintService()
        self.preview_builder = PreviewBuilder()
        self.logger = get_logger(component='generate_id_photo_pipeline')

    def _resolve_template(self, source_type: str, scene_key: str | None, custom_w: int | None, custom_h: int | None):
        if source_type == 'scene':
            if scene_key not in PHOTO_SIZE_TEMPLATES:
                raise AppException(f'Unknown sceneKey: {scene_key}', 'INVALID_ARGUMENT', 400)
            return PHOTO_SIZE_TEMPLATES[scene_key]
        return build_custom_template(custom_w, custom_h, dpi=self.settings.jpeg_dpi)

    @staticmethod
    def _final_output_type(print_result: dict | None) -> str:
        return 'print_layout' if print_result else 'id_photo'

    def run(self, payload: dict) -> dict:
        image_id = payload['imageId']
        original_image_path = payload['originalImagePath']
        pipeline_logger = self.logger.bind(
            image_id=image_id,
            source_type=payload.get('sourceType'),
            scene_key=payload.get('sceneKey'),
            background_color=payload.get('backgroundColor'),
            original_image_path=original_image_path,
        )
        pipeline_logger.info('starting id photo generation pipeline')

        if not Path(original_image_path).exists():
            pipeline_logger.warning('original image path does not exist')
            raise AppException(f'original image not found: {original_image_path}', 'FILE_NOT_FOUND', 404)

        detect_result = self.detect_service.detect(image_id=image_id, image_path=original_image_path)
        validation_passed = getattr(detect_result, 'validationPassed', getattr(detect_result, 'passed', False))
        default_face_detected = validation_passed or bool(getattr(detect_result, 'primaryFaceBox', None))
        reasons = list(getattr(detect_result, 'reasons', []))
        if 'NO_FACE_DETECTED' not in reasons and reasons:
            default_face_detected = True
        face_detected = getattr(detect_result, 'faceDetected', getattr(detect_result, 'hasFace', default_face_detected))
        detect_payload = detect_result.to_dict() if hasattr(detect_result, 'to_dict') else None
        pipeline_logger.bind(
            face_detected=face_detected,
            validation_passed=validation_passed,
            reasons=','.join(reasons) if reasons else 'none',
        ).info('detection and validation stage completed')
        if not face_detected:
            pipeline_logger.warning('pipeline aborted: no face detected')
            raise AppException('No face detected from source image', ERROR_NO_FACE_DETECTED, 400, data=detect_payload)
        if not validation_passed:
            error_code, message = self.validation_service.build_generate_error(reasons)
            pipeline_logger.bind(error_code=error_code).warning('pipeline aborted by validation failure')
            raise AppException(message, error_code, 400, data=detect_payload)

        size_tpl = self._resolve_template(
            source_type=payload['sourceType'],
            scene_key=payload.get('sceneKey'),
            custom_w=payload.get('customWidthMm'),
            custom_h=payload.get('customHeightMm'),
        )
        pipeline_logger.bind(width_mm=size_tpl.widthMm, height_mm=size_tpl.heightMm, pixel_width=size_tpl.pixelWidth, pixel_height=size_tpl.pixelHeight).info(
            'resolved target photo template'
        )

        process_notes: list[str] = []
        seg_output_path = build_output_path('temp', f'{image_id}_segmented.png')
        segmentation_succeeded = False
        whether_fallback_used = False
        try:
            self.segment_service.segment_person(original_image_path, seg_output_path)
            segmentation_succeeded = True
            background_result = self.background_service.apply_background(
                transparent_png_path=seg_output_path,
                background_color=payload['backgroundColor'],
                preview_path=seg_output_path,
            )
            pipeline_logger.bind(segmentation_output=seg_output_path).info('segmentation and background stage completed')
        except Exception as exc:
            whether_fallback_used = True
            process_notes.append(f'segmentation fallback: {exc}')
            pipeline_logger.bind(error=str(exc)).warning('segmentation stage failed, switching to fallback mode')
            background_result = self.background_service.fallback_original(
                image_path=original_image_path,
                background_color=payload['backgroundColor'],
                reason='抠图未启用或执行失败，已回退到原图继续处理',
            )

        crop_result = self.crop_service.crop_to_size(
            background_result['image'],
            size_tpl.pixelWidth,
            size_tpl.pixelHeight,
            face_box=detect_result.primaryFaceBox,
        )
        process_notes.append(crop_result.method)
        pipeline_logger.bind(crop_method=crop_result.method, head_ratio=crop_result.headRatio).info('crop stage completed')
        if background_result.get('note'):
            process_notes.append(background_result['note'])

        enhance_result = self.enhance_service.enhance(crop_result.image, bool(payload.get('beautyEnabled', False)))
        enhanced = enhance_result['image']
        pipeline_logger.bind(applied_operations=','.join(enhance_result['appliedOperations'])).info('enhance stage completed')

        hd_output_path = build_output_path('hd', f'{image_id}_hd.jpg')
        save_pil_image(enhanced, hd_output_path, quality=self.settings.hd_quality, dpi=self.settings.jpeg_dpi)
        pipeline_logger.bind(hd_output_path=hd_output_path).info('hd output saved')

        preview_result = self.preview_builder.build_preview(image_id=image_id, hd_image=enhanced)
        pipeline_logger.bind(preview_path=preview_result['previewPath']).info('preview output saved')
        source_width = getattr(detect_result, 'imageWidth', enhanced.size[0])
        source_height = getattr(detect_result, 'imageHeight', enhanced.size[1])
        quality_details = self.quality_service.evaluate_details(
            enhanced,
            source_size=(source_width, source_height),
            expected_output_size=(size_tpl.pixelWidth, size_tpl.pixelHeight),
            face_box=detect_result.primaryFaceBox,
            blur_score=getattr(detect_result, 'blurScore', None),
        )
        pipeline_logger.bind(
            quality_status=quality_details['qualityStatus'],
            suitable=quality_details['suitableForIdPhoto'],
            likely_upscaled=quality_details['likelyUpscaled'],
        ).info('quality evaluation completed')

        print_result = None
        layout_type = payload.get('printLayoutType')
        if layout_type:
            print_result = self.print_service.generate_layout(image_id=image_id, hd_image=enhanced, layout_type=layout_type)
            pipeline_logger.bind(print_path=print_result['printPath'], layout_type=layout_type).info('optional print layout generated')

        if not self.settings.save_intermediate:
            Path(seg_output_path).unlink(missing_ok=True)
            pipeline_logger.bind(temp_path=seg_output_path).debug('temporary segmentation output removed')

        result = {
            'imageId': image_id,
            'originalImagePath': to_url_like_path(original_image_path),
            'originalImageUrl': public_url_for_path(original_image_path),
            'previewPath': preview_result['previewPath'],
            'previewUrl': preview_result['previewUrl'],
            'hdPath': to_url_like_path(hd_output_path),
            'hdUrl': public_url_for_path(hd_output_path),
            'printPath': print_result['printPath'] if print_result else None,
            'printUrl': print_result['printUrl'] if print_result else None,
            'backgroundColor': payload['backgroundColor'],
            'method': background_result['method'],
            'widthMm': size_tpl.widthMm,
            'heightMm': size_tpl.heightMm,
            'pixelWidth': size_tpl.pixelWidth,
            'pixelHeight': size_tpl.pixelHeight,
            'qualityStatus': quality_details['qualityStatus'],
            'qualityMessage': quality_details['qualityMessage'],
            'sourceResolutionTooLow': quality_details['sourceResolutionTooLow'],
            'outputSizeIsStandard': quality_details['outputSizeIsStandard'],
            'likelyUpscaled': quality_details['likelyUpscaled'],
            'cropBox': crop_result.cropBox,
            'targetWidth': crop_result.targetWidth,
            'targetHeight': crop_result.targetHeight,
            'headRatio': crop_result.headRatio,
            'appliedOperations': enhance_result['appliedOperations'],
            'processNotes': process_notes,
            'whetherFallbackUsed': whether_fallback_used,
            'segmentationSucceeded': segmentation_succeeded,
            'finalOutputType': self._final_output_type(print_result),
            'canDirectlyUseForRegistration': quality_details['suitableForIdPhoto'] and validation_passed,
            'layoutType': print_result['layoutType'] if print_result else layout_type,
            'paperType': print_result['paperType'] if print_result else None,
            'photoCount': print_result['photoCount'] if print_result else None,
        }
        pipeline_logger.bind(
            method=result['method'],
            fallback_used=result['whetherFallbackUsed'],
            final_output_type=result['finalOutputType'],
        ).info('id photo generation pipeline completed successfully')
        return result
