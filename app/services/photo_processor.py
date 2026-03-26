from functools import lru_cache
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import (
    BadLightingError,
    BadCompositionError,
    EyeOccludedError,
    FaceOccludedError,
    HandOcclusionError,
    HeadAccessoryError,
    InvalidArgumentError,
    InvalidImageError,
    InvalidPoseError,
    LandmarkUnstableError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.core.logger import get_logger
from app.schemas.common import FileInfo, SizeInfo
from app.schemas.detect import DetectData
from app.schemas.generate import GenerateData
from app.schemas.layout import LayoutData
from app.services.background import BackgroundService
from app.services.cropper import CropperService
from app.services.enhancer import EnhancerService
from app.services.face_detection import FaceDetectionResult, FaceDetectionService
from app.services.layout import LayoutService
from app.services.matte_refine_service import MatteRefineService
from app.services.output_quality_service import OutputQualityService
from app.services.segmentation import SegmentationService
from app.services.specs import PhotoSpec, get_photo_spec
from app.services.storage import StorageService
from app.utils.file_naming import build_task_id
from app.utils.image_io import load_image_from_bytes, load_image_from_path, resolve_input_path, save_image
from app.utils.validators import validate_content_size, validate_upload

logger = get_logger(__name__)


class PhotoProcessor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage = StorageService()
        self.detector = FaceDetectionService()
        self.segmenter = SegmentationService()
        self.background = BackgroundService()
        self.cropper = CropperService()
        self.enhancer = EnhancerService()
        self.matte_refiner = MatteRefineService()
        self.output_quality = OutputQualityService()
        self.layout_service = LayoutService()

    async def read_upload(self, file: UploadFile) -> tuple[bytes, Image.Image]:
        validate_upload(file)
        logger.info('Received upload filename=%s', file.filename)
        content = await file.read()
        validate_content_size(content)
        if not content:
            raise InvalidImageError('Uploaded file is empty')
        image = load_image_from_bytes(content)
        return content, image

    def read_image_path(self, path_or_url: str) -> tuple[Path, Image.Image]:
        resolved_path = resolve_input_path(path_or_url)
        logger.info('Reading shared image path=%s', resolved_path)
        return resolved_path, load_image_from_path(resolved_path)

    def _build_detect_data(self, result: FaceDetectionResult) -> DetectData:
        compliance_status = self._to_compliance_status(result.status)
        compliance_message = self._compliance_message(compliance_status)
        compliance_details = [
            {
                'code': issue.code,
                'message': issue.message,
                'severity': issue.severity,
            }
            for issue in result.issues
            if issue.severity in {'WARNING', 'FAIL'}
        ]
        return DetectData(
            hasFace=result.has_face,
            faceCount=result.face_count,
            width=result.width,
            height=result.height,
            pass_=result.can_generate,
            status=result.status,
            resultLevel=result.result_level,
            canGenerate=result.can_generate,
            reasons=[
                {
                    'code': reason.code,
                    'title': reason.title,
                    'detail': reason.detail,
                }
                for reason in result.reasons
            ],
            suggestions=result.suggestions,
            reasonCodes=result.reason_codes,
            warnings=result.warnings,
            warningCodes=result.warning_codes,
            issues=[
                {
                    'code': issue.code,
                    'message': issue.message,
                    'severity': issue.severity,
                }
                for issue in result.issues
            ],
            faceBoxes=result.face_boxes,
            recommended=result.recommended,
            warning='；'.join(result.warnings) if result.warnings else None,
            blurScore=result.blur_score,
            occlusionDetected=result.occlusion_detected,
            occlusionAreas=result.occlusion_areas,
            poseAccepted=result.pose_accepted,
            landmarkStable=result.landmark_stable,
            compositionAccepted=result.composition_accepted,
            metrics=result.metrics,
            primaryIssue=result.primary_issue,
            primaryMessage=result.primary_message,
            secondaryWarnings=result.secondary_warnings,
            qualityStatus=result.status,
            qualityMessage=(
                result.quality_message
                if compliance_status == 'passed'
                else compliance_message
            ),
            processStatus='success',
            processMessage='图片检测流程已完成',
            complianceStatus=compliance_status,
            complianceMessage=compliance_message,
            complianceDetails=compliance_details,
        )

    @staticmethod
    def _to_compliance_status(status: str) -> str:
        mapping = {
            'PASS': 'passed',
            'WARNING': 'warning',
            'FAIL': 'failed',
        }
        return mapping.get(status, 'warning')

    @staticmethod
    def _compliance_message(compliance_status: str) -> str:
        if compliance_status == 'passed':
            return '满足证件照合规要求'
        if compliance_status == 'warning':
            return '图片已生成，但存在合规风险，不建议直接用于正式证件照提交'
        return '图片已生成，但不符合证件照规范，不建议用于正式提交'

    def detect(self, image: Image.Image) -> DetectData:
        return self._build_detect_data(self.detector.detect(image))

    def _raise_detect_failure(self, detect_result: FaceDetectionResult) -> None:
        details = self._build_detect_data(detect_result).model_dump(by_alias=True)
        code = detect_result.reason_codes[0] if detect_result.reason_codes else 'NO_FACE_DETECTED'
        message = '当前照片暂不适合进入证件照处理流程'
        if code == 'NO_FACE_DETECTED':
            raise NoFaceDetectedError(message, details)
        if code == 'MULTIPLE_FACES_DETECTED':
            raise MultipleFacesDetectedError(message, details)
        if code in {'SEVERE_POSE', 'INVALID_POSE'}:
            raise InvalidPoseError(message, details)
        if code in {'IMAGE_TOO_BLURRY', 'LANDMARK_UNSTABLE'}:
            raise LandmarkUnstableError(message, details)
        if code in {'HEAD_SHOULDER_INCOMPLETE', 'FACE_RATIO_INVALID', 'BAD_COMPOSITION'}:
            raise BadCompositionError(message, details)
        if code in {'EXTREME_LIGHTING', 'BAD_LIGHTING'}:
            raise BadLightingError(message, details)
        if code == 'RESOLUTION_TOO_LOW':
            raise InvalidImageError(message, details)
        if code == 'NOT_SUITABLE_PORTRAIT':
            raise InvalidImageError(message, details)
        if code == 'HEAD_ACCESSORY':
            raise HeadAccessoryError(message, details)
        if code == 'HAND_OCCLUSION':
            raise HandOcclusionError(message, details)
        if code in {'EYE_OCCLUDED', 'WINK_EXPRESSION'}:
            raise EyeOccludedError(message, details)
        if code == 'FACE_OCCLUDED':
            raise FaceOccludedError(message, details)
        raise InvalidImageError(message, details)

    def _size_info(self, spec: PhotoSpec) -> SizeInfo:
        return SizeInfo(
            key=spec.key,
            name=spec.name,
            widthPx=spec.width_px,
            heightPx=spec.height_px,
            widthMm=spec.width_mm,
            heightMm=spec.height_mm,
        )

    def _file_info(self, path: Path) -> FileInfo:
        stored = self.storage.stored_file(path)
        return FileInfo(path=str(stored.path.resolve()), url=stored.url)

    def _build_preview_image(self, hd_image: Image.Image, max_long_side: int = 480, scale_cap: float = 0.72) -> tuple[Image.Image, int]:
        preview_image = hd_image.copy()
        width, height = preview_image.size
        long_side = max(width, height)
        scale = min(max_long_side / max(long_side, 1), scale_cap, 1.0)
        target_width = max(160, int(round(width * scale)))
        target_height = max(160, int(round(height * scale)))
        if target_width != width or target_height != height:
            preview_image = preview_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        preview_quality = min(self.settings.preview_quality, 75)
        return preview_image, preview_quality

    def generate(
        self,
        image: Image.Image,
        size_key: str,
        background_color: str,
        enhance: bool,
        save_output: bool,
    ) -> GenerateData:
        logger.info('Start generate pipeline size=%s background=%s enhance=%s', size_key, background_color, enhance)
        spec = get_photo_spec(size_key)
        detect_result = self.detector.detect(image)
        if not detect_result.can_generate:
            logger.info('Generate blocked by detect gatekeeper status=%s reasons=%s', detect_result.status, detect_result.reason_codes)
            self._raise_detect_failure(detect_result)

        task_id = build_task_id('gen')
        self.storage.category_task_dir('temp', task_id)
        logger.info('Task directories prepared task_id=%s upload_root=%s', task_id, self.storage.upload_root)

        rgba_foreground = self.segmenter.remove_background(image)
        logger.info('Background removed')
        if self.settings.save_intermediate:
            save_image(rgba_foreground, self.storage.temp_path(task_id, 'foreground.png'))

        refined = self.matte_refiner.refine(image, rgba_foreground)
        refined_rgba = refined.rgba
        decontaminated_refined_rgba = refined.decontaminated_rgba
        logger.info('Matte refined')
        if self.settings.save_intermediate:
            save_image(refined.alpha, self.storage.temp_path(task_id, 'refined_alpha.png'))
            save_image(refined.trimap, self.storage.temp_path(task_id, 'trimap.png'))
            save_image(refined_rgba, self.storage.temp_path(task_id, 'refined_foreground.png'))
            if refined.edge_band_mask is not None:
                save_image(refined.edge_band_mask, self.storage.temp_path(task_id, 'edge_band_mask.png'))
            if decontaminated_refined_rgba is not None:
                save_image(decontaminated_refined_rgba, self.storage.temp_path(task_id, 'foreground_decontaminated.png'))

        cropped_legacy_rgba = self.cropper.crop(refined_rgba, spec, detect_result.primary_face)
        cropped_decontaminated_rgba = None
        if self.settings.enable_foreground_decontamination and decontaminated_refined_rgba is not None:
            cropped_decontaminated_rgba = self.cropper.crop(decontaminated_refined_rgba, spec, detect_result.primary_face)
        use_decontaminated_output = bool(
            self.settings.enable_decontaminated_output_as_default
            and self.settings.enable_foreground_decontamination
            and cropped_decontaminated_rgba is not None
        )
        cropped_rgba = cropped_decontaminated_rgba if use_decontaminated_output else cropped_legacy_rgba
        logger.info('Portrait cropped to target size')
        if self.settings.save_intermediate:
            save_image(cropped_rgba, self.storage.temp_path(task_id, 'cropped_rgba.png'))
            save_image(cropped_legacy_rgba, self.storage.temp_path(task_id, 'cropped_rgba_legacy.png'))
            if cropped_decontaminated_rgba is not None:
                save_image(cropped_decontaminated_rgba, self.storage.temp_path(task_id, 'cropped_rgba_decontaminated.png'))

        hd_image_legacy = self.background.apply(cropped_legacy_rgba, background_color)
        hd_image = hd_image_legacy
        hd_image_decontaminated = None
        if cropped_decontaminated_rgba is not None:
            hd_image_decontaminated = self.background.apply_edge_aware(cropped_decontaminated_rgba, background_color)
            if use_decontaminated_output:
                hd_image = hd_image_decontaminated
        logger.info('Background applied')
        if self.settings.save_intermediate and hd_image_decontaminated is not None:
            save_image(hd_image_legacy, self.storage.temp_path(task_id, 'hd_legacy.png'))
            save_image(hd_image_decontaminated, self.storage.temp_path(task_id, 'hd_decontaminated.png'))
        if enhance:
            hd_image = self.enhancer.enhance(hd_image)
            logger.info('Enhancement applied')

        output_quality = self.output_quality.evaluate(
            source_image=image,
            output_image=hd_image,
            foreground_rgba=cropped_rgba,
            face_box=detect_result.primary_face,
            background_color=background_color,
        )
        logger.info('Output quality evaluated status=%s reasons=%s warnings=%s', output_quality.status, output_quality.reason_codes, output_quality.warnings)

        preview_image, preview_quality = self._build_preview_image(hd_image)

        hd_path = self.storage.hd_path(task_id, 'id_photo_hd.png')
        preview_path = self.storage.preview_path(task_id, 'id_photo_preview.jpg')
        if save_output:
            save_image(hd_image, hd_path)
            save_image(preview_image, preview_path, quality=preview_quality)
            logger.info('Output files saved hd=%s preview=%s', hd_path, preview_path)

        intermediate_files = None
        if self.settings.save_intermediate:
            if output_quality.cloth_pollution_mask is not None:
                save_image(output_quality.cloth_pollution_mask, self.storage.temp_path(task_id, 'cloth_pollution_mask.png'))
            intermediate_files = {}
            for name in (
                'foreground.png',
                'refined_alpha.png',
                'trimap.png',
                'refined_foreground.png',
                'foreground_decontaminated.png',
                'edge_band_mask.png',
                'cropped_rgba.png',
                'cropped_rgba_legacy.png',
                'cropped_rgba_decontaminated.png',
                'hd_legacy.png',
                'hd_decontaminated.png',
                'cloth_pollution_mask.png',
            ):
                path = self.storage.temp_path(task_id, name)
                if path.exists():
                    intermediate_files[name] = self._file_info(path)

        warnings = detect_result.warnings.copy()
        warnings.extend(output_quality.warnings)
        detect_summary = self._build_detect_data(detect_result)
        compliance_status = self._to_compliance_status(detect_result.status)
        compliance_message = self._compliance_message(compliance_status)
        compliance_details = detect_summary.complianceDetails
        if output_quality.status == 'FAIL':
            final_quality_status = 'FAIL'
            final_quality_message = '输出图存在明显异常，不可用于正式证件照'
            safe_to_submit = False
        elif detect_result.status == 'PASS' and output_quality.status == 'PASS':
            final_quality_status = 'PASS'
            final_quality_message = '输入与输出质量均通过，可用于正式证件照提交'
            safe_to_submit = True
        else:
            final_quality_status = 'WARNING'
            final_quality_message = '图片已生成，但仍存在质量风险，不建议直接用于正式证件照提交'
            safe_to_submit = False

        preview_info = self._file_info(preview_path) if save_output else FileInfo(path='', url='')
        hd_info = self._file_info(hd_path) if save_output else FileInfo(path='', url='')
        return GenerateData(
            taskId=task_id,
            previewPath=preview_info.path,
            previewUrl=preview_info.url,
            hdPath=hd_info.path,
            hdUrl=hd_info.url,
            backgroundColor=background_color,
            size=self._size_info(spec),
            width=spec.width_px,
            height=spec.height_px,
            warnings=warnings,
            detect=detect_summary,
            detectSummary=detect_summary,
            primaryIssue=output_quality.primary_issue or detect_result.primary_issue,
            primaryMessage=output_quality.primary_message or detect_result.primary_message,
            secondaryWarnings=detect_result.secondary_warnings,
            qualityStatus=final_quality_status,
            qualityMessage=final_quality_message,
            outputQualityStatus=output_quality.status,
            outputQualityMessage=output_quality.primary_message or '输出成片质量正常',
            outputReasonCodes=output_quality.reason_codes,
            allowPreviewSave=output_quality.status != 'FAIL',
            allowHdSave=output_quality.status == 'PASS' and detect_result.status == 'PASS',
            previewWidth=preview_image.width,
            previewHeight=preview_image.height,
            previewFormat='JPEG',
            previewQuality=preview_quality,
            hdWidth=hd_image.width,
            hdHeight=hd_image.height,
            hdFormat='PNG',
            hdQuality=100,
            intermediateFiles=intermediate_files,
            processStatus='generated',
            processMessage='图片已生成',
            complianceStatus=compliance_status,
            complianceMessage=compliance_message,
            complianceDetails=compliance_details,
            safeToSubmit=safe_to_submit,
            outputQualityMetrics=output_quality.metrics,
        )

    def layout_from_photo(self, photo: Image.Image, spec: PhotoSpec, paper: str) -> tuple[Image.Image, int]:
        return self.layout_service.build(photo, spec, paper)

    def resolve_spec(self, size_key: str | None) -> PhotoSpec:
        return get_photo_spec(size_key or self.settings.default_size_key)

    async def generate_from_upload(
        self,
        file: UploadFile,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
    ) -> GenerateData:
        _, image = await self.read_upload(file)
        return self.generate(
            image=image,
            size_key=size_key or self.settings.default_size_key,
            background_color=background_color or self.settings.default_background_color,
            enhance=enhance,
            save_output=save_output,
        )

    def generate_from_path(
        self,
        image_path: str,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
    ) -> GenerateData:
        _, image = self.read_image_path(image_path)
        return self.generate(
            image=image,
            size_key=size_key or self.settings.default_size_key,
            background_color=background_color or self.settings.default_background_color,
            enhance=enhance,
            save_output=save_output,
        )

    async def layout(
        self,
        id_photo: UploadFile | None,
        image: UploadFile | None,
        id_photo_path: str | None,
        image_path: str | None,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
        paper: str,
    ) -> LayoutData:
        if id_photo is None and image is None and id_photo_path is None and image_path is None:
            raise InvalidArgumentError('Either idPhoto, image, idPhotoPath, or imagePath must be provided')

        task_id = build_task_id('layout')
        self.storage.category_task_dir('temp', task_id)
        warnings: list[str] = []
        source_hd_info = None
        spec = self.resolve_spec(size_key)

        if id_photo is not None or id_photo_path is not None:
            if id_photo is not None:
                _, photo = await self.read_upload(id_photo)
            else:
                _, photo = self.read_image_path(id_photo_path or '')
            source_hd_path = self.storage.temp_path(task_id, 'source_id_photo.png')
            if save_output:
                save_image(photo.convert('RGB'), source_hd_path)
                source_hd_info = self._file_info(source_hd_path)
            else:
                source_hd_info = FileInfo(path='', url='')
            hd_image = photo.convert('RGB')
        else:
            if image is not None:
                generated = await self.generate_from_upload(
                    file=image,
                    size_key=spec.key,
                    background_color=background_color,
                    enhance=enhance,
                    save_output=True,
                )
            else:
                generated = self.generate_from_path(
                    image_path=image_path or '',
                    size_key=spec.key,
                    background_color=background_color,
                    enhance=enhance,
                    save_output=True,
                )
            warnings.extend(generated.warnings)
            hd_image = Image.open(generated.hdPath).convert('RGB')
            source_hd_info = FileInfo(path=generated.hdPath, url=generated.hdUrl)
            task_id = generated.taskId
            self.storage.category_task_dir('temp', task_id)
            spec = get_photo_spec(generated.size.key)

        layout_image, count = self.layout_service.build(hd_image, spec, paper)
        layout_path = self.storage.print_path(task_id, 'layout_6inch.jpg')
        if save_output:
            save_image(layout_image, layout_path, quality=self.settings.hd_quality)
            logger.info('Layout saved path=%s count=%s', layout_path, count)

        layout_info = self._file_info(layout_path) if save_output else FileInfo(path='', url='')
        return LayoutData(
            taskId=task_id,
            layoutPath=layout_info.path,
            layoutUrl=layout_info.url,
            paper=paper,
            count=count,
            photoSize=self._size_info(spec),
            warnings=warnings,
            sourceHd=source_hd_info,
        )


@lru_cache(maxsize=1)
def get_photo_processor() -> PhotoProcessor:
    return PhotoProcessor()
