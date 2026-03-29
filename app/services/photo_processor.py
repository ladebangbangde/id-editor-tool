import json
import time
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from app.schemas.generate import GenerateCandidate, GenerateData, GenerateSelectionData
from app.schemas.layout import LayoutData
from app.services.background import BackgroundService
from app.services.engines.legacy_engine import LegacyPhotoGenerationEngine
from app.services.cropper import CropperService
from app.services.enhancer import EnhancerService
from app.services.face_detection import FaceDetectionResult, FaceDetectionService
from app.services.layout import LayoutService
from app.services.matte_refine_service import MatteRefineService
from app.services.output_quality_service import OutputQualityService
from app.services.segmentation import SegmentationService
from app.services.photo_generation_engine import EngineInput
from app.services.specs import PhotoSpec, get_photo_spec
from app.services.storage import StorageService
from app.services.task_status_store import (
    STAGE_ADJUSTING,
    STAGE_CHECKING,
    STAGE_FAILED,
    STAGE_FINALIZING,
    STAGE_GENERATING,
    STAGE_SUCCESS,
    TaskStatusStore,
    get_task_status_store,
)
from app.utils.file_naming import build_task_id
from app.utils.image_io import load_image_from_bytes, load_image_from_path, resolve_input_path, save_image
from app.utils.validators import validate_content_size, validate_upload

logger = get_logger(__name__)


class PhotoProcessor:
    _debug_save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='debug-save')

    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage = StorageService()
        self.detector = FaceDetectionService()
        self.segmenter = SegmentationService(backend_mode='baidu')
        self.legacy_segmenter = SegmentationService(backend_mode='legacy')
        self.background = BackgroundService()
        self.cropper = CropperService()
        self.enhancer = EnhancerService()
        self.matte_refiner = MatteRefineService()
        self.output_quality = OutputQualityService()
        self.layout_service = LayoutService()
        self.task_status_store: TaskStatusStore = get_task_status_store()
        self.baidu_engine = LegacyPhotoGenerationEngine(
            segmenter=self.segmenter,
            matte_refiner=self.matte_refiner,
            cropper=self.cropper,
            background=self.background,
            enhancer=self.enhancer,
            preview_builder=self._build_preview_image,
            settings=self.settings,
        )
        self.legacy_engine = LegacyPhotoGenerationEngine(
            segmenter=self.legacy_segmenter,
            matte_refiner=self.matte_refiner,
            cropper=self.cropper,
            background=self.background,
            enhancer=self.enhancer,
            preview_builder=self._build_preview_image,
            settings=self.settings,
        )

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
        detect_warnings = [
            warning
            for warning in result.warnings
            if warning and not self._looks_like_internal_message(warning)
        ]
        secondary_warnings = [
            warning
            for warning in result.secondary_warnings
            if warning and not self._looks_like_internal_message(warning)
        ]
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
            warnings=detect_warnings,
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
            warning='；'.join(detect_warnings) if detect_warnings else None,
            blurScore=result.blur_score,
            occlusionDetected=result.occlusion_detected,
            occlusionAreas=result.occlusion_areas,
            poseAccepted=result.pose_accepted,
            landmarkStable=result.landmark_stable,
            compositionAccepted=result.composition_accepted,
            metrics=result.metrics,
            primaryIssue=result.primary_issue,
            primaryMessage=(
                None
                if result.primary_message and self._looks_like_internal_message(result.primary_message)
                else result.primary_message
            ),
            secondaryWarnings=secondary_warnings,
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

    @staticmethod
    def _looks_like_internal_message(message: str) -> bool:
        lowered = message.lower()
        internal_markers = (
            'enginecomparison',
            'legacy-only',
            'backend=',
            'fallback',
            'segmentation',
            'compare=',
            'output_quality=',
            'debug',
        )
        return any(marker in lowered for marker in internal_markers)

    def _build_user_facing_warnings(
        self,
        detect_warnings: list[str],
        quality_warning_codes: list[str],
    ) -> list[str]:
        user_warnings: list[str] = []
        for warning in detect_warnings:
            cleaned = warning.strip()
            if not cleaned or self._looks_like_internal_message(cleaned):
                continue
            if cleaned not in user_warnings:
                user_warnings.append(cleaned)

        for message in self.output_quality.user_messages_for_issues(quality_warning_codes):
            if message not in user_warnings:
                user_warnings.append(message)

        return user_warnings

    def _save_intermediate_images(
        self,
        task_id: str,
        debug_images: dict[str, Image.Image],
    ) -> dict[str, FileInfo]:
        intermediate_files: dict[str, FileInfo] = {}
        for name, img in debug_images.items():
            path = self.storage.temp_path(task_id, name)
            # 异步写盘，避免中间图 I/O 阻塞正式响应主链路。
            self._debug_save_executor.submit(save_image, img, path)
            intermediate_files[name] = self._file_info(path)
        return intermediate_files

    def _save_candidate_manifest(self, task_id: str, candidates: list[GenerateCandidate]) -> None:
        manifest_path = self.storage.temp_path(task_id, 'candidates_manifest.json')
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'taskId': task_id,
            'candidates': [candidate.model_dump() for candidate in candidates],
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def _load_candidate_manifest(self, task_id: str) -> dict:
        manifest_path = self.storage.temp_path(task_id, 'candidates_manifest.json')
        if not manifest_path.exists():
            raise InvalidArgumentError('未找到候选结果，请先生成图片')
        try:
            return json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise InvalidArgumentError('候选结果数据损坏，请重新生成') from exc

    def _build_candidate_quality(
        self,
        *,
        detect_result: FaceDetectionResult,
        selected_quality,
    ) -> tuple[str, str, bool]:
        if selected_quality.status == 'FAIL':
            return 'FAIL', '输出图存在明显异常，不可用于正式证件照', False
        if detect_result.status == 'PASS' and selected_quality.status == 'PASS':
            return 'PASS', '输入与输出质量均通过，可用于正式证件照提交', True
        return 'WARNING', '图片已生成，但仍存在质量风险，不建议直接用于正式证件照提交', False

    def _run_candidate_pipeline(
        self,
        *,
        task_id: str,
        candidate_id: str,
        engine_key: str,
        label: str,
        engine: LegacyPhotoGenerationEngine,
        payload: EngineInput,
        source_image: Image.Image,
        detect_result: FaceDetectionResult,
        save_output: bool,
    ) -> tuple[GenerateCandidate, dict[str, FileInfo] | None, dict[str, float], list[str]]:
        started = time.perf_counter()
        stage_messages = {
            'adjusting': '正在进行抠图、换底与裁切准备',
            'generating': '正在并行生成候选证件照',
        }

        def stage_reporter(stage_name: str) -> None:
            if stage_name == 'adjusting':
                self.task_status_store.update_stage(task_id, STAGE_ADJUSTING, stage_messages[stage_name])
            elif stage_name == 'generating':
                self.task_status_store.update_stage(task_id, STAGE_GENERATING, stage_messages[stage_name])

        selected_result = engine.generate(payload, stage_reporter=stage_reporter)
        selected_quality = self.output_quality.evaluate(
            source_image=source_image,
            output_image=selected_result.hd_image,
            foreground_rgba=selected_result.foreground_rgba,
            face_box=detect_result.primary_face,
            background_color=payload.background_color,
        )
        duration_ms = (time.perf_counter() - started) * 1000.0

        candidate_quality_status, candidate_quality_message, candidate_safe = self._build_candidate_quality(
            detect_result=detect_result,
            selected_quality=selected_quality,
        )
        warnings = self._build_user_facing_warnings(
            detect_warnings=detect_result.warnings,
            quality_warning_codes=selected_quality.warnings,
        )
        primary_message = selected_quality.primary_message or detect_result.primary_message or (warnings[0] if warnings else None)

        hd_path = self.storage.hd_path(task_id, f'{candidate_id}.png')
        preview_path = self.storage.preview_path(task_id, f'{candidate_id}_preview.jpg')
        if save_output:
            save_image(selected_result.hd_image, hd_path)
            save_image(selected_result.preview_image, preview_path, quality=selected_result.preview_quality)

        candidate_hd_info = self._file_info(hd_path) if save_output else FileInfo(path='', url='')
        candidate_preview_info = self._file_info(preview_path) if save_output else FileInfo(path='', url='')

        candidate_intermediate = None
        if self.settings.save_intermediate:
            all_debug = {f'{candidate_id}/{name}': img for name, img in selected_result.debug_images.items()}
            if selected_quality.cloth_pollution_mask is not None:
                all_debug[f'{candidate_id}/cloth_pollution_mask.png'] = selected_quality.cloth_pollution_mask
            if selected_quality.hair_gap_residue_mask is not None:
                all_debug[f'{candidate_id}/hair_gap_residue_mask.png'] = selected_quality.hair_gap_residue_mask
            candidate_intermediate = self._save_intermediate_images(task_id=task_id, debug_images=all_debug)

        logger.info(
            'Candidate generated candidate=%s engine=%s duration_ms=%.2f quality=%s output_quality=%s',
            candidate_id,
            engine_key,
            duration_ms,
            candidate_quality_status,
            selected_quality.status,
        )

        candidate = GenerateCandidate(
            candidateId=candidate_id,
            engineKey=engine_key,
            label=label,
            imagePath=candidate_hd_info.path,
            imageUrl=candidate_hd_info.url,
            width=selected_result.hd_image.width,
            height=selected_result.hd_image.height,
            format='PNG',
            previewPath=candidate_preview_info.path,
            previewUrl=candidate_preview_info.url,
            qualityStatus=candidate_quality_status,
            qualityMessage=candidate_quality_message,
            outputQualityStatus=selected_quality.status,
            outputQualityMessage=selected_quality.primary_message or '输出成片质量正常',
            outputReasonCodes=selected_quality.reason_codes,
            warnings=warnings,
            primaryIssue=selected_quality.primary_issue or detect_result.primary_issue,
            primaryMessage=primary_message,
            safeToSubmit=candidate_safe,
            debugInfo={
                'durationMs': round(duration_ms, 2),
                'engine': engine_key,
                'outputQualityWarningCodes': selected_quality.warnings,
                'outputQualityReasonCodes': selected_quality.reason_codes,
            },
        )
        return candidate, candidate_intermediate, selected_quality.metrics, selected_quality.warnings

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
        task_id: str | None = None,
    ) -> GenerateData:
        logger.info('Start generate pipeline size=%s background=%s enhance=%s', size_key, background_color, enhance)
        active_task_id = (task_id or '').strip() or build_task_id('gen')
        self.task_status_store.create_task(task_id=active_task_id, message='任务已创建，等待检测')
        try:
            self.task_status_store.update_stage(
                active_task_id,
                STAGE_CHECKING,
                '正在进行图片读取与人像检测',
            )
            spec = get_photo_spec(size_key)
            detect_result = self.detector.detect(image)
            if not detect_result.can_generate:
                logger.info('Generate blocked by detect gatekeeper status=%s reasons=%s', detect_result.status, detect_result.reason_codes)
                self._raise_detect_failure(detect_result)

            self.task_status_store.update_stage(
                active_task_id,
                STAGE_ADJUSTING,
                '正在进行抠图、换底与裁切准备',
            )
            self.storage.category_task_dir('temp', active_task_id)
            logger.info('Task directories prepared task_id=%s upload_root=%s', active_task_id, self.storage.upload_root)

            payload = EngineInput(
                source_image=image,
                spec=spec,
                background_color=background_color,
                enhance=enhance,
                face_box=detect_result.primary_face,
            )
            logger.info('Parallel candidate generation start engines=[baidu, legacy]')
            candidate_specs = [
                ('baidu', 'baidu', '方案A', self.baidu_engine),
                ('legacy', 'legacy', '方案B', self.legacy_engine),
            ]
            candidates: list[GenerateCandidate] = []
            candidate_intermediate_files: dict[str, FileInfo] = {}
            candidate_metrics: dict[str, float] = {}
            candidate_warning_codes: dict[str, list[str]] = {}
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='candidate-gen') as executor:
                future_to_candidate = {
                    executor.submit(
                        self._run_candidate_pipeline,
                        task_id=active_task_id,
                        candidate_id=candidate_id,
                        engine_key=engine_key,
                        label=label,
                        engine=engine,
                        payload=payload,
                        source_image=image,
                        detect_result=detect_result,
                        save_output=save_output,
                    ): candidate_id
                    for candidate_id, engine_key, label, engine in candidate_specs
                }
                for future in as_completed(future_to_candidate):
                    candidate_id = future_to_candidate[future]
                    candidate, intermediate, metrics, quality_warning_codes = future.result()
                    candidates.append(candidate)
                    if intermediate:
                        candidate_intermediate_files.update(intermediate)
                    candidate_metrics[candidate_id] = float(metrics.get('edge_noise_score', 0.0))
                    candidate_warning_codes[candidate_id] = quality_warning_codes

            self.task_status_store.update_stage(
                active_task_id,
                STAGE_FINALIZING,
                '正在保存输出并整理返回数据',
            )
            candidates.sort(key=lambda item: item.candidateId)
            self._save_candidate_manifest(task_id=active_task_id, candidates=candidates)
            logger.info('Parallel candidate generation done task=%s candidates=%s', active_task_id, [item.candidateId for item in candidates])

            # 前端主展示区只能使用用户可读中文提示，内部工程信息必须留在 debugInfo / 日志。
            warnings: list[str] = []
            for candidate in candidates:
                for warning in candidate.warnings:
                    if warning not in warnings:
                        warnings.append(warning)
            detect_summary = self._build_detect_data(detect_result)
            compliance_status = self._to_compliance_status(detect_result.status)
            compliance_message = self._compliance_message(compliance_status)
            compliance_details = detect_summary.complianceDetails
            overall_status_rank = {'PASS': 0, 'WARNING': 1, 'FAIL': 2}
            global_quality_status = max((candidate.qualityStatus for candidate in candidates), key=lambda status: overall_status_rank.get(status, 1))
            if global_quality_status == 'FAIL':
                final_quality_message = '已生成候选图，但存在明显质量风险，请谨慎选择'
            elif global_quality_status == 'WARNING':
                final_quality_message = '已生成候选图，建议放大检查边缘细节后再选择'
            else:
                final_quality_message = '已生成候选图，请选择你更满意的一张保存'

            compatibility_candidate = next((item for item in candidates if item.candidateId == 'baidu'), candidates[0])
            secondary_warnings = [
                warning
                for warning in detect_result.secondary_warnings
                if warning and not self._looks_like_internal_message(warning)
            ]

            data = GenerateData(
                taskId=active_task_id,
                previewPath=compatibility_candidate.previewPath,
                previewUrl=compatibility_candidate.previewUrl,
                hdPath=compatibility_candidate.imagePath,
                hdUrl=compatibility_candidate.imageUrl,
                backgroundColor=background_color,
                size=self._size_info(spec),
                width=spec.width_px,
                height=spec.height_px,
                warnings=warnings,
                detect=detect_summary,
                detectSummary=detect_summary,
                candidates=candidates,
                selectedCandidateId=None,
                requireUserSelection=True,
                primaryIssue=compatibility_candidate.primaryIssue or detect_result.primary_issue,
                primaryMessage=compatibility_candidate.primaryMessage or detect_result.primary_message or (warnings[0] if warnings else None),
                secondaryWarnings=secondary_warnings,
                qualityStatus=global_quality_status,
                qualityMessage=final_quality_message,
                outputQualityStatus=compatibility_candidate.outputQualityStatus,
                outputQualityMessage=compatibility_candidate.outputQualityMessage,
                outputReasonCodes=compatibility_candidate.outputReasonCodes,
                allowPreviewSave=False,
                allowHdSave=False,
                previewWidth=0,
                previewHeight=0,
                previewFormat='JPEG',
                previewQuality=0,
                hdWidth=compatibility_candidate.width,
                hdHeight=compatibility_candidate.height,
                hdFormat='PNG',
                hdQuality=100,
                intermediateFiles=candidate_intermediate_files or None,
                debugInfo={
                    'parallelCandidates': [item.candidateId for item in candidates],
                    'candidateMetrics': candidate_metrics,
                    'candidateWarningCodes': candidate_warning_codes,
                },
                processStatus='generated',
                processMessage='已生成两套候选结果，请先选择要保存的图片',
                complianceStatus=compliance_status,
                complianceMessage=compliance_message,
                complianceDetails=compliance_details,
                safeToSubmit=False,
                outputQualityMetrics={},
            )
            self.task_status_store.update_stage(
                active_task_id,
                STAGE_SUCCESS,
                '任务处理完成',
            )
            return data
        except Exception as exc:
            self.task_status_store.update_stage(
                active_task_id,
                STAGE_FAILED,
                '任务处理失败',
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise

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
        task_id: str | None = None,
    ) -> GenerateData:
        _, image = await self.read_upload(file)
        return self.generate(
            image=image,
            size_key=size_key or self.settings.default_size_key,
            background_color=background_color or self.settings.default_background_color,
            enhance=enhance,
            save_output=save_output,
            task_id=task_id,
        )

    def generate_from_path(
        self,
        image_path: str,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
        task_id: str | None = None,
    ) -> GenerateData:
        _, image = self.read_image_path(image_path)
        return self.generate(
            image=image,
            size_key=size_key or self.settings.default_size_key,
            background_color=background_color or self.settings.default_background_color,
            enhance=enhance,
            save_output=save_output,
            task_id=task_id,
        )

    def select_candidate(self, task_id: str, candidate_id: str) -> GenerateSelectionData:
        if not candidate_id:
            raise InvalidArgumentError('请先选择要保存的图片')
        manifest = self._load_candidate_manifest(task_id)
        candidates = manifest.get('candidates', [])
        selected = next((item for item in candidates if item.get('candidateId') == candidate_id), None)
        if selected is None:
            raise InvalidArgumentError(f'未找到候选图片: {candidate_id}')

        logger.info('Candidate selection confirmed task=%s candidate=%s', task_id, candidate_id)
        return GenerateSelectionData(
            taskId=task_id,
            candidateId=candidate_id,
            imagePath=selected.get('imagePath', ''),
            imageUrl=selected.get('imageUrl', ''),
            previewPath=selected.get('previewPath', ''),
            previewUrl=selected.get('previewUrl', ''),
            status='selected',
            message='已确认所选图片',
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
