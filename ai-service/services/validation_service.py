from __future__ import annotations

import imghdr
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from core.exceptions import (
    AppException,
    ERROR_EYE_OCCLUDED,
    ERROR_FACE_OCCLUDED,
    ERROR_FACE_TOO_SMALL,
    ERROR_FACIAL_KEYPOINTS_INCOMPLETE,
    ERROR_HAND_OCCLUSION,
    ERROR_HEAD_CROPPED,
    ERROR_HEADWEAR_DETECTED,
    ERROR_IMAGE_TOO_BLURRY,
    ERROR_IMAGE_TOO_SMALL,
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_IMAGE,
    ERROR_MULTIPLE_FACES_DETECTED,
    ERROR_NOT_SINGLE_FRONTAL_FACE,
    ERROR_NO_FACE_DETECTED,
    ERROR_POSE_INVALID,
)
from services.compliance_service import ComplianceService
from services.face_postprocess_service import FacePostprocessService
from services.quality_service import QualityService
from utils.config import get_settings
from utils.logger import get_logger


@dataclass
class LoadedImage:
    filename: str
    content_type: str
    file_size: int
    image: Image.Image
    image_np: np.ndarray
    width: int
    height: int
    format: str
    mode: str

    def metadata(self) -> dict:
        return {
            'filename': self.filename,
            'contentType': self.content_type,
            'fileSize': self.file_size,
            'width': self.width,
            'height': self.height,
            'format': self.format,
            'mode': self.mode,
        }


@dataclass
class ValidationOutcome:
    hasFace: bool
    faceCount: int
    passed: bool
    reasons: list[str] = field(default_factory=list)
    message: str = '未检测到有效人脸'
    blurScore: float | None = None
    imageWidth: int = 0
    imageHeight: int = 0
    primaryFaceBox: dict | None = None
    rawFaceCount: int = 0
    validFaceBoxes: list[dict] = field(default_factory=list)
    filteredOutReasons: list[dict] = field(default_factory=list)
    auditStatus: str = 'passed'
    auditCode: str = 'VALIDATION_PASSED'
    auditMessage: str = '审核通过'
    auditDetails: list[dict] = field(default_factory=list)
    keypointConfidences: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'hasFace': self.hasFace,
            'faceCount': self.faceCount,
            'pass': self.passed,
            'reasons': self.reasons,
            'message': self.message,
            'blurScore': self.blurScore,
            'imageWidth': self.imageWidth,
            'imageHeight': self.imageHeight,
            'primaryFaceBox': self.primaryFaceBox,
            'rawFaceCount': self.rawFaceCount,
            'validFaceBoxes': self.validFaceBoxes,
            'filteredOutReasons': self.filteredOutReasons,
            'auditResult': {
                'status': self.auditStatus,
                'code': self.auditCode,
                'message': self.auditMessage,
                'details': self.auditDetails,
            },
            'keypointConfidences': self.keypointConfidences,
            'warnings': self.warnings,
        }


class ValidationService:
    reason_priority = [
        ERROR_NO_FACE_DETECTED,
        ERROR_MULTIPLE_FACES_DETECTED,
        ERROR_EYE_OCCLUDED,
        ERROR_POSE_INVALID,
        ERROR_FACE_OCCLUDED,
        ERROR_FACIAL_KEYPOINTS_INCOMPLETE,
        ERROR_HAND_OCCLUSION,
        ERROR_HEADWEAR_DETECTED,
        ERROR_NOT_SINGLE_FRONTAL_FACE,
        ERROR_FACE_TOO_SMALL,
        ERROR_HEAD_CROPPED,
        ERROR_IMAGE_TOO_BLURRY,
    ]

    detect_messages = {
        ERROR_NO_FACE_DETECTED: '未检测到人脸',
        ERROR_MULTIPLE_FACES_DETECTED: '检测到多张人脸，不符合证件照制作要求',
        ERROR_IMAGE_TOO_BLURRY: '图片过于模糊，不符合证件照制作要求',
        ERROR_FACE_TOO_SMALL: '人脸区域过小，不符合证件照制作要求',
        ERROR_POSE_INVALID: '人脸姿态不符合证件照要求',
        ERROR_FACE_OCCLUDED: '人脸存在严重遮挡，不符合证件照制作要求',
        ERROR_EYE_OCCLUDED: '人脸存在明显遮挡，请露出双眼和完整面部后重试',
        ERROR_FACIAL_KEYPOINTS_INCOMPLETE: '当前照片不符合证件照规范，请重新拍摄',
        ERROR_HAND_OCCLUSION: '检测到帽子或手部遮挡，不符合证件照要求',
        ERROR_HEAD_CROPPED: '头部截断过于严重，不符合证件照制作要求',
        ERROR_NOT_SINGLE_FRONTAL_FACE: '当前照片不符合证件照规范，请使用正脸、无遮挡照片',
        ERROR_HEADWEAR_DETECTED: '检测到帽子或头部遮挡，不符合证件照要求',
    }

    generate_messages = {
        ERROR_NO_FACE_DETECTED: '未检测到有效人脸，请上传清晰的单人正脸照片',
        ERROR_MULTIPLE_FACES_DETECTED: '检测到多张人脸，请上传单人正脸照片',
        ERROR_IMAGE_TOO_BLURRY: '图片过于模糊，请上传清晰照片',
        ERROR_FACE_TOO_SMALL: '人脸区域过小，请上传人物主体更清晰的单人正脸照片',
        ERROR_POSE_INVALID: '人脸姿态不符合要求，请上传单人正脸照片',
        ERROR_FACE_OCCLUDED: '人脸遮挡过于严重，请上传无遮挡的单人正脸照片',
        ERROR_EYE_OCCLUDED: '人脸存在明显遮挡，请露出双眼和完整面部后重试',
        ERROR_FACIAL_KEYPOINTS_INCOMPLETE: '当前照片不符合证件照规范，请重新拍摄',
        ERROR_HAND_OCCLUSION: '检测到帽子或手部遮挡，不符合证件照要求',
        ERROR_HEAD_CROPPED: '头部截断过于严重，请上传头部完整的单人正脸照片',
        ERROR_NOT_SINGLE_FRONTAL_FACE: '当前照片不符合证件照规范，请使用正脸、无遮挡照片',
        ERROR_HEADWEAR_DETECTED: '检测到帽子或头部遮挡，不符合证件照要求',
    }

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger()
        self.quality_service = QualityService()
        self.face_postprocess_service = FacePostprocessService()
        self.compliance_service = ComplianceService()

    def load_image(self, file_bytes: bytes, filename: str, content_type: str | None = None) -> LoadedImage:
        suffix = Path(filename or 'upload.jpg').suffix.lower()
        if suffix and suffix not in self.settings.allowed_image_extensions:
            raise AppException('不支持的图片格式，请上传 JPG/PNG/BMP/WEBP 图片', ERROR_INVALID_IMAGE, 400)

        declared_type = (content_type or '').lower()
        if declared_type and declared_type not in self.settings.allowed_image_content_types:
            raise AppException('文件 Content-Type 不是受支持的图片类型', ERROR_INVALID_IMAGE, 400)

        size_limit = self.settings.max_upload_mb * 1024 * 1024
        if len(file_bytes) == 0:
            raise AppException('上传文件为空', ERROR_INVALID_ARGUMENT, 400)
        if len(file_bytes) > size_limit:
            raise AppException(
                f'文件大小超过限制，当前最大允许 {self.settings.max_upload_mb} MB',
                ERROR_INVALID_ARGUMENT,
                400,
            )

        guessed_kind = imghdr.what(None, h=file_bytes)
        if guessed_kind is None:
            raise AppException('上传文件不是有效图片', ERROR_INVALID_IMAGE, 400)

        try:
            image = Image.open(BytesIO(file_bytes))
            exif_orientation = None
            try:
                exif_orientation = image.getexif().get(274)
            except Exception:
                exif_orientation = None
            image = ImageOps.exif_transpose(image)
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise AppException('图片解析失败，请确认文件未损坏', ERROR_INVALID_IMAGE, 400) from exc

        rgb_image = image.convert('RGB')
        width, height = rgb_image.size
        if width < self.settings.min_image_width or height < self.settings.min_image_height:
            raise AppException(
                f'图片尺寸过小，最小要求为 {self.settings.min_image_width}x{self.settings.min_image_height}',
                ERROR_IMAGE_TOO_SMALL,
                400,
            )
        self.logger.info(
            '[detect-chain] preprocess image_read filename={} exif_orientation={} final_size={}x{} mode={}',
            filename,
            exif_orientation,
            width,
            height,
            image.mode,
        )

        return LoadedImage(
            filename=filename or f'upload.{guessed_kind}',
            content_type=declared_type or f'image/{guessed_kind}',
            file_size=len(file_bytes),
            image=rgb_image,
            image_np=np.array(rgb_image),
            width=width,
            height=height,
            format=(image.format or guessed_kind or 'UNKNOWN').upper(),
            mode=image.mode,
        )

    def validate_upload_image(self, loaded_image: LoadedImage) -> dict:
        quality_status, quality_message = self.quality_service.evaluate(loaded_image.image)
        return {
            'passed': True,
            'message': '图片基础校验通过',
            'reasons': [],
            'metadata': loaded_image.metadata(),
            'qualityStatus': quality_status,
            'qualityMessage': quality_message,
        }

    def _build_message(self, reasons: list[str], passed: bool, for_generate: bool = False) -> str:
        if passed:
            return '图片符合证件照制作要求'
        if not reasons:
            return '图片不符合证件照制作要求'
        mapping = self.generate_messages if for_generate else self.detect_messages
        return mapping.get(reasons[0], '图片不符合证件照制作要求')

    def validate(self, image_shape: tuple[int, ...], faces, blur_score: float, image_bgr=None, gray_image=None) -> ValidationOutcome:
        postprocess_result = self.face_postprocess_service.face_box_postprocess(faces)
        face_count = len(postprocess_result.validFaces)
        has_face = face_count > 0
        primary_face_box = postprocess_result.primaryFaceBox
        reasons: list[str] = []

        if face_count == 0:
            reasons.append(ERROR_NO_FACE_DETECTED)
        elif face_count > 1:
            reasons.append(ERROR_MULTIPLE_FACES_DETECTED)
        else:
            if self.quality_service.is_image_too_blurry(blur_score):
                reasons.append(ERROR_IMAGE_TOO_BLURRY)
            if primary_face_box and self.quality_service.is_face_too_small(image_shape, primary_face_box):
                reasons.append(ERROR_FACE_TOO_SMALL)
            if primary_face_box and self.quality_service.is_pose_invalid(image_shape, primary_face_box):
                reasons.append(ERROR_POSE_INVALID)
            if primary_face_box and self.quality_service.is_face_occluded(image_shape, primary_face_box):
                reasons.append(ERROR_FACE_OCCLUDED)
            if primary_face_box and self.quality_service.is_head_cropped(image_shape, primary_face_box):
                reasons.append(ERROR_HEAD_CROPPED)
            if primary_face_box and image_bgr is not None and gray_image is not None:
                compliance_result = self.compliance_service.evaluate(
                    image_bgr=image_bgr,
                    gray_image=gray_image,
                    face_box=primary_face_box,
                    face_count=face_count,
                )
                if compliance_result['status'] == 'failed':
                    for detail in compliance_result['details']:
                        if detail.get('status', 'failed') == 'failed':
                            reasons.append(detail['code'])
                self.logger.info(
                    '[detect-chain] compliance_result status={} code={} keypoints={} warnings={} details={}',
                    compliance_result['status'],
                    compliance_result['code'],
                    compliance_result.get('keypointConfidences', {}),
                    compliance_result.get('warnings', []),
                    compliance_result.get('details', []),
                )
            else:
                compliance_result = {
                    'status': 'warning',
                    'code': 'COMPLIANCE_SKIPPED',
                    'message': '未执行合规性审核',
                    'details': [],
                    'keypointConfidences': {},
                    'warnings': ['未执行合规性审核'],
                }

        ordered_reasons = [reason for reason in self.reason_priority if reason in reasons]
        non_priority = [reason for reason in reasons if reason not in ordered_reasons]
        ordered_reasons.extend(non_priority)
        passed = not ordered_reasons
        warnings: list[str] = []
        if passed:
            audit_status = 'passed'
            audit_code = 'VALIDATION_PASSED'
            audit_message = '图片符合证件照制作要求'
        else:
            audit_status = 'failed'
            audit_code = ordered_reasons[0]
            if ordered_reasons[0] == ERROR_FACE_OCCLUDED:
                audit_message = '人脸存在明显遮挡，请露出双眼和完整面部后重试'
            elif ordered_reasons[0] in {ERROR_EYE_OCCLUDED, ERROR_HAND_OCCLUSION}:
                audit_message = '检测到帽子或手部遮挡，不符合证件照要求'
            elif ordered_reasons[0] == ERROR_FACIAL_KEYPOINTS_INCOMPLETE:
                audit_message = '当前照片不符合证件照规范，请重新拍摄'
            elif ordered_reasons[0] == ERROR_NOT_SINGLE_FRONTAL_FACE:
                audit_message = '当前照片不符合证件照规范，请使用正脸、无遮挡照片'
            else:
                audit_message = self._build_message(ordered_reasons, passed=passed, for_generate=False)
        compliance_details = compliance_result['details'] if 'compliance_result' in locals() else []
        compliance_codes = {item['code'] for item in compliance_details}
        audit_details = list(compliance_details)
        for warning in compliance_result.get('warnings', []) if 'compliance_result' in locals() else []:
            warnings.append(warning)
            audit_details.append(
                {
                    'code': 'COMPLIANCE_WARNING',
                    'message': warning,
                    'status': 'warning',
                    'stage': 'keypoint_detection',
                }
            )
        compliance_warning_details = [
            item for item in compliance_details if item.get('status') == 'warning' and item.get('message')
        ]
        for item in compliance_warning_details:
            warnings.append(item['message'])
        for reason in ordered_reasons:
            if reason in compliance_codes:
                continue
            audit_details.append(
                {
                    'code': reason,
                    'message': self.detect_messages.get(reason, '图片不符合证件照制作要求'),
                    'status': 'failed',
                    'stage': 'validation',
                }
            )
        if passed and warnings:
            audit_status = 'warning'
            audit_code = 'COMPLIANCE_WARNING'
            audit_message = warnings[0]
        return ValidationOutcome(
            hasFace=has_face,
            faceCount=face_count,
            passed=passed,
            reasons=ordered_reasons,
            message=audit_message,
            blurScore=blur_score,
            imageWidth=image_shape[1],
            imageHeight=image_shape[0],
            primaryFaceBox=primary_face_box,
            rawFaceCount=postprocess_result.rawFaceCount,
            validFaceBoxes=postprocess_result.validFaces,
            filteredOutReasons=postprocess_result.filteredOutReasons,
            auditStatus=audit_status,
            auditCode=audit_code,
            auditMessage=audit_message,
            auditDetails=audit_details,
            keypointConfidences=compliance_result['keypointConfidences'] if 'compliance_result' in locals() else {},
            warnings=warnings,
        )

    def build_generate_error(self, reasons: list[str]) -> tuple[str, str]:
        ordered_reasons = [reason for reason in self.reason_priority if reason in reasons]
        ordered_reasons.extend([reason for reason in reasons if reason not in ordered_reasons])
        error_code = ordered_reasons[0] if ordered_reasons else ERROR_NO_FACE_DETECTED
        if error_code == ERROR_FACE_OCCLUDED:
            return error_code, '人脸存在明显遮挡，请露出双眼和完整面部后重试'
        if error_code in {ERROR_EYE_OCCLUDED, ERROR_HAND_OCCLUSION}:
            return error_code, '检测到帽子或手部遮挡，不符合证件照要求'
        if error_code == ERROR_FACIAL_KEYPOINTS_INCOMPLETE:
            return error_code, '当前照片不符合证件照规范，请重新拍摄'
        if error_code == ERROR_HEADWEAR_DETECTED:
            return error_code, '检测到帽子或手部遮挡，不符合证件照要求'
        if error_code == ERROR_NOT_SINGLE_FRONTAL_FACE:
            return error_code, '当前照片不符合证件照规范，请使用正脸、无遮挡照片'
        return error_code, self._build_message(ordered_reasons, passed=False, for_generate=True)
