from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
from PIL import Image, ImageOps

from core.exceptions import ERROR_FACE_OCCLUDED, ERROR_POSE_INVALID
from services.quality_service import QualityService
from services.validation_service import LoadedImage, ValidationOutcome, ValidationService
from utils.logger import get_logger


@dataclass
class DetectOutcome:
    imageId: str
    faceDetected: bool
    faceCount: int
    faceBoxes: list[dict]
    blurScore: float
    poseValid: bool
    occlusionDetected: bool
    isProcessable: bool
    qualityStatus: str
    qualityMessage: str
    imageWidth: int
    imageHeight: int
    imageFormat: str
    imageMode: str
    validationPassed: bool
    reasons: list[str]
    suggestion: str
    message: str
    auditResult: dict
    keypointConfidences: dict[str, float]
    primaryFaceBox: dict | None = None

    @property
    def hasFace(self) -> bool:
        return self.faceDetected

    @property
    def passed(self) -> bool:
        return self.validationPassed

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['hasFace'] = self.faceDetected
        return payload


class DetectService:
    def __init__(self) -> None:
        self.validation_service = ValidationService()
        self.quality_service = QualityService()
        self.logger = get_logger()
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    @staticmethod
    def _calc_blur_score(image_bgr) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized = min(max(lap_var / 500.0, 0.0), 1.0)
        return round(float(normalized), 2)

    @staticmethod
    def _normalize_faces(faces) -> list[dict]:
        return [
            {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
            }
            for (x, y, w, h) in faces
        ]

    @staticmethod
    def _build_suggestion(validation_result: ValidationOutcome, quality_details: dict) -> str:
        if validation_result.passed:
            return '照片可以直接进入证件照制作流程'
        if not validation_result.hasFace:
            return '请上传单人正脸、五官清晰、背景相对简洁的照片'
        if quality_details['resolutionTooLow']:
            return '请使用更高分辨率原图，避免上传被压缩后的聊天截图'
        if quality_details['clarityInsufficient']:
            return '可继续生成，但建议换用更清晰的原图以提升最终效果'
        return validation_result.message

    def detect_from_loaded_image(self, image_id: str, loaded_image: LoadedImage) -> DetectOutcome:
        self.logger.info(
            '[detect-chain] image_read image_id={} size={}x{} format={} mode={}',
            image_id,
            loaded_image.width,
            loaded_image.height,
            loaded_image.format,
            loaded_image.mode,
        )
        image_bgr = cv2.cvtColor(loaded_image.image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        normalized_faces = self._normalize_faces(faces)
        self.logger.info(
            '[detect-chain] face_detection image_id={} face_count={} boxes={}',
            image_id,
            len(normalized_faces),
            normalized_faces,
        )
        blur_score = self._calc_blur_score(image_bgr)
        self.logger.info(
            '[detect-chain] quality_detection image_id={} blur_score={} blur_threshold={}',
            image_id,
            blur_score,
            self.quality_service.settings.blur_score_threshold,
        )
        validation_result: ValidationOutcome = self.validation_service.validate(
            image_shape=image_bgr.shape,
            faces=normalized_faces,
            blur_score=blur_score,
            image_bgr=image_bgr,
            gray_image=gray,
        )
        quality_details = self.quality_service.evaluate_details(loaded_image.image)
        self.logger.info(
            '[detect-chain] compliance_and_mapping image_id={} validation_passed={} audit_status={} reasons={} details={}',
            image_id,
            validation_result.passed,
            validation_result.auditStatus,
            validation_result.reasons,
            validation_result.auditDetails,
        )

        if validation_result.passed:
            message = '检测完成，图片可进入证件照处理流程'
        elif validation_result.hasFace:
            message = validation_result.message
        else:
            message = '未检测到稳定可处理人像，请上传单人正脸照片'
        audit_result = {
            'status': validation_result.auditStatus,
            'code': validation_result.auditCode,
            'message': validation_result.auditMessage,
            'details': validation_result.auditDetails,
        }
        if validation_result.passed and quality_details['qualityStatus'] != 'passed':
            audit_result = {
                'status': 'warning',
                'code': 'QUALITY_WARNING',
                'message': quality_details['qualityMessage'],
                'details': validation_result.auditDetails
                + [
                    {
                        'code': 'QUALITY_WARNING',
                        'message': quality_details['qualityMessage'],
                    }
                ],
            }

        return DetectOutcome(
            imageId=image_id,
            faceDetected=validation_result.hasFace,
            faceCount=validation_result.faceCount,
            faceBoxes=validation_result.validFaceBoxes,
            blurScore=blur_score,
            poseValid=ERROR_POSE_INVALID not in validation_result.reasons,
            occlusionDetected=ERROR_FACE_OCCLUDED in validation_result.reasons,
            isProcessable=validation_result.passed,
            qualityStatus=quality_details['qualityStatus'],
            qualityMessage=quality_details['qualityMessage'],
            imageWidth=loaded_image.width,
            imageHeight=loaded_image.height,
            imageFormat=loaded_image.format,
            imageMode=loaded_image.mode,
            validationPassed=validation_result.passed,
            reasons=validation_result.reasons,
            suggestion=self._build_suggestion(validation_result, quality_details),
            message=message,
            auditResult=audit_result,
            keypointConfidences=validation_result.keypointConfidences,
            primaryFaceBox=validation_result.primaryFaceBox,
        )

    def detect(self, image_id: str, image_path: str) -> DetectOutcome:
        import numpy as np

        image = Image.open(image_path)
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        loaded = LoadedImage(
            filename=image_path,
            content_type='image/jpeg',
            file_size=0,
            image=rgb,
            image_np=np.array(rgb),
            width=rgb.size[0],
            height=rgb.size[1],
            format=(image.format or 'JPEG').upper(),
            mode=image.mode,
        )
        return self.detect_from_loaded_image(image_id=image_id, loaded_image=loaded)
