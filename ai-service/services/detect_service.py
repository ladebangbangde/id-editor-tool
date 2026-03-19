from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2

from core.exceptions import ERROR_FACE_OCCLUDED, ERROR_POSE_INVALID
from services.quality_service import QualityService
from services.validation_service import LoadedImage, ValidationOutcome, ValidationService


@dataclass
class DetectOutcome:
    imageId: str
    hasFace: bool
    faceCount: int
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
    message: str
    primaryFaceBox: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DetectService:
    def __init__(self) -> None:
        self.validation_service = ValidationService()
        self.quality_service = QualityService()
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

    def detect_from_loaded_image(self, image_id: str, loaded_image: LoadedImage) -> DetectOutcome:
        image_bgr = cv2.cvtColor(loaded_image.image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        normalized_faces = self._normalize_faces(faces)
        blur_score = self._calc_blur_score(image_bgr)
        validation_result: ValidationOutcome = self.validation_service.validate(
            image_shape=image_bgr.shape,
            faces=normalized_faces,
            blur_score=blur_score,
        )
        quality_status, quality_message = self.quality_service.evaluate(loaded_image.image)

        if validation_result.passed:
            message = '检测完成，图片可进入证件照处理流程'
        elif validation_result.hasFace:
            message = validation_result.message
        else:
            message = '未检测到稳定可处理人像，请上传单人正脸照片'

        return DetectOutcome(
            imageId=image_id,
            hasFace=validation_result.hasFace,
            faceCount=validation_result.faceCount,
            blurScore=blur_score,
            poseValid=ERROR_POSE_INVALID not in validation_result.reasons,
            occlusionDetected=ERROR_FACE_OCCLUDED in validation_result.reasons,
            isProcessable=validation_result.passed,
            qualityStatus=quality_status,
            qualityMessage=quality_message,
            imageWidth=loaded_image.width,
            imageHeight=loaded_image.height,
            imageFormat=loaded_image.format,
            imageMode=loaded_image.mode,
            validationPassed=validation_result.passed,
            reasons=validation_result.reasons,
            message=message,
            primaryFaceBox=validation_result.primaryFaceBox,
        )

    def detect(self, image_id: str, image_path: str) -> DetectOutcome:
        import numpy as np
        from PIL import Image

        image = Image.open(image_path)
        rgb = image.convert('RGB')
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
