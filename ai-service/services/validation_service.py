from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import (
    ERROR_FACE_OCCLUDED,
    ERROR_FACE_TOO_SMALL,
    ERROR_HEAD_CROPPED,
    ERROR_IMAGE_TOO_BLURRY,
    ERROR_IMAGE_TOO_SMALL,
    ERROR_MULTIPLE_FACES_DETECTED,
    ERROR_NO_FACE_DETECTED,
    ERROR_POSE_INVALID,
)
from services.quality_service import QualityService


@dataclass
class ValidationOutcome:
    hasFace: bool
    faceCount: int
    passed: bool
    blurScore: float | None
    imageWidth: int
    imageHeight: int
    reasons: list[str] = field(default_factory=list)
    message: str = '未检测到有效人脸'
    primaryFaceBox: dict | None = None

    def to_dict(self) -> dict:
        return {
            'hasFace': self.hasFace,
            'faceCount': self.faceCount,
            'pass': self.passed,
            'reasons': self.reasons,
            'message': self.message,
            'blurScore': self.blurScore,
            'primaryFaceBox': self.primaryFaceBox,
            'imageWidth': self.imageWidth,
            'imageHeight': self.imageHeight,
        }


class ValidationService:
    reason_priority = [
        ERROR_NO_FACE_DETECTED,
        ERROR_MULTIPLE_FACES_DETECTED,
        ERROR_IMAGE_TOO_BLURRY,
        ERROR_FACE_TOO_SMALL,
        ERROR_POSE_INVALID,
        ERROR_IMAGE_TOO_SMALL,
        ERROR_FACE_OCCLUDED,
        ERROR_HEAD_CROPPED,
    ]

    detect_messages = {
        ERROR_NO_FACE_DETECTED: '未检测到人脸，请上传清晰的单人正脸照片',
        ERROR_MULTIPLE_FACES_DETECTED: '检测到多张人脸，不符合证件照制作要求，请上传单人照片',
        ERROR_IMAGE_TOO_BLURRY: '图片过于模糊，请上传更清晰的照片',
        ERROR_FACE_TOO_SMALL: '人脸区域过小，请上传人物更近、更清晰的照片',
        ERROR_POSE_INVALID: '人物姿态不符合要求，请上传正脸且身体基本正对镜头的照片',
        ERROR_IMAGE_TOO_SMALL: '图片分辨率过低，请上传更高分辨率照片',
        ERROR_FACE_OCCLUDED: '人脸存在明显遮挡，请上传无遮挡的正脸照片',
        ERROR_HEAD_CROPPED: '头部截断过于严重，请上传头部完整的正脸照片',
    }

    generate_messages = {
        ERROR_NO_FACE_DETECTED: '未检测到有效人脸，请上传清晰的单人正脸照片',
        ERROR_MULTIPLE_FACES_DETECTED: '检测到多张人脸，请上传单人正脸照片',
        ERROR_IMAGE_TOO_BLURRY: '图片过于模糊，请上传清晰照片',
        ERROR_FACE_TOO_SMALL: '人脸区域过小，请上传人物更近、更清晰的照片',
        ERROR_POSE_INVALID: '人物姿态不符合要求，请上传正脸且身体基本正对镜头的照片',
        ERROR_IMAGE_TOO_SMALL: '图片分辨率过低，请上传更高分辨率照片',
        ERROR_FACE_OCCLUDED: '人脸存在明显遮挡，请上传无遮挡的单人正脸照片',
        ERROR_HEAD_CROPPED: '头部截断过于严重，请上传头部完整的单人正脸照片',
    }

    def __init__(self):
        self.quality_service = QualityService()

    @staticmethod
    def _build_primary_face_box(faces) -> dict | None:
        if len(faces) != 1:
            return None
        face = faces[0]
        return {
            'x': int(face['c']),
            'y': int(face['r']),
            'width': int(face['width']),
            'height': int(face['height']),
        }

    def _build_message(self, reasons: list[str], passed: bool, for_generate: bool = False) -> str:
        if passed:
            return '图片符合证件照制作要求'
        if not reasons:
            return '图片不符合证件照制作要求'
        mapping = self.generate_messages if for_generate else self.detect_messages
        return mapping.get(reasons[0], '图片不符合证件照制作要求')

    def validate(self, image_shape: tuple[int, ...], faces, blur_score: float | None) -> ValidationOutcome:
        image_height, image_width = image_shape[:2]
        face_count = len(faces)
        has_face = face_count > 0
        primary_face_box = self._build_primary_face_box(faces)
        reasons: list[str] = []

        if self.quality_service.is_image_too_small(image_shape):
            reasons.append(ERROR_IMAGE_TOO_SMALL)

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

        ordered_reasons = [reason for reason in self.reason_priority if reason in reasons]
        passed = not ordered_reasons
        message = self._build_message(ordered_reasons, passed=passed, for_generate=False)
        return ValidationOutcome(
            hasFace=has_face,
            faceCount=face_count,
            passed=passed,
            blurScore=blur_score,
            reasons=ordered_reasons,
            message=message,
            primaryFaceBox=primary_face_box,
            imageWidth=image_width,
            imageHeight=image_height,
        )

    def build_generate_error(self, reasons: list[str]) -> tuple[str, str]:
        ordered_reasons = [reason for reason in self.reason_priority if reason in reasons]
        error_code = ordered_reasons[0] if ordered_reasons else ERROR_NO_FACE_DETECTED
        return error_code, self._build_message(ordered_reasons, passed=False, for_generate=True)
