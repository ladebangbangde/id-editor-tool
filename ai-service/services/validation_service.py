from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import (
    ERROR_FACE_OCCLUDED,
    ERROR_FACE_TOO_SMALL,
    ERROR_HEAD_CROPPED,
    ERROR_IMAGE_TOO_BLURRY,
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
            'primaryFaceBox': self.primaryFaceBox,
        }


class ValidationService:
    reason_priority = [
        ERROR_NO_FACE_DETECTED,
        ERROR_MULTIPLE_FACES_DETECTED,
        ERROR_IMAGE_TOO_BLURRY,
        ERROR_FACE_TOO_SMALL,
        ERROR_POSE_INVALID,
        ERROR_FACE_OCCLUDED,
        ERROR_HEAD_CROPPED,
    ]

    detect_messages = {
        ERROR_NO_FACE_DETECTED: '未检测到人脸',
        ERROR_MULTIPLE_FACES_DETECTED: '检测到多张人脸，不符合证件照制作要求',
        ERROR_IMAGE_TOO_BLURRY: '图片过于模糊，不符合证件照制作要求',
        ERROR_FACE_TOO_SMALL: '人脸区域过小，不符合证件照制作要求',
        ERROR_POSE_INVALID: '人脸姿态不符合证件照要求',
        ERROR_FACE_OCCLUDED: '人脸存在严重遮挡，不符合证件照制作要求',
        ERROR_HEAD_CROPPED: '头部截断过于严重，不符合证件照制作要求',
    }

    generate_messages = {
        ERROR_NO_FACE_DETECTED: '未检测到有效人脸，请上传清晰的单人正脸照片',
        ERROR_MULTIPLE_FACES_DETECTED: '检测到多张人脸，请上传单人正脸照片',
        ERROR_IMAGE_TOO_BLURRY: '图片过于模糊，请上传清晰照片',
        ERROR_FACE_TOO_SMALL: '人脸区域过小，请上传人物主体更清晰的单人正脸照片',
        ERROR_POSE_INVALID: '人脸姿态不符合要求，请上传单人正脸照片',
        ERROR_FACE_OCCLUDED: '人脸遮挡过于严重，请上传无遮挡的单人正脸照片',
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
            return '照片符合证件照制作要求'
        if not reasons:
            return '图片不符合证件照制作要求'
        mapping = self.generate_messages if for_generate else self.detect_messages
        return mapping.get(reasons[0], '图片不符合证件照制作要求')

    def validate(self, image_shape: tuple[int, ...], faces, blur_score: float) -> ValidationOutcome:
        face_count = len(faces)
        has_face = face_count > 0
        primary_face_box = self._build_primary_face_box(faces)
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

        ordered_reasons = [reason for reason in self.reason_priority if reason in reasons]
        passed = not ordered_reasons
        message = self._build_message(ordered_reasons, passed=passed, for_generate=False)
        return ValidationOutcome(
            hasFace=has_face,
            faceCount=face_count,
            passed=passed,
            reasons=ordered_reasons,
            message=message,
            primaryFaceBox=primary_face_box,
        )

    def build_generate_error(self, reasons: list[str]) -> tuple[str, str]:
        ordered_reasons = [reason for reason in self.reason_priority if reason in reasons]
        error_code = ordered_reasons[0] if ordered_reasons else ERROR_NO_FACE_DETECTED
        return error_code, self._build_message(ordered_reasons, passed=False, for_generate=True)
