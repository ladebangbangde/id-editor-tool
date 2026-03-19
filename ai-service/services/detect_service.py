from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import laplace
from skimage import color, data
from skimage.feature import Cascade

from core.exceptions import AppException, ERROR_FACE_OCCLUDED, ERROR_INVALID_IMAGE, ERROR_POSE_INVALID
from services.validation_service import ValidationService
from utils.image_utils import read_image_array


@dataclass
class DetectOutcome:
    imageId: str
    hasFace: bool
    faceCount: int
    passed: bool
    reasons: list[str]
    blurScore: float
    poseValid: bool
    occlusionDetected: bool
    message: str
    primaryFaceBox: dict | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload['pass'] = payload.pop('passed')
        return payload


class DetectService:
    def __init__(self):
        self.face_detector = Cascade(data.lbp_frontal_face_cascade_filename())
        self.validation_service = ValidationService()

    @staticmethod
    def _calc_blur_score(image_rgb: np.ndarray) -> float:
        gray = color.rgb2gray(image_rgb)
        lap_var = float(laplace(gray).var())
        normalized = min(max(lap_var / 0.02, 0.0), 1.0)
        return round(normalized, 2)

    def detect_faces(self, image_path: str, min_face_size: tuple[int, int] = (60, 60)):
        image = read_image_array(image_path)
        if image.ndim < 2:
            raise AppException('Invalid image content', ERROR_INVALID_IMAGE, 400)
        gray = color.rgb2gray(image[:, :, :3])
        faces = self.face_detector.detect_multi_scale(
            img=gray,
            scale_factor=1.2,
            step_ratio=1,
            min_size=min_face_size,
            max_size=(image.shape[0], image.shape[1]),
        )
        return image, faces

    def detect(
        self,
        image_id: str,
        image_path: str,
        min_face_size: tuple[int, int] = (60, 60),
    ) -> DetectOutcome:
        image, faces = self.detect_faces(image_path, min_face_size=min_face_size)
        face_count = len(faces)
        blur_score = self._calc_blur_score(image[:, :, :3])
        validation = self.validation_service.validate(image.shape, faces, blur_score)
        pose_valid = ERROR_POSE_INVALID not in validation.reasons and face_count == 1
        occlusion_detected = ERROR_FACE_OCCLUDED in validation.reasons

        return DetectOutcome(
            imageId=image_id,
            hasFace=validation.hasFace,
            faceCount=face_count,
            passed=validation.passed,
            reasons=validation.reasons,
            blurScore=blur_score,
            poseValid=pose_valid,
            occlusionDetected=occlusion_detected,
            message=validation.message,
            primaryFaceBox=validation.primaryFaceBox,
        )
