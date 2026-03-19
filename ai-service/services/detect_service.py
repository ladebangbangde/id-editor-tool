from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import laplace
from skimage import color, data
from skimage.feature import Cascade

from core.exceptions import AppException, ERROR_INVALID_IMAGE
from utils.image_utils import read_image_array


@dataclass
class DetectOutcome:
    imageId: str
    hasFace: bool
    faceCount: int
    blurScore: float
    poseValid: bool
    occlusionDetected: bool
    message: str
    primaryFaceBox: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DetectService:
    def __init__(self):
        self.face_detector = Cascade(data.lbp_frontal_face_cascade_filename())

    @staticmethod
    def _calc_blur_score(image_rgb: np.ndarray) -> float:
        gray = color.rgb2gray(image_rgb)
        lap_var = float(laplace(gray).var())
        normalized = min(max(lap_var / 0.02, 0.0), 1.0)
        return round(normalized, 2)

    def detect_faces(self, image_path: str):
        image = read_image_array(image_path)
        if image.ndim < 2:
            raise AppException('Invalid image content', ERROR_INVALID_IMAGE, 400)
        gray = color.rgb2gray(image[:, :, :3])
        faces = self.face_detector.detect_multi_scale(
            img=gray,
            scale_factor=1.2,
            step_ratio=1,
            min_size=(60, 60),
            max_size=(image.shape[0], image.shape[1]),
        )
        return image, faces

    def detect(self, image_id: str, image_path: str) -> DetectOutcome:
        image, faces = self.detect_faces(image_path)
        face_count = len(faces)
        blur_score = self._calc_blur_score(image[:, :, :3])
        has_face = face_count > 0
        pose_valid = face_count == 1
        occlusion_detected = False

        if not has_face:
            msg = '未检测到人脸'
        elif face_count > 1:
            msg = '检测到多张人脸，建议更换照片'
        elif blur_score < 0.35:
            msg = '照片清晰度不足，建议更换照片'
        else:
            msg = '照片可用于制作证件照'

        primary_face_box = None
        if face_count > 0:
            largest_face = max(faces, key=lambda box: box['width'] * box['height'])
            primary_face_box = {
                'x': int(largest_face['c']),
                'y': int(largest_face['r']),
                'width': int(largest_face['width']),
                'height': int(largest_face['height']),
            }

        return DetectOutcome(
            imageId=image_id,
            hasFace=has_face,
            faceCount=face_count,
            blurScore=blur_score,
            poseValid=pose_valid,
            occlusionDetected=occlusion_detected,
            message=msg,
            primaryFaceBox=primary_face_box,
        )
