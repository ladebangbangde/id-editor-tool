from dataclasses import asdict, dataclass

import cv2

from utils.image_utils import read_image_cv


@dataclass
class DetectOutcome:
    imageId: str
    hasFace: bool
    faceCount: int
    blurScore: float
    poseValid: bool
    occlusionDetected: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class DetectService:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def _calc_blur_score(image_bgr) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        normalized = min(max(lap_var / 500.0, 0.0), 1.0)
        return round(float(normalized), 2)

    def detect(self, image_id: str, image_path: str) -> DetectOutcome:
        image = read_image_cv(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        face_count = len(faces)
        blur_score = self._calc_blur_score(image)
        has_face = face_count > 0
        pose_valid = face_count == 1
        occlusion_detected = False

        if not has_face:
            msg = "未检测到人脸"
        elif face_count > 1:
            msg = "检测到多张人脸，建议更换照片"
        elif blur_score < 0.35:
            msg = "照片清晰度不足，建议更换照片"
        else:
            msg = "照片可用于制作证件照"

        return DetectOutcome(
            imageId=image_id,
            hasFace=has_face,
            faceCount=face_count,
            blurScore=blur_score,
            poseValid=pose_valid,
            occlusionDetected=occlusion_detected,
            message=msg,
        )
