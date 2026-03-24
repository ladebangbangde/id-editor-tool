from dataclasses import dataclass
from typing import Any


@dataclass
class ErrorDescriptor:
    code: str
    message: str
    status_code: int


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class InvalidImageError(AppError):
    def __init__(self, message: str = 'Invalid image uploaded', details: Any = None):
        super().__init__('INVALID_IMAGE', message, 400, details)


class NoFaceDetectedError(AppError):
    def __init__(self, message: str = 'No face detected in the uploaded image', details: Any = None):
        super().__init__('NO_FACE_DETECTED', message, 422, details)


class MultipleFacesDetectedError(AppError):
    def __init__(self, message: str = 'Multiple faces detected in the uploaded image', details: Any = None):
        super().__init__('MULTIPLE_FACES_DETECTED', message, 422, details)


class ImageTooSmallError(AppError):
    def __init__(self, message: str = 'Image is too small for ID photo generation'):
        super().__init__('IMAGE_TOO_SMALL', message, 422)


class InvalidArgumentError(AppError):
    def __init__(self, message: str = 'Invalid request argument'):
        super().__init__('INVALID_ARGUMENT', message, 400)


class ProcessFailedError(AppError):
    def __init__(self, message: str = 'Image processing failed'):
        super().__init__('PROCESS_FAILED', message, 500)


class FaceOccludedError(AppError):
    def __init__(self, message: str = 'Face is occluded and unsuitable for ID photo generation', details: Any = None):
        super().__init__('FACE_OCCLUDED', message, 422, details)


class EyeOccludedError(AppError):
    def __init__(self, message: str = 'One or both eyes are occluded', details: Any = None):
        super().__init__('EYE_OCCLUDED', message, 422, details)


class InvalidPoseError(AppError):
    def __init__(self, message: str = 'Face pose is not suitable for an ID photo', details: Any = None):
        super().__init__('INVALID_POSE', message, 422, details)


class LandmarkUnstableError(AppError):
    def __init__(self, message: str = 'Facial landmarks are unstable or incomplete', details: Any = None):
        super().__init__('LANDMARK_UNSTABLE', message, 422, details)


class BadCompositionError(AppError):
    def __init__(self, message: str = 'Image composition is not safe for ID photo cropping', details: Any = None):
        super().__init__('BAD_COMPOSITION', message, 422, details)


class HeadAccessoryError(AppError):
    def __init__(self, message: str = 'Head accessory detected and unsuitable for ID photo generation', details: Any = None):
        super().__init__('HEAD_ACCESSORY', message, 422, details)


class HandOcclusionError(AppError):
    def __init__(self, message: str = 'Hand occludes face and unsuitable for ID photo generation', details: Any = None):
        super().__init__('HAND_OCCLUSION', message, 422, details)


class BadLightingError(AppError):
    def __init__(self, message: str = 'Lighting is unsuitable for ID photo generation', details: Any = None):
        super().__init__('BAD_LIGHTING', message, 422, details)



class ImageTooBlurryError(AppError):
    def __init__(self, message: str = 'Image is too blurry for formal wear generation', details: Any = None):
        super().__init__('IMAGE_TOO_BLURRY', message, 422, details)


class ShoulderNeckIncompleteError(AppError):
    def __init__(self, message: str = 'Shoulder and neck area is incomplete for formal wear generation', details: Any = None):
        super().__init__('SHOULDER_NECK_INCOMPLETE', message, 422, details)
