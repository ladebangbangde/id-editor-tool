from dataclasses import dataclass


@dataclass
class ErrorDescriptor:
    code: str
    message: str
    status_code: int


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class InvalidImageError(AppError):
    def __init__(self, message: str = 'Invalid image uploaded'):
        super().__init__('INVALID_IMAGE', message, 400)


class NoFaceDetectedError(AppError):
    def __init__(self, message: str = 'No face detected in the uploaded image'):
        super().__init__('NO_FACE_DETECTED', message, 422)


class MultipleFacesDetectedError(AppError):
    def __init__(self, message: str = 'Multiple faces detected in the uploaded image'):
        super().__init__('MULTIPLE_FACES_DETECTED', message, 422)


class ImageTooSmallError(AppError):
    def __init__(self, message: str = 'Image is too small for ID photo generation'):
        super().__init__('IMAGE_TOO_SMALL', message, 422)


class InvalidArgumentError(AppError):
    def __init__(self, message: str = 'Invalid request argument'):
        super().__init__('INVALID_ARGUMENT', message, 400)


class ProcessFailedError(AppError):
    def __init__(self, message: str = 'Image processing failed'):
        super().__init__('PROCESS_FAILED', message, 500)
