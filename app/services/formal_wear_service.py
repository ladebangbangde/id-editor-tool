from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.core.exceptions import (
    AppError,
    ImageTooBlurryError,
    InvalidImageError,
    InvalidPoseError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    ProcessFailedError,
    ShoulderNeckIncompleteError,
)
from app.core.logger import get_logger
from app.schemas.common import FileInfo
from app.schemas.formal_wear import FormalWearColor, FormalWearData, FormalWearGender, FormalWearStyle
from app.services.enhancer import EnhancerService
from app.services.face_detection import FaceDetectionResult, FaceDetectionService
from app.services.formal_wear_geometry import FormalWearGeometry
from app.services.formal_wear_renderer import FormalWearRenderer
from app.services.segmentation import SegmentationService
from app.services.storage import StorageService
from app.utils.file_naming import build_task_id
from app.utils.image_io import load_image_from_bytes, load_image_from_path, resolve_input_path, save_image
from app.utils.validators import validate_content_size, validate_upload

logger = get_logger(__name__)


class FormalWearService:
    def __init__(self) -> None:
        self.storage = StorageService()
        self.detector = FaceDetectionService()
        self.segmenter = SegmentationService()
        self.enhancer = EnhancerService()
        self.geometry = FormalWearGeometry()
        self.renderer = FormalWearRenderer()

    async def read_upload(self, file: UploadFile) -> tuple[bytes, Image.Image]:
        validate_upload(file)
        content = await file.read()
        validate_content_size(content)
        if not content:
            raise InvalidImageError('Uploaded file is empty')
        return content, load_image_from_bytes(content)

    def read_image_path(self, path_or_url: str) -> tuple[Path, Image.Image]:
        resolved_path = resolve_input_path(path_or_url)
        return resolved_path, load_image_from_path(resolved_path)

    def _file_info(self, path: Path) -> FileInfo:
        stored = self.storage.stored_file(path)
        return FileInfo(path=str(stored.path.resolve()), url=stored.url)

    def _raise_detect_failure(self, detect_result: FaceDetectionResult) -> None:
        details = {
            'status': detect_result.status,
            'reasonCodes': detect_result.reason_codes,
            'reasons': detect_result.reasons,
            'warningCodes': detect_result.warning_codes,
            'warnings': detect_result.warnings,
            'metrics': detect_result.metrics,
            'blurScore': detect_result.blur_score,
        }
        if detect_result.face_count == 0:
            raise NoFaceDetectedError('未检测到人脸，无法换正装', details)
        if detect_result.face_count > 1:
            raise MultipleFacesDetectedError('检测到多个人脸，换正装仅支持单人照片', details)
        if not detect_result.pose_accepted:
            raise InvalidPoseError('人脸姿态不适合换正装，请使用更正面的照片', details)
        if (detect_result.blur_score or 0.0) < 0.0018:
            raise ImageTooBlurryError('图片太模糊，无法稳定估算肩颈结构', details)
        raise AppError('FORMAL_WEAR_UNSUITABLE', '图像不适合换正装，请更换更清晰且肩颈完整的照片', 422, details)

    def _validate_detection(self, image: Image.Image) -> FaceDetectionResult:
        detect_result = self.detector.detect(image)
        if detect_result.status == 'FAILED' or detect_result.face_count == 0 or detect_result.face_count > 1 or not detect_result.pose_accepted:
            self._raise_detect_failure(detect_result)
        if (detect_result.blur_score or 0.0) < 0.0018:
            raise ImageTooBlurryError(
                '图片太模糊，无法稳定估算肩颈结构',
                {'blurScore': detect_result.blur_score, 'warnings': detect_result.warnings},
            )
        if not detect_result.primary_face:
            raise AppError('FORMAL_WEAR_UNSUITABLE', '图像不适合换正装，未获得有效人脸框', 422)
        return detect_result

    def _build_upper_body_mask(self, foreground: Image.Image, detect_result: FaceDetectionResult) -> Image.Image:
        width, height = foreground.size
        face = detect_result.primary_face or {'x': width // 4, 'y': height // 6, 'width': width // 2, 'height': height // 2}
        x = face['x']
        y = face['y']
        face_w = face['width']
        face_h = face['height']
        alpha = foreground.getchannel('A')
        keep = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(keep)

        draw.ellipse(
            [
                x - face_w * 0.24,
                y - face_h * 0.30,
                x + face_w * 1.24,
                y + face_h * 1.04,
            ],
            fill=255,
        )
        draw.polygon(
            [
                (x + face_w * 0.30, y + face_h * 0.74),
                (x + face_w * 0.18, y + face_h * 1.16),
                (x + face_w * 0.82, y + face_h * 1.16),
                (x + face_w * 0.70, y + face_h * 0.74),
            ],
            fill=235,
        )
        draw.rounded_rectangle(
            [
                x + face_w * 0.36,
                y + face_h * 0.88,
                x + face_w * 0.64,
                min(height - 1, y + face_h * 1.40),
            ],
            radius=max(6, int(face_w * 0.07)),
            fill=210,
        )
        keep = keep.filter(ImageFilter.GaussianBlur(radius=max(6, int(face_w * 0.06))))
        return ImageChops.multiply(alpha, keep)

    def _compose(
        self,
        image: Image.Image,
        foreground: Image.Image,
        detect_result: FaceDetectionResult,
        gender: str,
        style: str,
        color: str,
    ) -> tuple[Image.Image, list[str]]:
        face_box = detect_result.primary_face
        if face_box is None:
            raise AppError('FORMAL_WEAR_UNSUITABLE', '图像不适合换正装，未获得有效人脸框', 422)

        assessment = self.geometry.assess_shoulder_neck(image.size, face_box, detect_result, gender, style)
        if not assessment.passed:
            raise ShoulderNeckIncompleteError(
                '肩颈区域不完整，无法稳定生成正装效果',
                {'reasons': assessment.reasons, 'metrics': assessment.metrics},
            )

        anchors = self.geometry.estimate_anchors(image.size, face_box, gender, style)
        wear_layer = self.renderer.render(image.size, anchors, gender, style, color)
        upper_body = foreground.copy()
        upper_body.putalpha(self._build_upper_body_mask(foreground, detect_result))
        composite = Image.alpha_composite(wear_layer, upper_body)
        return composite, assessment.warnings + detect_result.warnings.copy()

    def _save_outputs(self, task_id: str, composite: Image.Image, save_output: bool) -> tuple[FileInfo, FileInfo]:
        hd_path = self.storage.hd_path(task_id, 'formal_wear_hd.png')
        preview_path = self.storage.preview_path(task_id, 'formal_wear_preview.jpg')
        if save_output:
            save_image(composite, hd_path)
            preview = Image.new('RGB', composite.size, (255, 255, 255))
            preview.paste(composite.convert('RGBA'), mask=composite.getchannel('A'))
            preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
            save_image(preview, preview_path, quality=92)
            return self._file_info(preview_path), self._file_info(hd_path)
        return FileInfo(path='', url=''), FileInfo(path='', url='')

    def generate(
        self,
        image: Image.Image,
        gender: FormalWearGender,
        style: FormalWearStyle,
        color: FormalWearColor,
        enhance: bool,
        save_output: bool,
    ) -> FormalWearData:
        task_id = build_task_id('fw')
        self.storage.category_task_dir('temp', task_id)
        detect_result = self._validate_detection(image)

        try:
            foreground = self.segmenter.remove_background(image)
            composite, warnings = self._compose(image, foreground, detect_result, gender, style, color)
            if enhance:
                alpha = composite.getchannel('A')
                enhanced_rgb = self.enhancer.enhance(Image.alpha_composite(Image.new('RGBA', composite.size, (255, 255, 255, 255)), composite).convert('RGB'))
                composite = enhanced_rgb.convert('RGBA')
                composite.putalpha(alpha)
        except AppError:
            raise
        except Exception as exc:
            logger.exception('Formal wear render/composite failed: %s', exc)
            raise ProcessFailedError('正装绘制或合成失败，请稍后重试') from exc

        preview_info, hd_info = self._save_outputs(task_id, composite, save_output)
        return FormalWearData(
            taskId=task_id,
            previewUrl=preview_info.url,
            hdUrl=hd_info.url,
            gender=gender,
            style=style,
            color=color,
            warnings=warnings,
        )

    async def generate_from_upload(
        self,
        file: UploadFile,
        gender: FormalWearGender,
        style: FormalWearStyle,
        color: FormalWearColor,
        enhance: bool,
        save_output: bool,
    ) -> FormalWearData:
        _, image = await self.read_upload(file)
        return self.generate(image, gender, style, color, enhance, save_output)

    def generate_from_path(
        self,
        image_path: str,
        gender: FormalWearGender,
        style: FormalWearStyle,
        color: FormalWearColor,
        enhance: bool,
        save_output: bool,
    ) -> FormalWearData:
        _, image = self.read_image_path(image_path)
        return self.generate(image, gender, style, color, enhance, save_output)


@lru_cache(maxsize=1)
def get_formal_wear_service() -> FormalWearService:
    return FormalWearService()
