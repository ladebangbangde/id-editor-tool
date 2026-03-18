from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import (
    InvalidArgumentError,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.core.logger import get_logger
from app.schemas.common import FileInfo, SizeInfo
from app.schemas.detect import DetectData
from app.schemas.generate import GenerateData
from app.schemas.layout import LayoutData
from app.services.background import BackgroundService
from app.services.cropper import CropperService
from app.services.enhancer import EnhancerService
from app.services.face_detection import FaceDetectionService
from app.services.layout import LayoutService
from app.services.segmentation import SegmentationService
from app.services.specs import PhotoSpec, get_photo_spec
from app.services.storage import StorageService
from app.utils.file_naming import build_task_id
from app.utils.image_io import load_image_from_bytes, save_image
from app.utils.validators import validate_content_size, validate_upload

logger = get_logger(__name__)


class PhotoProcessor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage = StorageService()
        self.detector = FaceDetectionService()
        self.segmenter = SegmentationService()
        self.background = BackgroundService()
        self.cropper = CropperService()
        self.enhancer = EnhancerService()
        self.layout_service = LayoutService()

    async def read_upload(self, file: UploadFile) -> tuple[bytes, Image.Image]:
        validate_upload(file)
        logger.info('Received upload filename=%s', file.filename)
        content = await file.read()
        validate_content_size(content)
        if not content:
            raise InvalidImageError('Uploaded file is empty')
        image = load_image_from_bytes(content)
        return content, image

    def detect(self, image: Image.Image) -> DetectData:
        result = self.detector.detect(image)
        return DetectData(
            hasFace=result.has_face,
            faceCount=result.face_count,
            width=result.width,
            height=result.height,
            pass_=result.recommended,
            reasons=result.reasons,
            faceBoxes=result.face_boxes,
            recommended=result.recommended,
        )

    def _size_info(self, spec: PhotoSpec) -> SizeInfo:
        return SizeInfo(
            key=spec.key,
            name=spec.name,
            widthPx=spec.width_px,
            heightPx=spec.height_px,
            widthMm=spec.width_mm,
            heightMm=spec.height_mm,
        )

    def _file_info(self, path: Path) -> FileInfo:
        stored = self.storage.stored_file(path)
        return FileInfo(path=str(stored.path.resolve()), url=stored.url)

    def generate(
        self,
        image: Image.Image,
        size_key: str,
        background_color: str,
        enhance: bool,
        save_output: bool,
    ) -> GenerateData:
        logger.info('Start generate pipeline size=%s background=%s enhance=%s', size_key, background_color, enhance)
        spec = get_photo_spec(size_key)
        detect_result = self.detector.detect(image)
        if detect_result.face_count == 0:
            raise NoFaceDetectedError()
        if detect_result.face_count > 1:
            raise MultipleFacesDetectedError()

        task_id = build_task_id('gen')
        task_dir = self.storage.task_dir(task_id)
        logger.info('Task directory prepared: %s', task_dir)

        rgba_foreground = self.segmenter.remove_background(image)
        logger.info('Background removed')
        if self.settings.save_intermediate:
            save_image(rgba_foreground, task_dir / 'foreground.png')

        cropped_rgba = self.cropper.crop(rgba_foreground, spec, detect_result.primary_face)
        logger.info('Portrait cropped to target size')
        if self.settings.save_intermediate:
            save_image(cropped_rgba, task_dir / 'cropped_rgba.png')

        hd_image = self.background.apply(cropped_rgba, background_color)
        logger.info('Background applied')
        if enhance:
            hd_image = self.enhancer.enhance(hd_image)
            logger.info('Enhancement applied')

        preview_image = hd_image.copy()
        preview_image.thumbnail((max(256, spec.width_px), max(256, spec.height_px)), Image.Resampling.LANCZOS)

        hd_path = task_dir / 'id_photo_hd.png'
        preview_path = task_dir / 'id_photo_preview.jpg'
        if save_output:
            save_image(hd_image, hd_path)
            save_image(preview_image, preview_path, quality=self.settings.preview_quality)
            logger.info('Output files saved hd=%s preview=%s', hd_path, preview_path)

        intermediate_files = None
        if self.settings.save_intermediate:
            intermediate_files = {}
            for name in ('foreground.png', 'cropped_rgba.png'):
                path = task_dir / name
                if path.exists():
                    intermediate_files[name] = self._file_info(path)

        warnings = [] if detect_result.recommended else detect_result.reasons.copy()
        preview_info = self._file_info(preview_path) if save_output else FileInfo(path='', url='')
        hd_info = self._file_info(hd_path) if save_output else FileInfo(path='', url='')
        return GenerateData(
            taskId=task_id,
            previewPath=preview_info.path,
            previewUrl=preview_info.url,
            hdPath=hd_info.path,
            hdUrl=hd_info.url,
            backgroundColor=background_color,
            size=self._size_info(spec),
            width=spec.width_px,
            height=spec.height_px,
            warnings=warnings,
            detect=self.detect(image),
            intermediateFiles=intermediate_files,
        )

    def layout_from_photo(self, photo: Image.Image, spec: PhotoSpec, paper: str) -> tuple[Image.Image, int]:
        return self.layout_service.build(photo, spec, paper)

    def resolve_spec(self, size_key: str | None) -> PhotoSpec:
        return get_photo_spec(size_key or self.settings.default_size_key)

    async def generate_from_upload(
        self,
        file: UploadFile,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
    ) -> GenerateData:
        _, image = await self.read_upload(file)
        return self.generate(
            image=image,
            size_key=size_key or self.settings.default_size_key,
            background_color=background_color or self.settings.default_background_color,
            enhance=enhance,
            save_output=save_output,
        )

    async def layout(
        self,
        id_photo: UploadFile | None,
        image: UploadFile | None,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
        paper: str,
    ) -> LayoutData:
        if id_photo is None and image is None:
            raise InvalidArgumentError('Either idPhoto or image must be provided')

        task_id = build_task_id('layout')
        task_dir = self.storage.task_dir(task_id)
        warnings: list[str] = []
        source_hd_info = None
        spec = self.resolve_spec(size_key)

        if id_photo is not None:
            _, photo = await self.read_upload(id_photo)
            source_hd_path = task_dir / 'source_id_photo.png'
            if save_output:
                save_image(photo.convert('RGB'), source_hd_path)
            source_hd_info = self._file_info(source_hd_path)
            hd_image = photo.convert('RGB')
        else:
            generated = await self.generate_from_upload(
                file=image,
                size_key=spec.key,
                background_color=background_color,
                enhance=enhance,
                save_output=True,
            )
            warnings.extend(generated.warnings)
            hd_image = Image.open(generated.hdPath).convert('RGB')
            source_hd_info = FileInfo(path=generated.hdPath, url=generated.hdUrl)
            task_id = generated.taskId
            task_dir = self.storage.task_dir(task_id)
            spec = get_photo_spec(generated.size.key)

        layout_image, count = self.layout_service.build(hd_image, spec, paper)
        layout_path = task_dir / 'layout_6inch.jpg'
        if save_output:
            save_image(layout_image, layout_path, quality=self.settings.hd_quality)
            logger.info('Layout saved path=%s count=%s', layout_path, count)

        layout_info = self._file_info(layout_path) if save_output else FileInfo(path='', url='')
        return LayoutData(
            taskId=task_id,
            layoutPath=layout_info.path,
            layoutUrl=layout_info.url,
            paper=paper,
            count=count,
            photoSize=self._size_info(spec),
            warnings=warnings,
            sourceHd=source_hd_info,
        )
