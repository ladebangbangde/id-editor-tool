from functools import lru_cache
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
from app.utils.image_io import load_image_from_bytes, load_image_from_path, resolve_input_path, save_image
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

    def read_image_path(self, path_or_url: str) -> tuple[Path, Image.Image]:
        resolved_path = resolve_input_path(path_or_url)
        logger.info('Reading shared image path=%s', resolved_path)
        return resolved_path, load_image_from_path(resolved_path)

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
        self.storage.category_task_dir('temp', task_id)
        logger.info('Task directories prepared task_id=%s upload_root=%s', task_id, self.storage.upload_root)

        rgba_foreground = self.segmenter.remove_background(image)
        logger.info('Background removed')
        if self.settings.save_intermediate:
            save_image(rgba_foreground, self.storage.temp_path(task_id, 'foreground.png'))

        cropped_rgba = self.cropper.crop(rgba_foreground, spec, detect_result.primary_face)
        logger.info('Portrait cropped to target size')
        if self.settings.save_intermediate:
            save_image(cropped_rgba, self.storage.temp_path(task_id, 'cropped_rgba.png'))

        hd_image = self.background.apply(cropped_rgba, background_color)
        logger.info('Background applied')
        if enhance:
            hd_image = self.enhancer.enhance(hd_image)
            logger.info('Enhancement applied')

        preview_image = hd_image.copy()
        preview_image.thumbnail((max(256, spec.width_px), max(256, spec.height_px)), Image.Resampling.LANCZOS)

        hd_path = self.storage.hd_path(task_id, 'id_photo_hd.png')
        preview_path = self.storage.preview_path(task_id, 'id_photo_preview.jpg')
        if save_output:
            save_image(hd_image, hd_path)
            save_image(preview_image, preview_path, quality=self.settings.preview_quality)
            logger.info('Output files saved hd=%s preview=%s', hd_path, preview_path)

        intermediate_files = None
        if self.settings.save_intermediate:
            intermediate_files = {}
            for name in ('foreground.png', 'cropped_rgba.png'):
                path = self.storage.temp_path(task_id, name)
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

    def generate_from_path(
        self,
        image_path: str,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
    ) -> GenerateData:
        _, image = self.read_image_path(image_path)
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
        id_photo_path: str | None,
        image_path: str | None,
        size_key: str | None,
        background_color: str | None,
        enhance: bool,
        save_output: bool,
        paper: str,
    ) -> LayoutData:
        if id_photo is None and image is None and id_photo_path is None and image_path is None:
            raise InvalidArgumentError('Either idPhoto, image, idPhotoPath, or imagePath must be provided')

        task_id = build_task_id('layout')
        self.storage.category_task_dir('temp', task_id)
        warnings: list[str] = []
        source_hd_info = None
        spec = self.resolve_spec(size_key)

        if id_photo is not None or id_photo_path is not None:
            if id_photo is not None:
                _, photo = await self.read_upload(id_photo)
            else:
                _, photo = self.read_image_path(id_photo_path or '')
            source_hd_path = self.storage.temp_path(task_id, 'source_id_photo.png')
            if save_output:
                save_image(photo.convert('RGB'), source_hd_path)
                source_hd_info = self._file_info(source_hd_path)
            else:
                source_hd_info = FileInfo(path='', url='')
            hd_image = photo.convert('RGB')
        else:
            if image is not None:
                generated = await self.generate_from_upload(
                    file=image,
                    size_key=spec.key,
                    background_color=background_color,
                    enhance=enhance,
                    save_output=True,
                )
            else:
                generated = self.generate_from_path(
                    image_path=image_path or '',
                    size_key=spec.key,
                    background_color=background_color,
                    enhance=enhance,
                    save_output=True,
                )
            warnings.extend(generated.warnings)
            hd_image = Image.open(generated.hdPath).convert('RGB')
            source_hd_info = FileInfo(path=generated.hdPath, url=generated.hdUrl)
            task_id = generated.taskId
            self.storage.category_task_dir('temp', task_id)
            spec = get_photo_spec(generated.size.key)

        layout_image, count = self.layout_service.build(hd_image, spec, paper)
        layout_path = self.storage.print_path(task_id, 'layout_6inch.jpg')
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


@lru_cache(maxsize=1)
def get_photo_processor() -> PhotoProcessor:
    return PhotoProcessor()
