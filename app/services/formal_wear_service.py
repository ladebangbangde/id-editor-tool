from functools import lru_cache

from fastapi import UploadFile
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import InvalidArgumentError
from app.schemas.common import FileInfo
from app.schemas.detect import DetectData
from app.schemas.formal_wear import FormalWearData
from app.services.formal_wear_geometry import compute_crop_box, project_face_box
from app.services.formal_wear_renderer import FormalWearRenderer
from app.services.photo_processor import PhotoProcessor, get_photo_processor
from app.services.specs import get_photo_spec
from app.utils.file_naming import build_task_id
from app.utils.image_io import save_image


class FormalWearService:
    FALLBACK_BACKGROUND_COLOR = 'white'

    def __init__(
        self,
        processor: PhotoProcessor | None = None,
        renderer: FormalWearRenderer | None = None,
    ) -> None:
        self.settings = get_settings()
        self.processor = processor or get_photo_processor()
        self.renderer = renderer or FormalWearRenderer()

    def _normalize_gender(self, gender: str | None) -> str:
        normalized = (gender or 'male').strip().lower()
        mapping = {
            'male': 'male',
            'man': 'male',
            'm': 'male',
            '男': 'male',
            'female': 'female',
            'woman': 'female',
            'f': 'female',
            '女': 'female',
        }
        return mapping.get(normalized, 'male')

    def _normalize_style(self, style: str | None) -> str:
        normalized = (style or 'standard').strip().lower()
        mapping = {
            'formal': 'standard',
            'standard': 'standard',
            'business': 'business',
            'simple': 'simple',
        }
        return mapping.get(normalized, 'standard')

    def _normalize_color(self, color: str | None) -> str:
        normalized = (color or 'black').strip().lower()
        mapping = {
            'black': 'black',
            '黑': 'black',
            '黑色': 'black',
            'navy': 'navy',
            'dark_blue': 'navy',
            '藏青': 'navy',
            '藏青色': 'navy',
            'gray': 'gray',
            'grey': 'gray',
            '灰': 'gray',
            '灰色': 'gray',
        }
        return mapping.get(normalized, 'black')

    def _build_response(
        self,
        *,
        task_id: str,
        preview_info: FileInfo,
        hd_info: FileInfo,
        gender: str,
        style: str,
        color: str,
        warnings: list[str],
        detect_summary: DetectData,
        preview_meta: dict[str, int | str],
        hd_meta: dict[str, int | str],
    ) -> FormalWearData:
        return FormalWearData(
            taskId=task_id,
            previewUrl=preview_info.url,
            hdUrl=hd_info.url,
            gender=gender,
            style=style,
            color=color,
            warnings=warnings,
            previewPath=preview_info.path,
            hdPath=hd_info.path,
            detectSummary=detect_summary,
            primaryIssue=detect_summary.primaryIssue,
            primaryMessage=detect_summary.primaryMessage,
            secondaryWarnings=detect_summary.secondaryWarnings,
            qualityStatus=detect_summary.qualityStatus,
            qualityMessage=detect_summary.qualityMessage,
            previewWidth=int(preview_meta['width']),
            previewHeight=int(preview_meta['height']),
            previewFormat=str(preview_meta['format']),
            previewQuality=int(preview_meta['quality']),
            hdWidth=int(hd_meta['width']),
            hdHeight=int(hd_meta['height']),
            hdFormat=str(hd_meta['format']),
            hdQuality=int(hd_meta['quality']),
        )

    def _save_outputs(
        self,
        task_id: str,
        dressed_rgba: Image.Image,
        enhance: bool,
        save_output: bool,
    ) -> tuple[FileInfo, FileInfo, dict[str, int | str], dict[str, int | str]]:
        hd_image = self.processor.background.apply(dressed_rgba, self.FALLBACK_BACKGROUND_COLOR)
        if enhance:
            hd_image = self.processor.enhancer.enhance(hd_image)

        preview_image, preview_quality = self.processor._build_preview_image(hd_image)

        hd_path = self.processor.storage.hd_path(task_id, 'formal_wear_hd.png')
        preview_path = self.processor.storage.preview_path(task_id, 'formal_wear_preview.jpg')
        if save_output:
            save_image(hd_image, hd_path)
            save_image(preview_image, preview_path, quality=preview_quality)

        hd_info = self.processor._file_info(hd_path) if save_output else FileInfo(path='', url='')
        preview_info = self.processor._file_info(preview_path) if save_output else FileInfo(path='', url='')
        preview_meta = {'width': preview_image.width, 'height': preview_image.height, 'format': 'JPEG', 'quality': preview_quality}
        hd_meta = {'width': hd_image.width, 'height': hd_image.height, 'format': 'PNG', 'quality': 100}
        return preview_info, hd_info, preview_meta, hd_meta

    def _render_formal_wear(
        self,
        *,
        image: Image.Image,
        gender: str,
        style: str,
        color: str,
        enhance: bool,
        save_output: bool,
    ) -> FormalWearData:
        detect_result = self.processor.detector.detect(image)
        if not detect_result.can_generate:
            self.processor._raise_detect_failure(detect_result)
        detect_summary = self.processor._build_detect_data(detect_result)

        spec = get_photo_spec(self.settings.default_size_key)
        rgba_foreground = self.processor.segmenter.remove_background(image)
        crop_box = compute_crop_box(image.width, image.height, spec, detect_result.primary_face)
        cropped_rgba = rgba_foreground.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))
        cropped_rgba = cropped_rgba.resize((spec.width_px, spec.height_px), Image.Resampling.LANCZOS)
        cropped_face_box = project_face_box(detect_result.primary_face, crop_box, spec.width_px, spec.height_px)

        dressed_rgba, render_warnings = self.renderer.render(
            cropped_rgba,
            face_box=cropped_face_box,
            gender=gender,
            style=style,
            color=color,
        )

        task_id = build_task_id('formal')
        if save_output:
            self.processor.storage.category_task_dir('temp', task_id)
            if self.settings.save_intermediate:
                save_image(dressed_rgba, self.processor.storage.temp_path(task_id, 'formal_wear_rgba.png'))

        preview_info, hd_info, preview_meta, hd_meta = self._save_outputs(task_id, dressed_rgba, enhance, save_output)
        warnings = list(detect_result.warnings) + render_warnings
        warnings.append('换装结果基于轻量矢量正装渲染，建议优先使用高清图用于正式提交')
        return self._build_response(
            task_id=task_id,
            preview_info=preview_info,
            hd_info=hd_info,
            gender=gender,
            style=style,
            color=color,
            warnings=warnings,
            detect_summary=detect_summary,
            preview_meta=preview_meta,
            hd_meta=hd_meta,
        )

    async def create_from_upload(
        self,
        *,
        file: UploadFile,
        gender: str | None,
        style: str | None,
        color: str | None,
        enhance: bool,
        save_output: bool,
    ) -> FormalWearData:
        _, image = await self.processor.read_upload(file)
        return self._render_formal_wear(
            image=image,
            gender=self._normalize_gender(gender),
            style=self._normalize_style(style),
            color=self._normalize_color(color),
            enhance=enhance,
            save_output=save_output,
        )

    def create_from_path(
        self,
        *,
        image_path: str,
        gender: str | None,
        style: str | None,
        color: str | None,
        enhance: bool,
        save_output: bool,
    ) -> FormalWearData:
        _, image = self.processor.read_image_path(image_path)
        return self._render_formal_wear(
            image=image,
            gender=self._normalize_gender(gender),
            style=self._normalize_style(style),
            color=self._normalize_color(color),
            enhance=enhance,
            save_output=save_output,
        )

    async def create(
        self,
        *,
        file: UploadFile | None,
        image_path: str | None,
        gender: str | None,
        style: str | None,
        color: str | None,
        enhance: bool,
        save_output: bool,
    ) -> FormalWearData:
        if file is None and not image_path:
            raise InvalidArgumentError('Either file or imagePath must be provided')
        if file is not None:
            return await self.create_from_upload(
                file=file,
                gender=gender,
                style=style,
                color=color,
                enhance=enhance,
                save_output=save_output,
            )
        return self.create_from_path(
            image_path=image_path or '',
            gender=gender,
            style=style,
            color=color,
            enhance=enhance,
            save_output=save_output,
        )


@lru_cache(maxsize=1)
def get_formal_wear_service() -> FormalWearService:
    return FormalWearService()
