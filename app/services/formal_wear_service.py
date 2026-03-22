from functools import lru_cache

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import InvalidArgumentError
from app.schemas.formal_wear import FormalWearData
from app.services.photo_processor import PhotoProcessor, get_photo_processor


class FormalWearService:
    FALLBACK_GENERATE_BACKGROUND_COLOR = 'white'

    def __init__(self, processor: PhotoProcessor | None = None) -> None:
        self.settings = get_settings()
        self.processor = processor or get_photo_processor()

    def _normalize_gender(self, gender: str | None) -> str | None:
        if not gender:
            return None
        normalized = gender.strip().lower()
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
        return mapping.get(normalized, normalized)

    def _normalize_style(self, style: str | None) -> str:
        if not style:
            return 'formal'
        return style.strip().lower()

    def _normalize_color(self, color: str | None) -> str:
        if not color:
            return 'black'
        # formal wear color is the requested clothing color, not the legacy photo background color.
        return color.strip().lower()

    def _fallback_background_color(self) -> str:
        # The current /formal-wear fallback still reuses the legacy generate pipeline.
        # This background_color is only a temporary compatibility value and does not mean
        # that real formal-wear rendering or clothing replacement has been completed.
        return self.FALLBACK_GENERATE_BACKGROUND_COLOR

    def _build_response(
        self,
        *,
        generated,
        gender: str | None,
        style: str,
        color: str,
        warnings: list[str],
    ) -> FormalWearData:
        return FormalWearData(
            taskId=generated.taskId,
            previewUrl=generated.previewUrl,
            hdUrl=generated.hdUrl,
            gender=gender,
            style=style,
            color=color,
            warnings=warnings,
            previewPath=generated.previewPath,
            hdPath=generated.hdPath,
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
        normalized_gender = self._normalize_gender(gender)
        normalized_style = self._normalize_style(style)
        normalized_color = self._normalize_color(color)
        fallback_background_color = self._fallback_background_color()
        generated = await self.processor.generate_from_upload(
            file=file,
            size_key=self.settings.default_size_key,
            background_color=fallback_background_color,
            enhance=enhance,
            save_output=save_output,
        )
        warnings = list(generated.warnings)
        warnings.append(
            f'Current formal-wear fallback reuses the legacy generate pipeline with backgroundColor={fallback_background_color}; requested clothing color={normalized_color}'
        )
        if normalized_style != 'formal':
            warnings.append(f'Current tool fallback uses the default formal-wear pipeline for style={normalized_style}')
        return self._build_response(
            generated=generated,
            gender=normalized_gender,
            style=normalized_style,
            color=normalized_color,
            warnings=warnings,
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
        normalized_gender = self._normalize_gender(gender)
        normalized_style = self._normalize_style(style)
        normalized_color = self._normalize_color(color)
        fallback_background_color = self._fallback_background_color()
        generated = self.processor.generate_from_path(
            image_path=image_path,
            size_key=self.settings.default_size_key,
            background_color=fallback_background_color,
            enhance=enhance,
            save_output=save_output,
        )
        warnings = list(generated.warnings)
        warnings.append(
            f'Current formal-wear fallback reuses the legacy generate pipeline with backgroundColor={fallback_background_color}; requested clothing color={normalized_color}'
        )
        if normalized_style != 'formal':
            warnings.append(f'Current tool fallback uses the default formal-wear pipeline for style={normalized_style}')
        return self._build_response(
            generated=generated,
            gender=normalized_gender,
            style=normalized_style,
            color=normalized_color,
            warnings=warnings,
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
