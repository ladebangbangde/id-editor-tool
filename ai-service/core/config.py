from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', case_sensitive=False, extra='ignore')

    app_name: str = Field(default='ai-id-photo-service', validation_alias=AliasChoices('APP_NAME'))
    host: str = Field(default='0.0.0.0', validation_alias=AliasChoices('HOST', 'APP_HOST'))
    port: int = Field(default=8000, validation_alias=AliasChoices('PORT', 'APP_PORT'))
    log_level: str = Field(default='INFO', validation_alias=AliasChoices('LOG_LEVEL'))

    upload_root: str = Field(default='uploads', validation_alias=AliasChoices('UPLOAD_ROOT', 'UPLOAD_BASE_DIR'))
    original_dir: str = Field(default='original', validation_alias=AliasChoices('ORIGINAL_DIR'))
    preview_dir: str = Field(default='preview', validation_alias=AliasChoices('PREVIEW_DIR'))
    hd_dir: str = Field(default='hd', validation_alias=AliasChoices('HD_DIR'))
    print_dir: str = Field(default='print', validation_alias=AliasChoices('PRINT_DIR'))
    temp_dir: str = Field(default='temp', validation_alias=AliasChoices('TEMP_DIR'))

    max_upload_mb: int = Field(default=15, validation_alias=AliasChoices('MAX_UPLOAD_MB'))
    default_bg_color: str = Field(default='white', validation_alias=AliasChoices('DEFAULT_BG_COLOR'))
    default_layout_type: str = Field(default='six', validation_alias=AliasChoices('DEFAULT_LAYOUT_TYPE'))
    save_intermediate: bool = Field(default=True, validation_alias=AliasChoices('SAVE_INTERMEDIATE'))

    preview_quality: int = Field(default=88, validation_alias=AliasChoices('PREVIEW_QUALITY'))
    hd_quality: int = Field(default=95, validation_alias=AliasChoices('HD_QUALITY'))
    jpeg_dpi: int = Field(default=300, validation_alias=AliasChoices('JPEG_DPI'))

    blur_threshold: float = Field(default=30.0, validation_alias=AliasChoices('BLUR_THRESHOLD'))
    min_image_width: int = Field(default=300, validation_alias=AliasChoices('MIN_IMAGE_WIDTH'))
    min_image_height: int = Field(default=400, validation_alias=AliasChoices('MIN_IMAGE_HEIGHT'))
    min_face_width: int = Field(default=100, validation_alias=AliasChoices('MIN_FACE_WIDTH'))
    min_face_height: int = Field(default=120, validation_alias=AliasChoices('MIN_FACE_HEIGHT'))
    min_face_area_ratio: float = Field(default=0.03, validation_alias=AliasChoices('MIN_FACE_AREA_RATIO'))
    min_face_height_ratio: float = Field(default=0.18, validation_alias=AliasChoices('MIN_FACE_HEIGHT_RATIO'))
    max_face_center_offset_ratio: float = Field(
        default=0.18,
        validation_alias=AliasChoices('MAX_FACE_CENTER_OFFSET_RATIO'),
    )
    min_face_aspect_ratio: float = Field(default=0.65, validation_alias=AliasChoices('MIN_FACE_ASPECT_RATIO'))
    max_face_aspect_ratio: float = Field(default=1.1, validation_alias=AliasChoices('MAX_FACE_ASPECT_RATIO'))
    occluded_face_aspect_ratio: float = Field(
        default=0.72,
        validation_alias=AliasChoices('OCCLUDED_FACE_ASPECT_RATIO'),
    )
    edge_touch_ratio: float = Field(default=0.02, validation_alias=AliasChoices('EDGE_TOUCH_RATIO'))

    min_valid_face_width: int = Field(default=60, validation_alias=AliasChoices('MIN_VALID_FACE_WIDTH'))
    min_valid_face_height: int = Field(default=60, validation_alias=AliasChoices('MIN_VALID_FACE_HEIGHT'))
    multi_face_min_area_ratio: float = Field(default=0.25, validation_alias=AliasChoices('MULTI_FACE_MIN_AREA_RATIO'))
    face_box_iou_threshold: float = Field(default=0.35, validation_alias=AliasChoices('FACE_BOX_IOU_THRESHOLD'))

    static_mount_path: str = '/uploads'

    def _resolve_dir(self, directory: str) -> Path:
        path = Path(directory)
        if path.is_absolute():
            return path
        return self.upload_root_path / path

    @property
    def upload_root_path(self) -> Path:
        return Path(self.upload_root)

    @property
    def upload_dirs(self) -> dict[str, Path]:
        return {
            'base': self.upload_root_path,
            'original': self._resolve_dir(self.original_dir),
            'preview': self._resolve_dir(self.preview_dir),
            'hd': self._resolve_dir(self.hd_dir),
            'print': self._resolve_dir(self.print_dir),
            'temp': self._resolve_dir(self.temp_dir),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
