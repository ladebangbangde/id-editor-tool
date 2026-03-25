from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', case_sensitive=False, extra='ignore')

    app_name: str = Field(default='ai-id-photo-service', validation_alias=AliasChoices('APP_NAME'))
    app_env: str = Field(default='development', validation_alias=AliasChoices('APP_ENV'))
    app_host: str = Field(default='0.0.0.0', validation_alias=AliasChoices('HOST', 'APP_HOST'))
    app_port: int = Field(default=8000, validation_alias=AliasChoices('PORT', 'APP_PORT'))
    app_version: str = Field(default='1.2.0', validation_alias=AliasChoices('APP_VERSION'))
    docs_url: str = Field(default='/docs', validation_alias=AliasChoices('DOCS_URL'))
    redoc_url: str = Field(default='/redoc', validation_alias=AliasChoices('REDOC_URL'))
    openapi_url: str = Field(default='/openapi.json', validation_alias=AliasChoices('OPENAPI_URL'))
    log_level: str = Field(default='INFO', validation_alias=AliasChoices('LOG_LEVEL'))

    upload_base_dir: str = Field(default='uploads', validation_alias=AliasChoices('UPLOAD_BASE_DIR', 'UPLOAD_ROOT'))
    upload_public_prefix: str = Field(default='uploads', validation_alias=AliasChoices('UPLOAD_PUBLIC_PREFIX'))
    original_dir: str = Field(default='original', validation_alias=AliasChoices('ORIGINAL_DIR'))
    preview_dir: str = Field(default='preview', validation_alias=AliasChoices('PREVIEW_DIR'))
    hd_dir: str = Field(default='hd', validation_alias=AliasChoices('HD_DIR'))
    print_dir: str = Field(default='print', validation_alias=AliasChoices('PRINT_DIR'))
    temp_dir: str = Field(default='temp', validation_alias=AliasChoices('TEMP_DIR'))
    save_intermediate: bool = Field(default=True, validation_alias=AliasChoices('SAVE_INTERMEDIATE'))

    max_upload_mb: int = Field(default=15, validation_alias=AliasChoices('MAX_UPLOAD_MB'))
    allowed_image_extensions: tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    allowed_image_content_types: tuple[str, ...] = (
        'image/jpeg',
        'image/png',
        'image/bmp',
        'image/webp',
    )

    min_image_width: int = Field(default=295, validation_alias=AliasChoices('MIN_IMAGE_WIDTH'))
    min_image_height: int = Field(default=413, validation_alias=AliasChoices('MIN_IMAGE_HEIGHT'))
    blur_score_threshold: float = Field(default=0.35, validation_alias=AliasChoices('BLUR_SCORE_THRESHOLD', 'BLUR_THRESHOLD'))
    min_face_width: int = Field(default=60, validation_alias=AliasChoices('MIN_FACE_WIDTH'))
    min_face_height: int = Field(default=60, validation_alias=AliasChoices('MIN_FACE_HEIGHT'))
    min_face_area_ratio: float = Field(default=0.08, validation_alias=AliasChoices('MIN_FACE_AREA_RATIO'))
    min_face_height_ratio: float = Field(default=0.20, validation_alias=AliasChoices('MIN_FACE_HEIGHT_RATIO'))
    max_face_center_offset_ratio: float = Field(default=0.18, validation_alias=AliasChoices('MAX_FACE_CENTER_OFFSET_RATIO'))
    min_face_aspect_ratio: float = Field(default=0.65, validation_alias=AliasChoices('MIN_FACE_ASPECT_RATIO'))
    max_face_aspect_ratio: float = Field(default=1.35, validation_alias=AliasChoices('MAX_FACE_ASPECT_RATIO'))
    occluded_face_aspect_ratio: float = Field(default=0.55, validation_alias=AliasChoices('OCCLUDED_FACE_ASPECT_RATIO'))
    edge_touch_ratio: float = Field(default=0.03, validation_alias=AliasChoices('EDGE_TOUCH_RATIO'))
    # 合规审核阈值：双眼/鼻尖/嘴部关键点检测置信度最低要求
    landmark_confidence_threshold: float = Field(
        default=0.42,
        validation_alias=AliasChoices('LANDMARK_CONFIDENCE_THRESHOLD'),
    )
    # 合规审核阈值：侧脸计数超过该值认为不是标准正脸
    max_profile_face_count: int = Field(default=0, validation_alias=AliasChoices('MAX_PROFILE_FACE_COUNT'))
    # 合规审核阈值：头顶部区域占比，用于帽子/头部遮挡检测
    head_top_region_ratio: float = Field(default=0.22, validation_alias=AliasChoices('HEAD_TOP_REGION_RATIO'))
    # 合规审核阈值：头顶部边缘密度超过阈值时判定疑似帽子/头部遮挡
    headwear_edge_ratio_threshold: float = Field(
        default=0.2,
        validation_alias=AliasChoices('HEADWEAR_EDGE_RATIO_THRESHOLD'),
    )

    min_valid_face_width: int = Field(default=60, validation_alias=AliasChoices('MIN_VALID_FACE_WIDTH'))
    min_valid_face_height: int = Field(default=60, validation_alias=AliasChoices('MIN_VALID_FACE_HEIGHT'))
    multi_face_min_area_ratio: float = Field(default=0.25, validation_alias=AliasChoices('MULTI_FACE_MIN_AREA_RATIO'))
    face_box_iou_threshold: float = Field(default=0.35, validation_alias=AliasChoices('FACE_BOX_IOU_THRESHOLD'))

    default_bg_color: str = Field(default='white', validation_alias=AliasChoices('DEFAULT_BG_COLOR'))
    default_layout_type: str = Field(default='six', validation_alias=AliasChoices('DEFAULT_LAYOUT_TYPE'))
    preview_quality: int = Field(default=88, validation_alias=AliasChoices('PREVIEW_QUALITY'))
    hd_quality: int = Field(default=95, validation_alias=AliasChoices('HD_QUALITY'))
    jpeg_dpi: int = Field(default=300, validation_alias=AliasChoices('JPEG_DPI'))
    segmentation_enabled: bool = Field(default=False, validation_alias=AliasChoices('SEGMENTATION_ENABLED'))

    @property
    def upload_base_path(self) -> Path:
        return Path(self.upload_base_dir)

    @property
    def upload_root(self) -> str:
        return self.upload_base_dir

    @property
    def upload_root_path(self) -> Path:
        return self.upload_base_path

    def _resolve_dir(self, directory: str) -> Path:
        path = Path(directory)
        if path.is_absolute():
            return path
        return self.upload_base_path / path

    @property
    def upload_dirs(self) -> dict[str, Path]:
        return {
            'base': self.upload_base_path,
            'original': self._resolve_dir(self.original_dir),
            'preview': self._resolve_dir(self.preview_dir),
            'hd': self._resolve_dir(self.hd_dir),
            'print': self._resolve_dir(self.print_dir),
            'temp': self._resolve_dir(self.temp_dir),
        }

    @property
    def static_mount_path(self) -> str:
        return f"/{self.upload_public_prefix.strip('/')}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
