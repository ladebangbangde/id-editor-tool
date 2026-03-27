from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', case_sensitive=False, extra='ignore')

    service_name: str = Field(default='id-editor-tool', alias='SERVICE_NAME')
    app_host: str = Field(default='0.0.0.0', alias='APP_HOST')
    app_port: int = Field(default=8000, alias='APP_PORT')
    log_level: str = Field(default='INFO', alias='LOG_LEVEL')
    debug: bool = Field(default=False, alias='DEBUG')

    upload_root: Path = Field(default=Path('/app/uploads'), validation_alias=AliasChoices('UPLOAD_ROOT', 'UPLOAD_BASE_DIR'))
    static_mount_path: str = Field(default='/uploads', alias='STATIC_MOUNT_PATH')
    original_dir_name: str = Field(default='original', alias='ORIGINAL_DIR')
    preview_dir_name: str = Field(default='preview', alias='PREVIEW_DIR')
    hd_dir_name: str = Field(default='hd', alias='HD_DIR')
    print_dir_name: str = Field(default='print', alias='PRINT_DIR')
    temp_dir_name: str = Field(default='temp', alias='TEMP_DIR')
    save_intermediate: bool = Field(default=False, alias='SAVE_INTERMEDIATE')
    enable_wink_hard_fail: bool = Field(default=False, alias='ENABLE_WINK_HARD_FAIL')
    enable_foreground_decontamination: bool = Field(default=True, alias='ENABLE_FOREGROUND_DECONTAMINATION')
    enable_cloth_pollution_check: bool = Field(default=True, alias='ENABLE_CLOTH_POLLUTION_CHECK')
    enable_decontaminated_output_as_default: bool = Field(default=True, alias='ENABLE_DECONTAMINATED_OUTPUT_AS_DEFAULT')
    enable_guided_edge_refinement: bool = Field(default=True, alias='ENABLE_GUIDED_EDGE_REFINEMENT')

    photo_engine: str = Field(default='auto', alias='PHOTO_ENGINE')
    enable_hivision_comparison: bool = Field(default=True, alias='ENABLE_HIVISION_COMPARISON')
    enable_hivision_as_default: bool = Field(default=False, alias='ENABLE_HIVISION_AS_DEFAULT')
    hivision_repo_path: str = Field(default='', alias='HIVISION_REPO_PATH')
    max_upload_size_mb: int = Field(default=15, alias='MAX_UPLOAD_SIZE_MB')
    default_background_color: str = Field(default='blue', alias='DEFAULT_BACKGROUND_COLOR')
    default_size_key: str = Field(default='one_inch', alias='DEFAULT_SIZE_KEY')
    default_layout_paper: str = Field(default='6inch', alias='DEFAULT_LAYOUT_PAPER')

    min_image_width: int = Field(default=400, alias='MIN_IMAGE_WIDTH')
    min_image_height: int = Field(default=400, alias='MIN_IMAGE_HEIGHT')

    preview_quality: int = Field(default=90, alias='PREVIEW_QUALITY')
    hd_quality: int = Field(default=95, alias='HD_QUALITY')

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_root_path(self) -> Path:
        return self.upload_root.resolve()

    @property
    def normalized_static_mount_path(self) -> str:
        return f"/{self.static_mount_path.strip('/')}"

    @property
    def upload_dirs(self) -> dict[str, Path]:
        return {
            'base': self.upload_root_path,
            'original': self.upload_root_path / self.original_dir_name,
            'preview': self.upload_root_path / self.preview_dir_name,
            'hd': self.upload_root_path / self.hd_dir_name,
            'print': self.upload_root_path / self.print_dir_name,
            'temp': self.upload_root_path / self.temp_dir_name,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    for directory in settings.upload_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return settings
