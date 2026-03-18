from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', case_sensitive=False, extra='ignore')

    service_name: str = Field(default='id-editor-tool', alias='SERVICE_NAME')
    app_host: str = Field(default='0.0.0.0', alias='APP_HOST')
    app_port: int = Field(default=8000, alias='APP_PORT')
    log_level: str = Field(default='INFO', alias='LOG_LEVEL')
    debug: bool = Field(default=False, alias='DEBUG')

    base_dir: Path = Field(default=Path('.'), alias='BASE_DIR')
    input_dir: Path = Field(default=Path('inputs'), alias='INPUT_DIR')
    output_dir: Path = Field(default=Path('outputs'), alias='OUTPUT_DIR')
    save_intermediate: bool = Field(default=False, alias='SAVE_INTERMEDIATE')
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
    def resolved_input_dir(self) -> Path:
        return (self.base_dir / self.input_dir).resolve()

    @property
    def resolved_output_dir(self) -> Path:
        return (self.base_dir / self.output_dir).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.resolved_input_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return settings
