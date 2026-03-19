from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="ai-id-photo-service", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Host-shared uploads mount root in container.
    # e.g. /data/uploads (recommended in docker-compose)
    upload_base_dir: str = Field(default="uploads", alias="UPLOAD_BASE_DIR")
    # Public URL-like prefix used by upstream server records.
    # Keep this as "uploads" to stay compatible with server DB conventions.
    upload_public_prefix: str = Field(default="uploads", alias="UPLOAD_PUBLIC_PREFIX")

    preview_quality: int = Field(default=88, alias="PREVIEW_QUALITY")
    hd_quality: int = Field(default=95, alias="HD_QUALITY")
    jpeg_dpi: int = Field(default=300, alias="JPEG_DPI")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @property
    def upload_base_path(self) -> Path:
        return Path(self.upload_base_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
