from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field(default="ai-id-photo-service", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    upload_base_dir: str = Field(default="uploads", alias="UPLOAD_BASE_DIR")
    preview_quality: int = Field(default=88, alias="PREVIEW_QUALITY")
    hd_quality: int = Field(default=95, alias="HD_QUALITY")
    jpeg_dpi: int = Field(default=300, alias="JPEG_DPI")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def upload_dirs(self) -> dict:
        base = Path(self.upload_base_dir)
        return {
            "base": base,
            "preview": base / "preview",
            "hd": base / "hd",
            "print": base / "print",
            "temp": base / "temp",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
