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
    # 合规审核阈值：双眼必须分别达到的最低置信度
    eye_confidence_threshold: float = Field(default=0.38, validation_alias=AliasChoices('EYE_CONFIDENCE_THRESHOLD'))
    # 合规审核阈值：人脸关键区域（眼鼻嘴）若被遮挡，区域亮度方差会显著下降
    key_region_min_variance: float = Field(default=120.0, validation_alias=AliasChoices('KEY_REGION_MIN_VARIANCE'))
    # 合规审核阈值：判定手部/异物遮挡时，侧边高饱和区域占比阈值
    hand_occlusion_skin_ratio_threshold: float = Field(
        default=0.16,
        validation_alias=AliasChoices('HAND_OCCLUSION_SKIN_RATIO_THRESHOLD'),
    )
    # 证件照睁眼审核阈值：低于该值认为单眼明显闭合（建议范围 0.16~0.24）
    eye_open_ratio_fail_threshold: float = Field(
        default=0.2,
        validation_alias=AliasChoices('EYE_OPEN_RATIO_FAIL_THRESHOLD'),
    )
    # 证件照睁眼审核阈值：低于该值认为轻微眯眼（建议范围 0.23~0.30）
    eye_open_ratio_warn_threshold: float = Field(
        default=0.27,
        validation_alias=AliasChoices('EYE_OPEN_RATIO_WARN_THRESHOLD'),
    )
    # 左右眼开合差异阈值：超过时提示开眼不对称（建议范围 0.10~0.18）
    eye_asymmetry_warn_threshold: float = Field(
        default=0.14,
        validation_alias=AliasChoices('EYE_ASYMMETRY_WARN_THRESHOLD'),
    )
    # 姿态阈值：头部旋转(roll)超过该角度直接失败（建议范围 12~18）
    head_roll_fail_degrees: float = Field(default=14.0, validation_alias=AliasChoices('HEAD_ROLL_FAIL_DEGREES'))
    # 姿态阈值：头部旋转(roll)超过该角度告警（建议范围 7~12）
    head_roll_warn_degrees: float = Field(default=8.0, validation_alias=AliasChoices('HEAD_ROLL_WARN_DEGREES'))
    # 姿态阈值：yaw 超过该值直接失败（建议范围 0.18~0.28）
    yaw_fail_threshold: float = Field(default=0.22, validation_alias=AliasChoices('YAW_FAIL_THRESHOLD'))
    # 姿态阈值：yaw 超过该值告警（建议范围 0.10~0.16）
    yaw_warn_threshold: float = Field(default=0.12, validation_alias=AliasChoices('YAW_WARN_THRESHOLD'))
    # 姿态阈值：pitch 超过该值直接失败（建议范围 0.20~0.30）
    pitch_fail_threshold: float = Field(default=0.24, validation_alias=AliasChoices('PITCH_FAIL_THRESHOLD'))
    # 姿态阈值：pitch 超过该值告警（建议范围 0.10~0.18）
    pitch_warn_threshold: float = Field(default=0.14, validation_alias=AliasChoices('PITCH_WARN_THRESHOLD'))
    # 表情阈值：笑容区域宽度占比超过该值提示表情不规范（建议范围 0.42~0.58）
    smile_ratio_warn_threshold: float = Field(
        default=0.5,
        validation_alias=AliasChoices('SMILE_RATIO_WARN_THRESHOLD'),
    )
    # matte 收敛参数：alpha gamma 越大边缘越保守（建议范围 1.3~2.2）
    composite_alpha_gamma: float = Field(default=1.75, validation_alias=AliasChoices('COMPOSITE_ALPHA_GAMMA'))
    # matte 收敛参数：上半身扩张核（建议范围 1~3）
    composite_dilate_kernel_upper: int = Field(
        default=1,
        validation_alias=AliasChoices('COMPOSITE_DILATE_KERNEL_UPPER'),
    )
    # matte 收敛参数：下半身扩张核，保护衣领肩部（建议范围 2~5）
    composite_dilate_kernel_lower: int = Field(
        default=3,
        validation_alias=AliasChoices('COMPOSITE_DILATE_KERNEL_LOWER'),
    )
    # matte 收敛参数：下半身分界占比（建议范围 0.45~0.65）
    composite_lower_protect_ratio: float = Field(
        default=0.52,
        validation_alias=AliasChoices('COMPOSITE_LOWER_PROTECT_RATIO'),
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
