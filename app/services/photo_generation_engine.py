from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from PIL import Image

from app.services.specs import PhotoSpec


@dataclass
class EngineInput:
    source_image: Image.Image
    spec: PhotoSpec
    background_color: str
    enhance: bool
    face_box: dict[str, int] | None


@dataclass
class EngineResult:
    engine_name: str
    hd_image: Image.Image
    preview_image: Image.Image
    preview_quality: int
    foreground_rgba: Image.Image
    debug_images: dict[str, Image.Image] = field(default_factory=dict)
    debug_info: dict[str, str] = field(default_factory=dict)


class PhotoGenerationEngine(Protocol):
    name: str

    def generate(
        self,
        payload: EngineInput,
        stage_reporter: Callable[[str], None] | None = None,
    ) -> EngineResult:
        ...
