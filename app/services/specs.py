from dataclasses import dataclass

from app.core.exceptions import InvalidArgumentError


@dataclass(frozen=True)
class PhotoSpec:
    key: str
    name: str
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float


PHOTO_SPECS = {
    'one_inch': PhotoSpec('one_inch', '一寸', 295, 413, 25.0, 35.0),
    'small_one_inch': PhotoSpec('small_one_inch', '小一寸', 260, 378, 22.0, 32.0),
    'two_inch': PhotoSpec('two_inch', '二寸', 413, 579, 35.0, 49.0),
}

BACKGROUND_COLORS = {
    'white': (255, 255, 255),
    'blue': (67, 142, 219),
    'red': (220, 40, 40),
}

LAYOUT_PAPERS = {
    '6inch': {
        'name': '6inch',
        'width_px': 1800,
        'height_px': 1200,
        'margin_px': 60,
        'spacing_px': 24,
    }
}


def get_photo_spec(size_key: str) -> PhotoSpec:
    try:
        return PHOTO_SPECS[size_key]
    except KeyError as exc:
        raise InvalidArgumentError(f'Unsupported sizeKey: {size_key}') from exc


def get_background_color(background_color: str) -> tuple[int, int, int]:
    try:
        return BACKGROUND_COLORS[background_color]
    except KeyError as exc:
        raise InvalidArgumentError(f'Unsupported backgroundColor: {background_color}') from exc


def get_layout_paper(paper: str) -> dict:
    try:
        return LAYOUT_PAPERS[paper]
    except KeyError as exc:
        raise InvalidArgumentError(f'Unsupported paper type: {paper}') from exc
