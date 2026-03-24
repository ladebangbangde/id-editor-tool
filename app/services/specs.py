from dataclasses import dataclass
from difflib import get_close_matches

from app.core.exceptions import InvalidArgumentError


@dataclass(frozen=True)
class PhotoSpec:
    key: str
    name: str
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float


@dataclass(frozen=True)
class PhotoSpecDefinition:
    spec: PhotoSpec
    aliases: tuple[str, ...] = ()
    category: str = 'id_photo'
    featured: bool = False


PHOTO_SPEC_DEFINITIONS: tuple[PhotoSpecDefinition, ...] = (
    PhotoSpecDefinition(
        spec=PhotoSpec('one_inch', '一寸', 295, 413, 25.0, 35.0),
        aliases=('1inch', 'one-inch', 'yi_cun'),
        featured=True,
    ),
    PhotoSpecDefinition(
        spec=PhotoSpec('small_one_inch', '小一寸', 260, 378, 22.0, 32.0),
        aliases=('small-1inch', 'xiao_yi_cun'),
    ),
    PhotoSpecDefinition(
        spec=PhotoSpec('two_inch', '二寸', 413, 579, 35.0, 49.0),
        aliases=('2inch', 'two-inch', 'er_cun'),
        featured=True,
    ),
    PhotoSpecDefinition(
        spec=PhotoSpec('passport_photo', '护照', 390, 567, 33.0, 48.0),
        aliases=('passport', 'visa', 'travel_document'),
        category='passport',
        featured=True,
    ),
)

PHOTO_SPECS = {item.spec.key: item.spec for item in PHOTO_SPEC_DEFINITIONS}
PHOTO_SPEC_BY_ALIAS = {
    alias.lower(): item.spec.key
    for item in PHOTO_SPEC_DEFINITIONS
    for alias in item.aliases
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


def list_photo_specs() -> list[dict]:
    return [
        {
            'sizeKey': item.spec.key,
            'name': item.spec.name,
            'widthMm': item.spec.width_mm,
            'heightMm': item.spec.height_mm,
            'pixelWidth': item.spec.width_px,
            'pixelHeight': item.spec.height_px,
            'aliases': list(item.aliases),
            'category': item.category,
            'featured': item.featured,
            'canonical': True,
        }
        for item in PHOTO_SPEC_DEFINITIONS
    ]


def supported_size_keys() -> list[str]:
    return [item.spec.key for item in PHOTO_SPEC_DEFINITIONS]


def _normalize_size_key(size_key: str) -> str:
    normalized = size_key.strip()
    if normalized in PHOTO_SPECS:
        return normalized

    alias_hit = PHOTO_SPEC_BY_ALIAS.get(normalized.lower())
    if alias_hit:
        return alias_hit

    all_candidates = supported_size_keys() + list(PHOTO_SPEC_BY_ALIAS.keys())
    suggestion = get_close_matches(normalized.lower(), [item.lower() for item in all_candidates], n=1, cutoff=0.6)
    did_you_mean = None
    if suggestion:
        matched = suggestion[0]
        did_you_mean = PHOTO_SPEC_BY_ALIAS.get(matched, matched)

    details = {
        'inputSizeKey': size_key,
        'supportedSizeKeys': supported_size_keys(),
        'supportedSpecs': list_photo_specs(),
        'didYouMean': did_you_mean,
        'customSizeSupported': False,
        'customSizeHint': '当前不支持 widthMm/heightMm/pixelWidth/pixelHeight 自定义尺寸输入，请从 supportedSizeKeys 选择。',
    }
    message = f'Unsupported sizeKey: {size_key}'
    if did_you_mean:
        message += f'; did you mean: {did_you_mean}'
    raise InvalidArgumentError(message, details)


def get_photo_spec(size_key: str) -> PhotoSpec:
    normalized_key = _normalize_size_key(size_key)
    return PHOTO_SPECS[normalized_key]


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
