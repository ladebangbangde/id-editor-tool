from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BackgroundColorOption:
    key: str
    nameZh: str
    nameEn: str
    rgb: tuple[int, int, int]
    hex: str
    description: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data['rgb'] = list(self.rgb)
        return data


BACKGROUND_COLOR_OPTIONS: tuple[BackgroundColorOption, ...] = (
    BackgroundColorOption(
        key='white',
        nameZh='白底',
        nameEn='White',
        rgb=(255, 255, 255),
        hex='#FFFFFF',
        description='通用证件照底色，适合多数报名与档案场景。',
    ),
    BackgroundColorOption(
        key='blue',
        nameZh='蓝底',
        nameEn='Blue',
        rgb=(67, 142, 219),
        hex='#438EDB',
        description='常用于毕业、工作证、考试报名等场景。',
    ),
    BackgroundColorOption(
        key='red',
        nameZh='红底',
        nameEn='Red',
        rgb=(220, 40, 40),
        hex='#DC2828',
        description='常用于签证、社保和部分职业资格报名。',
    ),
)

BACKGROUND_COLORS = {item.key: item.rgb for item in BACKGROUND_COLOR_OPTIONS}
ALLOWED_BACKGROUND_COLORS = tuple(BACKGROUND_COLORS.keys())


def list_background_colors() -> list[dict]:
    return [item.to_dict() for item in BACKGROUND_COLOR_OPTIONS]
