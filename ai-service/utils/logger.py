from __future__ import annotations

import sys
from typing import Any

from loguru import logger

from utils.config import get_settings


def _serialize_extra(extra: dict[str, Any]) -> str:
    pairs = [f'{key}={value}' for key, value in extra.items() if value is not None and value != '']
    return f" | {' '.join(pairs)}" if pairs else ''


def _patch_record(record: dict[str, Any]) -> None:
    record['extra']['context'] = _serialize_extra(record['extra'])


def init_logger() -> None:
    settings = get_settings()
    level = (settings.log_level or 'INFO').upper()

    logger.remove()
    logger.configure(patcher=_patch_record)
    logger.add(
        sys.stdout,
        level=level,
        format=(
            '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> '
            '| <level>{level}</level> '
            '| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> '
            '| {message}{extra[context]}'
        ),
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


def get_logger(**context: Any):
    if context:
        return logger.bind(**context)
    return logger
