from __future__ import annotations

import sys

from loguru import logger

from utils.config import get_settings


def init_logger() -> None:
    settings = get_settings()
    level = (settings.log_level or 'INFO').upper()

    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format=(
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> '
            '| <level>{level}</level> '
            '| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> '
            '| {message}'
        ),
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


def get_logger():
    return logger
