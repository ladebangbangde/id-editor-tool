import sys

from loguru import logger

from utils.config import get_settings


def init_logger() -> None:
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        enqueue=True,
    )


def get_logger():
    return logger
