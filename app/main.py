from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_detect import router as detect_router
from app.api.routes_formal_wear import router as formal_wear_router
from app.api.routes_generate import router as generate_router
from app.api.routes_health import router as health_router
from app.api.routes_layout import router as layout_router
from app.api.routes_photo import router as photo_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logger import get_logger, setup_logging
from app.schemas.common import ApiResponse, ErrorBody

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title='id-editor-tool',
    version='1.0.0',
    description='Local HTTP microservice for ID photo detection, generation, and print layout.',
)

app.include_router(health_router)
app.include_router(detect_router)
app.include_router(generate_router)
app.include_router(formal_wear_router)
app.include_router(layout_router)
app.include_router(photo_router)
app.mount(
    settings.normalized_static_mount_path,
    StaticFiles(directory=str(settings.upload_root_path)),
    name='uploads',
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    logger.warning('Handled business error code=%s message=%s', exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse(
            success=False,
            message=exc.message,
            data=exc.details if isinstance(exc.details, dict) else None,
            error=ErrorBody(code=exc.code, message=exc.message, details=exc.details),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception('Unhandled exception: %s', exc)
    return JSONResponse(
        status_code=500,
        content=ApiResponse(
            success=False,
            message='error',
            error=ErrorBody(code='PROCESS_FAILED', message='Unexpected server error'),
        ).model_dump(),
    )


@app.get('/')
def root() -> dict:
    return {
        'service': settings.service_name,
        'docs': '/docs',
        'health': '/health',
        'uploadRoot': str(settings.upload_root_path),
        'staticMountPath': settings.normalized_static_mount_path,
    }
