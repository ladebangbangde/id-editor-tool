from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from api.detect_api import router as detect_router
from api.generate_api import router as generate_router
from api.health_api import router as health_router
from api.print_api import router as print_router
from api.upload_debug_api import router as upload_debug_router
from core.config import get_settings
from core.exceptions import (
    AppException,
    ERROR_FILE_NOT_FOUND,
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_IMAGE,
    ERROR_PROCESS_FAILED,
)
from utils.file_utils import ensure_upload_dirs
from utils.logger import get_logger, init_logger
from utils.response_utils import error_response

settings = get_settings()
ensure_upload_dirs()
init_logger()
logger = get_logger()

app = FastAPI(title='AI ID Photo Service', version='1.1.0')
app.mount(settings.static_mount_path, StaticFiles(directory=str(settings.upload_root_path), check_dir=False), name='uploads')
app.include_router(health_router)
app.include_router(detect_router)
app.include_router(generate_router)
app.include_router(print_router)
app.include_router(upload_debug_router)


@app.on_event('startup')
def startup_event() -> None:
    ensure_upload_dirs()
    logger.info('AI service startup complete')


@app.exception_handler(AppException)
async def handle_app_exception(_request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc.message, data=exc.data, error_code=exc.error_code),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(_request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else None
    message = first_error.get('msg', 'Invalid request arguments') if first_error else 'Invalid request arguments'
    return JSONResponse(
        status_code=422,
        content=error_response(message, data=None, error_code=ERROR_INVALID_ARGUMENT),
    )


@app.exception_handler(FileNotFoundError)
async def handle_file_not_found(_request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content=error_response(str(exc), data=None, error_code=ERROR_FILE_NOT_FOUND),
    )


@app.exception_handler(UnidentifiedImageError)
async def handle_invalid_image(_request: Request, exc: UnidentifiedImageError):
    return JSONResponse(
        status_code=400,
        content=error_response(str(exc) or 'Invalid image', data=None, error_code=ERROR_INVALID_IMAGE),
    )


@app.exception_handler(ValueError)
async def handle_value_error(_request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=error_response(str(exc), data=None, error_code=ERROR_INVALID_ARGUMENT),
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(_request: Request, exc: Exception):
    logger.exception('Unhandled exception: {}', exc)
    return JSONResponse(
        status_code=500,
        content=error_response('Image processing failed', data=None, error_code=ERROR_PROCESS_FAILED),
    )


@app.get('/')
def root():
    return {'service': settings.app_name, 'status': 'running'}
