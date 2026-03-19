from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.detect_api import router as detect_router
from api.generate_api import router as generate_router
from api.health_api import router as health_router
from api.print_api import router as print_router
from api.upload_debug_api import router as upload_debug_router
from core.exceptions import AppException, ERROR_INVALID_ARGUMENT, ERROR_PROCESS_FAILED
from models.response_models import ErrorResponse, RootResponse
from utils.config import get_settings
from utils.file_utils import ensure_upload_dirs
from utils.logger import get_logger, init_logger

init_logger()
logger = get_logger()
settings = get_settings()

app = FastAPI(
    title='AI ID Photo Service',
    version=settings.app_version,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
)
app.include_router(health_router)
app.include_router(detect_router)
app.include_router(generate_router)
app.include_router(print_router)
app.include_router(upload_debug_router)

ensure_upload_dirs()
app.mount(settings.static_mount_path, StaticFiles(directory=str(settings.upload_root_path)), name='uploads')


@app.exception_handler(AppException)
async def app_exception_handler(_request: Request, exc: AppException) -> JSONResponse:
    logger.warning('application error: code={}, message={}', exc.error_code, exc.message)
    payload = ErrorResponse(message=exc.message, errorCode=exc.error_code, data=exc.data)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning('request validation error: {}', exc.errors())
    payload = ErrorResponse(
        message='请求参数不合法',
        errorCode=ERROR_INVALID_ARGUMENT,
        data={'details': exc.errors()},
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception('unhandled error on {} {}', request.method, request.url.path)
    payload = ErrorResponse(message='服务内部处理失败，请稍后重试', errorCode=ERROR_PROCESS_FAILED)
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.on_event('startup')
async def startup_event() -> None:
    ensure_upload_dirs()
    logger.info(
        'service startup complete: app={}, host={}, port={}, uploads={}, static={}',
        settings.app_name,
        settings.app_host,
        settings.app_port,
        settings.upload_root,
        settings.static_mount_path,
    )


@app.get('/', response_model=RootResponse, summary='Service entry')
def root() -> RootResponse:
    return RootResponse(
        service=settings.app_name,
        status='running',
        version=settings.app_version,
        docs=settings.docs_url,
        openapi=settings.openapi_url,
    )
