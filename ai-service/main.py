from __future__ import annotations

import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.config_api import router as config_router
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
logger = get_logger(component='main')
settings = get_settings()

app = FastAPI(
    title='AI ID Photo Service',
    version=settings.app_version,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
)
app.include_router(health_router)
app.include_router(config_router)
app.include_router(detect_router)
app.include_router(generate_router)
app.include_router(print_router)
app.include_router(upload_debug_router)

ensure_upload_dirs()
app.mount(settings.static_mount_path, StaticFiles(directory=str(settings.upload_root_path)), name='uploads')


@app.middleware('http')
async def request_logging_middleware(request: Request, call_next):
    request_id = uuid4().hex[:12]
    started_at = time.perf_counter()
    request_logger = get_logger(
        component='http',
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=(request.client.host if request.client else None),
    )
    request.state.request_id = request_id
    request.state.request_logger = request_logger
    request_logger.info('request started')

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        request_logger.bind(duration_ms=elapsed_ms).exception('request failed with unhandled exception')
        raise

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    access_logger = request_logger.bind(status_code=response.status_code, duration_ms=elapsed_ms)
    if response.status_code >= 500:
        access_logger.error('request completed')
    elif response.status_code >= 400:
        access_logger.warning('request completed')
    else:
        access_logger.info('request completed')
    response.headers['X-Request-Id'] = request_id
    return response


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    request_logger = getattr(request.state, 'request_logger', logger)
    request_logger.bind(error_code=exc.error_code, status_code=exc.status_code).warning('application error: {}', exc.message)
    payload = ErrorResponse(message=exc.message, errorCode=exc.error_code, data=exc.data)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_logger = getattr(request.state, 'request_logger', logger)
    request_logger.bind(status_code=422).warning('request validation error: {}', exc.errors())
    payload = ErrorResponse(
        message='请求参数不合法',
        errorCode=ERROR_INVALID_ARGUMENT,
        data={'details': exc.errors()},
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_logger = getattr(request.state, 'request_logger', logger)
    request_logger.exception('unhandled error: {}', exc)
    payload = ErrorResponse(message='服务内部处理失败，请稍后重试', errorCode=ERROR_PROCESS_FAILED)
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.on_event('startup')
async def startup_event() -> None:
    ensure_upload_dirs()
    logger.bind(
        app=settings.app_name,
        host=settings.app_host,
        port=settings.app_port,
        uploads=settings.upload_root,
        static=settings.static_mount_path,
        log_level=settings.log_level,
        segmentation_enabled=settings.segmentation_enabled,
    ).info('service startup complete')


@app.get('/', response_model=RootResponse, summary='Service entry')
def root() -> RootResponse:
    return RootResponse(
        service=settings.app_name,
        status='running',
        version=settings.app_version,
        docs=settings.docs_url,
        openapi=settings.openapi_url,
    )
