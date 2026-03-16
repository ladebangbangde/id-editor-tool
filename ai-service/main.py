from fastapi import FastAPI

from api.detect_api import router as detect_router
from api.generate_api import router as generate_router
from api.health_api import router as health_router
from api.print_api import router as print_router
from utils.file_utils import ensure_upload_dirs
from utils.logger import get_logger, init_logger

init_logger()
logger = get_logger()

app = FastAPI(title="AI ID Photo Service", version="1.0.0")
app.include_router(health_router)
app.include_router(detect_router)
app.include_router(generate_router)
app.include_router(print_router)


@app.on_event("startup")
def startup_event() -> None:
    ensure_upload_dirs()
    logger.info("AI service startup complete")


@app.get("/")
def root():
    return {"service": "ai-id-photo-service", "status": "running"}
