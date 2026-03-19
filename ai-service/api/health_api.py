from __future__ import annotations

from fastapi import APIRouter

from models.response_models import HealthResponse
from utils.config import get_settings

router = APIRouter(tags=['health'])


@router.get('/health', response_model=HealthResponse, summary='Health check')
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(service=settings.app_name, status='ok')


@router.get('/ai/health', response_model=HealthResponse, include_in_schema=False)
def legacy_health_check() -> HealthResponse:
    return health_check()
