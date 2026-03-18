from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=['health'])


@router.get('/health')
def health() -> dict:
    settings = get_settings()
    return {
        'success': True,
        'service': settings.service_name,
        'status': 'ok',
    }
