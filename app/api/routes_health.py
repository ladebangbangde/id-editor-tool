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
        'uploadRoot': str(settings.upload_root_path),
        'staticMountPath': settings.normalized_static_mount_path,
        'directories': {name: str(path) for name, path in settings.upload_dirs.items() if name != 'base'},
    }
