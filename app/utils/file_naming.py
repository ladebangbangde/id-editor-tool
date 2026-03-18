from datetime import datetime
from uuid import uuid4


def build_task_id(prefix: str = 'task') -> str:
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return f'{prefix}_{timestamp}_{uuid4().hex[:8]}'


def date_slug() -> str:
    return datetime.utcnow().strftime('%Y%m%d')
