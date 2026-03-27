from __future__ import annotations

from io import BytesIO
import base64

import pytest
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.services.baidu_human_segmentation_service import BaiduHumanSegmentationService


def _png_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_segment_human_success(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv('BAIDU_SEGMENTATION_ENABLED', 'true')
    monkeypatch.setenv('BAIDU_API_KEY', 'k')
    monkeypatch.setenv('BAIDU_SECRET_KEY', 's')

    fg = Image.new('RGBA', (12, 16), (255, 0, 0, 128))
    label = Image.new('L', (12, 16), 255)
    score = Image.new('L', (12, 16), 128)

    calls = {'count': 0}

    class DummySession:
        def post(self, url, **kwargs):
            calls['count'] += 1
            if 'oauth' in url:
                return DummyResponse({'access_token': 'token-1', 'expires_in': 3600})
            return DummyResponse(
                {
                    'foreground': _png_base64(fg),
                    'labelmap': _png_base64(label),
                    'scoremap': _png_base64(score),
                }
            )

    monkeypatch.setattr(BaiduHumanSegmentationService, '_http_session', DummySession())
    BaiduHumanSegmentationService._cached_token = None
    BaiduHumanSegmentationService._token_expire_at = 0.0

    svc = BaiduHumanSegmentationService()
    result = svc.segment_human(Image.new('RGB', (12, 16), 'white'))

    assert result.foreground.mode == 'RGBA'
    assert result.foreground.size == (12, 16)
    assert result.labelmap is not None
    assert result.scoremap is not None
    assert calls['count'] == 2


def test_access_token_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv('BAIDU_SEGMENTATION_ENABLED', 'true')
    monkeypatch.setenv('BAIDU_API_KEY', 'k')
    monkeypatch.setenv('BAIDU_SECRET_KEY', 's')

    calls = {'count': 0}

    class DummySession:
        def post(self, url, **kwargs):
            calls['count'] += 1
            return DummyResponse({'access_token': 'token-1', 'expires_in': 3600})

    monkeypatch.setattr(BaiduHumanSegmentationService, '_http_session', DummySession())
    BaiduHumanSegmentationService._cached_token = None
    BaiduHumanSegmentationService._token_expire_at = 0.0

    svc = BaiduHumanSegmentationService()
    token_1 = svc.get_access_token()
    token_2 = svc.get_access_token()

    assert token_1 == token_2 == 'token-1'
    assert calls['count'] == 1


def test_segment_human_requires_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv('BAIDU_SEGMENTATION_ENABLED', 'true')
    monkeypatch.delenv('BAIDU_API_KEY', raising=False)
    monkeypatch.delenv('BAIDU_SECRET_KEY', raising=False)
    BaiduHumanSegmentationService._cached_token = None
    BaiduHumanSegmentationService._token_expire_at = 0.0

    svc = BaiduHumanSegmentationService()
    with pytest.raises(AppError) as exc_info:
        svc.get_access_token()

    assert exc_info.value.code == 'BAIDU_SEGMENTATION_CONFIG_MISSING'
