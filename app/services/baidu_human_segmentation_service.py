from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from io import BytesIO

import requests
import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BaiduSegmentationResult:
    foreground: Image.Image
    labelmap: Image.Image | None = None
    scoremap: Image.Image | None = None
    alpha_seed: Image.Image | None = None
    trimap_seed: Image.Image | None = None


class BaiduHumanSegmentationService:
    _token_lock = threading.RLock()
    _cached_token: str | None = None
    _token_expire_at: float = 0.0
    _http_session: requests.Session | None = None

    def __init__(self) -> None:
        self.settings = get_settings()

    @classmethod
    def _get_http_session(cls) -> requests.Session:
        if cls._http_session is None:
            cls._http_session = requests.Session()
        return cls._http_session

    def _decode_base64_image(self, value: str, mode: str | None = None) -> Image.Image:
        raw_value = value.split(',', 1)[1] if value.startswith('data:') and ',' in value else value
        image_bytes = base64.b64decode(raw_value)
        image = Image.open(BytesIO(image_bytes))
        if mode:
            return image.convert(mode)
        return image

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @staticmethod
    def _resize_like(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        if image.size == target_size:
            return image
        return image.resize(target_size, Image.Resampling.BILINEAR)

    def build_alpha_seed_from_maps(
        self,
        labelmap: Image.Image | None,
        scoremap: Image.Image | None,
        fallback_alpha: Image.Image | None,
        target_size: tuple[int, int],
    ) -> Image.Image:
        label_np = None
        score_np = None
        if labelmap is not None:
            label_np = np.asarray(self._resize_like(labelmap.convert('L'), target_size), dtype=np.uint8)
        if scoremap is not None:
            score_np = np.asarray(self._resize_like(scoremap.convert('L'), target_size), dtype=np.uint8)

        if score_np is not None and np.max(score_np) <= 1:
            score_np = score_np * 255
        if score_np is not None and np.max(score_np) <= 100:
            score_np = np.clip(score_np.astype(np.float32) * 2.55, 0, 255).astype(np.uint8)

        if label_np is None and score_np is None:
            if fallback_alpha is not None:
                return self._resize_like(fallback_alpha.convert('L'), target_size)
            return Image.new('L', target_size, color=255)

        if label_np is None:
            alpha = score_np.astype(np.float32)
        elif score_np is None:
            alpha = (label_np > 0).astype(np.float32) * 255.0
        else:
            hard = (label_np > 0).astype(np.float32) * 255.0
            soft = score_np.astype(np.float32)
            alpha = 0.70 * soft + 0.30 * hard
            alpha[label_np == 0] = np.minimum(alpha[label_np == 0], 90.0)
            alpha[label_np > 0] = np.maximum(alpha[label_np > 0], 130.0)

        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(alpha, mode='L')

    def build_trimap_seed_from_alpha(self, alpha_seed: Image.Image) -> Image.Image:
        alpha = np.asarray(alpha_seed.convert('L'), dtype=np.uint8)
        trimap = np.full_like(alpha, 128, dtype=np.uint8)
        trimap[alpha >= 235] = 255
        trimap[alpha <= 20] = 0
        return Image.fromarray(trimap, mode='L')

    def get_access_token(self) -> str:
        now = time.time()
        cached_token = self.__class__._cached_token
        if cached_token and now < self.__class__._token_expire_at:
            logger.info('Baidu access token cache hit')
            return cached_token

        if not self.settings.baidu_api_key or not self.settings.baidu_secret_key:
            raise AppError(
                code='BAIDU_SEGMENTATION_CONFIG_MISSING',
                message='Baidu segmentation is enabled but BAIDU_API_KEY/BAIDU_SECRET_KEY is missing',
                status_code=500,
            )

        with self.__class__._token_lock:
            now = time.time()
            cached_token = self.__class__._cached_token
            if cached_token and now < self.__class__._token_expire_at:
                logger.info('Baidu access token cache hit')
                return cached_token

            try:
                response = self._get_http_session().post(
                    self.settings.baidu_oauth_url,
                    params={
                        'grant_type': 'client_credentials',
                        'client_id': self.settings.baidu_api_key,
                        'client_secret': self.settings.baidu_secret_key,
                    },
                    timeout=self.settings.baidu_http_timeout_sec,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                logger.exception('Baidu OAuth request failed')
                raise AppError(
                    code='BAIDU_ACCESS_TOKEN_FAILED',
                    message=f'Failed to fetch Baidu access token: {exc}',
                    status_code=502,
                ) from exc

            access_token = payload.get('access_token')
            if not access_token:
                error_code = payload.get('error', payload.get('error_code', 'unknown'))
                error_message = payload.get('error_description', payload.get('error_msg', 'unknown error'))
                logger.error('Baidu OAuth response missing access_token error=%s message=%s', error_code, error_message)
                raise AppError(
                    code='BAIDU_ACCESS_TOKEN_FAILED',
                    message=f'Failed to fetch Baidu access token: {error_code} {error_message}',
                    status_code=502,
                    details=payload,
                )

            expires_in = int(payload.get('expires_in', 0))
            refresh_buffer = max(120, min(600, expires_in // 10)) if expires_in > 0 else 120
            self.__class__._cached_token = str(access_token)
            self.__class__._token_expire_at = now + max(expires_in - refresh_buffer, 60)
            logger.info(
                'Baidu access token refreshed expires_in=%s refresh_buffer=%s',
                expires_in,
                refresh_buffer,
            )
            return str(access_token)

    @staticmethod
    def _extract_result(payload: dict) -> dict:
        if isinstance(payload.get('result'), dict):
            return payload['result']
        return payload

    def segment_human(self, image: Image.Image) -> BaiduSegmentationResult:
        token = self.get_access_token()
        body = {
            'image': self._encode_image_to_base64(image),
            'type': 'foreground',
        }
        try:
            response = self._get_http_session().post(
                f"{self.settings.baidu_segmentation_url}?access_token={token}",
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                data=body,
                timeout=self.settings.baidu_http_timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.exception('Baidu segmentation request failed')
            raise AppError(
                code='BAIDU_SEGMENTATION_FAILED',
                message=f'Baidu segmentation request failed: {exc}',
                status_code=502,
            ) from exc

        if payload.get('error_code'):
            logger.error(
                'Baidu segmentation returned error error_code=%s error_msg=%s',
                payload.get('error_code'),
                payload.get('error_msg'),
            )
            raise AppError(
                code='BAIDU_SEGMENTATION_FAILED',
                message=f"Baidu segmentation failed: {payload.get('error_msg', 'unknown error')}",
                status_code=502,
                details=payload,
            )

        result_payload = self._extract_result(payload)
        foreground_raw = result_payload.get('foreground')
        if not foreground_raw:
            logger.error('Baidu segmentation response missing foreground payload_keys=%s', sorted(result_payload.keys()))
            raise AppError(
                code='BAIDU_SEGMENTATION_NO_FOREGROUND',
                message='Baidu segmentation response missing foreground image',
                status_code=502,
                details=payload,
            )

        foreground = self._decode_base64_image(str(foreground_raw), mode='RGBA')
        labelmap_raw = result_payload.get('labelmap')
        scoremap_raw = result_payload.get('scoremap')

        labelmap = self._decode_base64_image(str(labelmap_raw), mode='L') if labelmap_raw else None
        scoremap = self._decode_base64_image(str(scoremap_raw), mode='L') if scoremap_raw else None
        alpha_seed = self.build_alpha_seed_from_maps(
            labelmap=labelmap,
            scoremap=scoremap,
            fallback_alpha=foreground.getchannel('A'),
            target_size=foreground.size,
        )
        trimap_seed = self.build_trimap_seed_from_alpha(alpha_seed)

        logger.info(
            'Baidu segmentation succeeded has_labelmap=%s has_scoremap=%s',
            labelmap is not None,
            scoremap is not None,
        )

        return BaiduSegmentationResult(
            foreground=foreground,
            labelmap=labelmap,
            scoremap=scoremap,
            alpha_seed=alpha_seed,
            trimap_seed=trimap_seed,
        )
