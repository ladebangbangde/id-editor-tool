from __future__ import annotations

from fastapi import APIRouter

from constants.colors import list_background_colors
from constants.photo_sizes import list_photo_sizes
from models.response_models import ColorListResponse, PhotoSizeListResponse
from utils.response_utils import success_response

router = APIRouter(tags=['config'])


@router.get('/ai/colors', response_model=ColorListResponse, summary='List available background colors')
def get_colors():
    return success_response(list_background_colors(), message='Fetch colors success')


@router.get('/ai/photo-sizes', response_model=PhotoSizeListResponse, summary='List available photo size templates')
def get_photo_sizes():
    return success_response(list_photo_sizes(), message='Fetch photo sizes success')
