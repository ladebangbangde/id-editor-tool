from fastapi import APIRouter

from utils.response_utils import success_response

router = APIRouter(tags=["health"])


@router.get("/ai/health")
def health_check():
    return success_response({"service": "ai-id-photo-service"}, message="AI service is running")
