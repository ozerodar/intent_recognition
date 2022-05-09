from fastapi import APIRouter
from intent_detection.api import intents

api_router = APIRouter()
api_router.include_router(intents.router, prefix="/api", tags=["Intent detection"])
