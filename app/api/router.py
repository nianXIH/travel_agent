from fastapi import APIRouter

from app.api.routes import chat, ingest, search, sessions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(search.router)
api_router.include_router(ingest.router)
api_router.include_router(chat.router)
api_router.include_router(sessions.router)
