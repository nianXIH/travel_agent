from fastapi import APIRouter

from app.api.routes import admin, chat, ingest, planning, search, sessions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(admin.router)
api_router.include_router(search.router)
api_router.include_router(ingest.router)
api_router.include_router(chat.router)
api_router.include_router(planning.router)
api_router.include_router(sessions.router)
