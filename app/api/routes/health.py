import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_llm, get_milvus, get_mongo, get_settings_dep
from app.chat.llm import ChatLLM
from app.core.config import Settings
from app.repositories.mongo import MongoRepository
from app.retrieval.milvus_repo import MilvusRepository

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz(settings: Annotated[Settings, Depends(get_settings_dep)]) -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
    }


@router.get("/readyz")
async def readyz(
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
    milvus: Annotated[MilvusRepository, Depends(get_milvus)],
    llm: Annotated[ChatLLM, Depends(get_llm)],
) -> JSONResponse:
    checks: dict = {"mongo": False, "milvus": False, "llm_configured": llm.configured}
    chunk_count = 0

    try:
        checks["mongo"] = await mongo.ping()
    except Exception as exc:
        checks["mongo_error"] = f"{type(exc).__name__}: {exc}"

    try:
        if await asyncio.to_thread(milvus.has_collection):
            chunk_count = await asyncio.to_thread(milvus.count_chunks)
            checks["milvus"] = True
            checks["chunk_count"] = chunk_count
        else:
            checks["milvus_error"] = "collection not found"
    except Exception as exc:
        checks["milvus_error"] = f"{type(exc).__name__}: {exc}"

    ready = checks["mongo"] and checks["milvus"] and chunk_count > 0
    checks["status"] = "ready" if ready else "degraded"
    return JSONResponse(status_code=200 if ready else 503, content=checks)
