import secrets
from typing import Annotated

from fastapi import Header, Request

from app.chat.llm import ChatLLM
from app.chat.memory import ChatMemory
from app.chat.service import ChatService
from app.core.config import Settings
from app.core.exceptions import AppError
from app.ingestion.task_repo import IngestionRepository
from app.planner.service import PlannerService
from app.repositories.mongo import MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService
from app.workers.ingest_worker import IngestionWorker


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_mongo(request: Request) -> MongoRepository:
    return request.app.state.mongo


def get_milvus(request: Request) -> MilvusRepository:
    return request.app.state.milvus


def get_retrieval(request: Request) -> RetrievalService:
    return request.app.state.retrieval


def get_memory(request: Request) -> ChatMemory:
    return request.app.state.memory


def get_llm(request: Request) -> ChatLLM:
    return request.app.state.llm


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_planner(request: Request) -> PlannerService:
    return request.app.state.planner


def get_ingest_repo(request: Request) -> IngestionRepository:
    return request.app.state.ingest_repo


def get_ingest_worker(request: Request) -> IngestionWorker:
    return request.app.state.ingest_worker


def require_admin(
    request: Request,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    expected = request.app.state.settings.admin_token
    if not expected:
        return
    provided = (x_admin_token or "").encode("utf-8")
    if not secrets.compare_digest(provided, expected.encode("utf-8")):
        raise AppError("管理员令牌无效", status_code=401, code="UNAUTHORIZED")
