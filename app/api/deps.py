from fastapi import Request

from app.chat.llm import ChatLLM
from app.chat.memory import ChatMemory
from app.chat.service import ChatService
from app.core.config import Settings
from app.ingestion.task_repo import IngestionRepository
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


def get_ingest_repo(request: Request) -> IngestionRepository:
    return request.app.state.ingest_repo


def get_ingest_worker(request: Request) -> IngestionWorker:
    return request.app.state.ingest_worker
