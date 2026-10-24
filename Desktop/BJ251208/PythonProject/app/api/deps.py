from fastapi import Request

from app.chat.llm import ChatLLM
from app.chat.memory import ChatMemory
from app.chat.service import ChatService
from app.core.config import Settings
from app.repositories.mongo import MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService


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
