import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.router import api_router
from app.api.routes import health
from app.chat.llm import ChatLLM
from app.chat.memory import ChatMemory
from app.chat.service import ChatService
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.core.tenancy import TenantRegistry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.task_repo import IngestionRepository
from app.planner.service import PlannerService
from app.repositories.mongo import MongoRepository
from app.retrieval.hyde import HyDEGenerator
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService
from app.retrieval.three_way import ThreeWayRetriever
from app.retrieval.web_search import WebSearchClient
from app.workers.ingest_worker import IngestionWorker

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    mongo = MongoRepository(settings)
    milvus = MilvusRepository(settings)
    retrieval = RetrievalService(settings, milvus=milvus)
    memory = ChatMemory(mongo)
    llm = ChatLLM(settings)
    hyde = HyDEGenerator(settings, llm)
    web = WebSearchClient(settings)
    three_way = ThreeWayRetriever(settings, retrieval, hyde, web)
    chat_service = ChatService(settings, memory, three_way, llm)
    planner = PlannerService(settings, retrieval, llm)
    ingest_repo = IngestionRepository(mongo)
    ingest_pipeline = IngestionPipeline(
        settings,
        embedder=retrieval.embedder,
        mongo=mongo,
        milvus=milvus,
        repo=ingest_repo,
    )
    ingest_worker = IngestionWorker(ingest_pipeline, ingest_repo)

    app.state.mongo = mongo
    app.state.milvus = milvus
    app.state.retrieval = retrieval
    app.state.memory = memory
    app.state.llm = llm
    app.state.three_way = three_way
    app.state.chat_service = chat_service
    app.state.planner = planner
    app.state.ingest_repo = ingest_repo
    app.state.ingest_worker = ingest_worker

    try:
        await mongo.ensure_schema()
    except Exception:
        logger.exception("failed to ensure MongoDB schema; /readyz will report degraded")

    try:
        if await asyncio.to_thread(milvus.has_collection):
            await asyncio.to_thread(milvus.ensure_loaded)
    except Exception:
        logger.exception("failed to load Milvus collection; /readyz will report degraded")

    logger.info("application started: {}", settings.app_name)
    try:
        yield
    finally:
        await ingest_worker.shutdown()
        await three_way.aclose()
        await mongo.close()
        retrieval.close()
        logger.info("application stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.tenants = TenantRegistry(settings.tenants, settings.admin_token)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "app": settings.app_name,
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/healthz",
            "admin": "/admin/",
        }

    app.include_router(health.router)
    app.include_router(api_router)

    admin_dir = STATIC_DIR / "admin"
    if admin_dir.is_dir():
        app.mount("/admin", StaticFiles(directory=admin_dir, html=True), name="admin")

    return app


app = create_app()
