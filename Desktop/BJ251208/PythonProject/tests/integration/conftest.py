import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from app.core.config import Settings, get_settings
from app.repositories.mongo import MongoRepository
from app.retrieval.milvus_repo import MilvusRepository


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest_asyncio.fixture
async def mongo(settings: Settings):
    repo = MongoRepository(settings)
    try:
        await asyncio.wait_for(repo.ping(), timeout=6)
    except Exception as exc:
        await repo.close()
        pytest.skip(f"MongoDB unavailable at {settings.mongo_uri}: {exc}")

    try:
        await repo.ensure_schema()
        yield repo
    finally:
        await repo.close()


@pytest.fixture
def milvus(settings: Settings):
    repo = MilvusRepository(settings)
    try:
        if not repo.has_collection():
            pytest.skip(f"Milvus collection missing: {settings.milvus_collection}")
        repo.ensure_loaded()
    except Exception as exc:
        repo.close()
        pytest.skip(f"Milvus unavailable at {settings.milvus_uri}: {exc}")

    try:
        yield repo
    finally:
        repo.close()


@pytest.fixture
def embedder(settings: Settings):
    if not _embedding_model_available(settings):
        pytest.skip(
            "BGE-m3 model is not cached locally; run scripts/download_models.py first"
        )
    from app.retrieval.embedder import BGEM3Embedder

    return BGEM3Embedder(settings)


def _embedding_model_available(settings: Settings) -> bool:
    if Path(settings.embedding_model).is_dir():
        return True
    hub = settings.hf_home_path / "hub"
    return hub.is_dir() and any(hub.glob("models--*bge-m3*"))
