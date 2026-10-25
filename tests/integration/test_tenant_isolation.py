import shutil
import uuid

import pytest
from bson import ObjectId

from app.core.tenancy import DEFAULT_TENANT_ID, TENANTS_DIR
from app.ingestion.metadata import make_doc_id
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import build_scanned_file
from app.ingestion.task_repo import IngestionRepository
from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS, MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService

pytestmark = pytest.mark.integration

SAMPLE = """# 租户隔离测试

## 元数据
- 城市：杭州
- 内容类型：景点介绍
- 景点名称：西湖

## 简介

西湖适合清晨骑行，湖岸有苏堤和白堤。
"""


def _write(root, name: str, text: str):
    path = root / "uploads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_tenants_cannot_retrieve_each_others_chunks(settings, mongo, milvus, embedder):
    suffix = uuid.uuid4().hex[:8]
    tenant_b = f"it{suffix}"
    root_a = settings.tenant_data_path(DEFAULT_TENANT_ID)
    root_b = settings.tenant_data_path(tenant_b)
    file_a = _write(root_a, f"tenant-a-{suffix}.md", SAMPLE)
    file_b = _write(root_b, f"tenant-b-{suffix}.md", SAMPLE)
    scanned_a = build_scanned_file(file_a, root_a)
    scanned_b = build_scanned_file(file_b, root_b)
    doc_a = make_doc_id(DEFAULT_TENANT_ID, scanned_a.relative_path)
    doc_b = make_doc_id(tenant_b, scanned_b.relative_path)
    assert doc_a != doc_b

    repo = IngestionRepository(mongo)
    pipeline = IngestionPipeline(settings, embedder=embedder, mongo=mongo, milvus=milvus, repo=repo)
    retrieval = RetrievalService(settings, embedder=embedder, milvus=milvus)
    task_a = await repo.create_task(DEFAULT_TENANT_ID, "integration", 1)
    task_b = await repo.create_task(tenant_b, "integration", 1)

    def _search(tenant_id: str, doc_ids: list[str]) -> list[dict]:
        return retrieval.search(
            "西湖适合骑行吗",
            tenant_id=tenant_id,
            mode="hybrid",
            top_k=5,
            rerank=False,
            doc_ids=doc_ids,
        )

    try:
        summary_a = await pipeline.run_task(
            task_a,
            [scanned_a],
            tenant_id=DEFAULT_TENANT_ID,
            force=True,
            progress=lambda *_args: None,
        )
        summary_b = await pipeline.run_task(
            task_b,
            [scanned_b],
            tenant_id=tenant_b,
            force=True,
            progress=lambda *_args: None,
        )
        assert summary_a.failed == 0, summary_a.errors
        assert summary_b.failed == 0, summary_b.errors

        document_a = await repo.get_document(DEFAULT_TENANT_ID, scanned_a.relative_path)
        document_b = await repo.get_document(tenant_b, scanned_b.relative_path)
        assert document_a["tenant_id"] == DEFAULT_TENANT_ID
        assert document_b["tenant_id"] == tenant_b
        assert document_a["doc_id"] == doc_a
        assert document_b["doc_id"] == doc_b
        assert doc_a != doc_b

        hits_a = _search(DEFAULT_TENANT_ID, [doc_a])
        assert hits_a
        assert all(hit["doc_id"] == doc_a for hit in hits_a)

        hits_b = _search(tenant_b, [doc_b])
        assert hits_b
        assert all(hit["doc_id"] == doc_b for hit in hits_b)

        leaks_a = _search(DEFAULT_TENANT_ID, [doc_b])
        assert leaks_a == []

        leaks_b = _search(tenant_b, [doc_a])
        assert leaks_b == []
    finally:
        file_a.unlink(missing_ok=True)
        file_b.unlink(missing_ok=True)
        assert root_b.resolve() != settings.data_path.resolve()
        assert root_b.resolve().is_relative_to((settings.data_path / TENANTS_DIR).resolve())
        shutil.rmtree(root_b, ignore_errors=True)

        cleanup_mongo = MongoRepository(settings)
        cleanup_milvus = MilvusRepository(settings)
        try:
            for tenant_id, doc_id in (
                (DEFAULT_TENANT_ID, doc_a),
                (tenant_b, doc_b),
            ):
                cleanup_milvus.delete_document_chunks(tenant_id, doc_id)
            cleanup_milvus.flush()
            await cleanup_mongo.db[DOCUMENTS].delete_many(
                {
                    "$or": [
                        {
                            "tenant_id": DEFAULT_TENANT_ID,
                            "source_path": scanned_a.relative_path,
                        },
                        {
                            "tenant_id": tenant_b,
                            "source_path": scanned_b.relative_path,
                        },
                    ]
                }
            )
            for tenant_id, task_id in (
                (DEFAULT_TENANT_ID, task_a),
                (tenant_b, task_b),
            ):
                await cleanup_mongo.db[INGESTION_TASKS].delete_one(
                    {"_id": ObjectId(task_id), "tenant_id": tenant_id}
                )
        finally:
            await cleanup_mongo.close()
            cleanup_milvus.close()
