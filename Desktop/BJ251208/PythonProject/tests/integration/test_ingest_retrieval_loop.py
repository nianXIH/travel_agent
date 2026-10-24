import uuid

import pytest
from bson import ObjectId

from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import ScannedFile, build_scanned_file
from app.ingestion.task_repo import IngestionRepository
from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS
from app.retrieval.service import RetrievalService

pytestmark = pytest.mark.integration

SAMPLE = """# 集成测试·杭州西湖

## 元数据
- 城市：杭州
- 内容类型：景点介绍
- 景点名称：西湖

## 简介

西湖位于杭州市区西部，是杭州最著名的淡水湖。湖岸有苏堤、白堤，
适合步行和骑行游览。傍晚可以在湖边看日落。

## 游览建议

建议清晨从断桥出发，沿白堤步行到孤山，再骑行环湖。
周末游客较多，尽量避开中午时段。

## 交通

地铁 1 号线龙翔桥站步行约十分钟可到湖滨。
"""


async def test_ingest_then_retrieve(settings, mongo, milvus, embedder, tmp_path):
    suffix = uuid.uuid4().hex[:8]
    source = tmp_path / f"integration-{suffix}.md"
    source.write_text(SAMPLE, encoding="utf-8")
    scanned: ScannedFile = build_scanned_file(source, tmp_path)
    relative_path = scanned.relative_path

    repo = IngestionRepository(mongo)
    task_id = await repo.create_task("integration", 1)
    pipeline = IngestionPipeline(
        settings,
        embedder=embedder,
        mongo=mongo,
        milvus=milvus,
        repo=repo,
    )
    retrieval = RetrievalService(settings, embedder=embedder, milvus=milvus)
    doc_id = ""

    try:
        summary = await pipeline.run_task(
            task_id,
            [scanned],
            force=True,
            progress=lambda *_args: None,
        )
        assert summary.failed == 0, summary.errors
        assert summary.succeeded == 1
        assert summary.chunks_written >= 1

        document = await repo.get_document(relative_path)
        assert document is not None
        assert document["status"] == "imported"
        assert document["city"] == "杭州"
        assert document["content_type"] == "景点介绍"
        assert document["chunk_count"] == summary.chunks_written

        task = await repo.get_task(task_id)
        assert task["status"] == "success"
        assert task["succeeded"] == 1

        doc_id = document["doc_id"]
        hits = retrieval.search(
            "西湖适合骑行还是步行游览",
            mode="hybrid",
            top_k=5,
            rerank=False,
            doc_ids=[doc_id],
        )
        assert hits, "ingested document was not retrievable"
        assert all(hit["doc_id"] == doc_id for hit in hits)
        assert any("西湖" in hit["content"] for hit in hits)
    finally:
        if doc_id:
            milvus.delete_document_chunks(doc_id)
            milvus.flush()
        await mongo.db[DOCUMENTS].delete_one({"source_path": relative_path})
        await mongo.db[INGESTION_TASKS].delete_one({"_id": ObjectId(task_id)})
