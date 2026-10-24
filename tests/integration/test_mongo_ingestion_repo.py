import uuid
from datetime import UTC, datetime

import pytest
from bson import ObjectId

from app.ingestion.task_repo import IngestionRepository
from app.repositories.mongo import COLLECTIONS, DOCUMENTS, INGESTION_TASKS

pytestmark = pytest.mark.integration


async def test_ping_and_schema_are_idempotent(mongo):
    assert await mongo.ping()

    await mongo.ensure_schema()
    names = set(await mongo.db.list_collection_names())
    assert set(COLLECTIONS) <= names

    document_indexes = await mongo.db[DOCUMENTS].index_information()
    assert "uq_doc_id" in document_indexes
    assert "uq_source_path" in document_indexes
    assert document_indexes["uq_doc_id"].get("unique") is True

    task_indexes = await mongo.db[INGESTION_TASKS].index_information()
    assert "ix_status_created" in task_indexes


async def test_task_lifecycle_and_pagination(mongo):
    repo = IngestionRepository(mongo)
    task_type = f"integration-{uuid.uuid4().hex[:8]}"
    task_id = await repo.create_task(task_type, 2)

    try:
        created = await repo.get_task(task_id)
        assert created["status"] == "running"
        assert created["total"] == 2
        assert created["started_at"] is not None

        await repo.update_task(
            task_id,
            status="partial",
            succeeded=1,
            failed=1,
            current_file="",
            finished_at=datetime.now(UTC),
        )
        updated = await repo.get_task(task_id)
        assert updated["status"] == "partial"
        assert (updated["succeeded"], updated["failed"]) == (1, 1)

        total, tasks = await repo.list_tasks(status="partial", limit=20)
        assert total >= 1
        assert any(task["_id"] == ObjectId(task_id) for task in tasks)

        assert await repo.get_task("not-an-object-id") is None
    finally:
        await mongo.db[INGESTION_TASKS].delete_one({"_id": ObjectId(task_id)})


async def test_document_upsert_and_failure_marking(mongo):
    repo = IngestionRepository(mongo)
    suffix = uuid.uuid4().hex[:8]
    source_path = f"__integration__/sample-{suffix}.md"
    doc_id = f"it_{suffix}"

    try:
        await repo.upsert_document(
            {
                "doc_id": doc_id,
                "source_path": source_path,
                "source_file": f"sample-{suffix}.md",
                "title": "first",
                "sha256": "a" * 64,
                "content_type": "景点介绍",
                "city": "杭州",
                "status": "imported",
                "size": 128,
                "mtime": 1.0,
                "chunk_count": 1,
                "imported_at": datetime.now(UTC),
                "error": "",
            }
        )
        first = await repo.get_document(source_path)
        assert first["title"] == "first"
        assert first["status"] == "imported"

        await repo.upsert_document(
            {
                "doc_id": doc_id,
                "source_path": source_path,
                "source_file": f"sample-{suffix}.md",
                "title": "second",
                "sha256": "b" * 64,
                "content_type": "景点介绍",
                "city": "杭州",
                "status": "imported",
                "size": 129,
                "mtime": 2.0,
                "chunk_count": 1,
                "imported_at": datetime.now(UTC),
                "error": "",
            }
        )
        updated = await repo.get_document(source_path)
        assert updated["title"] == "second"
        assert updated["sha256"] == "b" * 64

        await repo.mark_document_failed(
            source_path=source_path,
            doc_id=doc_id,
            source_file=f"sample-{suffix}.md",
            error="ValueError: boom",
        )
        failed = await repo.get_document(source_path)
        assert failed["status"] == "failed"
        assert failed["error"] == "ValueError: boom"
    finally:
        await mongo.db[DOCUMENTS].delete_one({"source_path": source_path})
