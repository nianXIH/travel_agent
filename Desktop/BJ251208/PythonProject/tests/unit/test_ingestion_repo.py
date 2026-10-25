from types import SimpleNamespace

import pytest
from bson import ObjectId

from app.ingestion.metadata import make_doc_id
from app.ingestion.task_repo import IngestionRepository
from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS


def _matches(document: dict, query: dict) -> bool:
    return all(document.get(key) == value for key, value in query.items())


class FakeCursor:
    def __init__(self, documents: list[dict]) -> None:
        self._documents = list(documents)

    def sort(self, key: str, direction: int) -> "FakeCursor":
        reverse = direction < 0
        self._documents.sort(key=lambda doc: doc.get(key), reverse=reverse)
        return self

    def skip(self, offset: int) -> "FakeCursor":
        self._documents = self._documents[offset:]
        return self

    def limit(self, limit: int) -> "FakeCursor":
        self._documents = self._documents[:limit]
        return self

    async def to_list(self, length: int | None = None) -> list[dict]:
        return list(self._documents)


class FakeCollection:
    def __init__(self) -> None:
        self.documents: list[dict] = []

    async def insert_one(self, document: dict) -> SimpleNamespace:
        stored = {**document, "_id": ObjectId()}
        self.documents.append(stored)
        return SimpleNamespace(inserted_id=stored["_id"])

    async def find_one(self, query: dict) -> dict | None:
        return next(
            (document for document in self.documents if _matches(document, query)),
            None,
        )

    async def count_documents(self, query: dict) -> int:
        return len([document for document in self.documents if _matches(document, query)])

    def find(self, query: dict) -> FakeCursor:
        return FakeCursor(
            [document for document in self.documents if _matches(document, query)]
        )

    def _upsert(self, query: dict, update: dict) -> dict:
        document = {"_id": ObjectId(), **query}
        document.update(update.get("$setOnInsert", {}))
        document.update(update.get("$set", {}))
        self.documents.append(document)
        return document

    async def find_one_and_update(
        self,
        query: dict,
        update: dict,
        *,
        upsert: bool = False,
        return_document=None,
    ) -> dict | None:
        document = await self.find_one(query)
        if document is None:
            if not upsert:
                return None
            return self._upsert(query, update)
        document.update(update.get("$set", {}))
        return document

    async def update_one(
        self,
        query: dict,
        update: dict,
        *,
        upsert: bool = False,
    ) -> SimpleNamespace:
        document = await self.find_one(query)
        if document is None:
            if not upsert:
                return SimpleNamespace(matched_count=0, upserted_id=None)
            self._upsert(query, update)
            return SimpleNamespace(matched_count=0, upserted_id=ObjectId())
        document.update(update.get("$set", {}))
        return SimpleNamespace(matched_count=1, upserted_id=None)


class FakeMongo:
    def __init__(self) -> None:
        self.db = {
            INGESTION_TASKS: FakeCollection(),
            DOCUMENTS: FakeCollection(),
        }


async def test_create_task_with_queued_status():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)

    task_id = await repo.create_task("default", "upload", 1, status="queued")

    document = mongo.db[INGESTION_TASKS].documents[0]
    assert document["tenant_id"] == "default"
    assert str(document["_id"]) == task_id
    assert document["task_type"] == "upload"
    assert document["status"] == "queued"
    assert document["total"] == 1


async def test_get_task_accepts_valid_id_and_rejects_invalid():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)
    task_id = await repo.create_task("default", "scan", 2)

    assert (await repo.get_task("default", task_id))["total"] == 2
    assert await repo.get_task("sanya", task_id) is None
    assert await repo.get_task(None, task_id) is not None
    assert await repo.get_task("default", "not-an-object-id") is None


async def test_list_tasks_filters_and_paginates():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)
    await repo.create_task("default", "scan", 1, status="success")
    await repo.create_task("default", "scan", 2, status="failed")
    await repo.create_task("sanya", "scan", 3, status="failed")

    total, tasks = await repo.list_tasks("default", status="failed")

    assert total == 1
    assert tasks[0]["total"] == 2
    assert (await repo.list_tasks(None, status="failed"))[0] == 2


async def test_update_task_is_tenant_scoped():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)
    task_id = await repo.create_task("default", "scan", 1)

    await repo.update_task("sanya", task_id, status="failed")

    assert (await repo.get_task("default", task_id))["status"] == "running"


async def test_document_writes_are_tenant_scoped():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)
    source_path = "景点攻略/杭州.md"

    stored = await repo.upsert_document(
        "default",
        {"source_path": source_path, "title": "杭州", "doc_id": "stale-doc-id"},
    )

    assert stored["tenant_id"] == "default"
    assert stored["doc_id"] == make_doc_id("default", source_path)
    assert (await repo.get_document("default", source_path))["title"] == "杭州"
    assert await repo.get_document("sanya", source_path) is None

    await repo.upsert_document("sanya", {"source_path": source_path, "title": "杭州"})

    assert len(mongo.db[DOCUMENTS].documents) == 2
    sanya = await repo.get_document("sanya", source_path)
    assert sanya["doc_id"] == make_doc_id("sanya", source_path)
    assert [document["tenant_id"] for document in await repo.list_documents("sanya")] == [
        "sanya"
    ]


async def test_document_writes_require_tenant():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)
    source_path = "景点攻略/杭州.md"

    with pytest.raises(ValueError, match="tenant_id is required"):
        await repo.upsert_document(None, {"source_path": source_path, "title": "杭州"})

    with pytest.raises(ValueError, match="tenant_id is required"):
        await repo.mark_document_deleted(
            None,
            source_path=source_path,
            doc_id="doc-1",
        )

    with pytest.raises(ValueError, match="tenant_id is required"):
        await repo.mark_document_failed(
            None,
            source_path=source_path,
            doc_id="doc-1",
            source_file=source_path,
            error="boom",
        )

    assert mongo.db[DOCUMENTS].documents == []
