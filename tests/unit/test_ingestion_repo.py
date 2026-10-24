from types import SimpleNamespace

from bson import ObjectId

from app.ingestion.task_repo import IngestionRepository
from app.repositories.mongo import INGESTION_TASKS


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
            (
                document
                for document in self.documents
                if all(document.get(key) == value for key, value in query.items())
            ),
            None,
        )

    async def count_documents(self, query: dict) -> int:
        return len(
            [
                document
                for document in self.documents
                if all(document.get(key) == value for key, value in query.items())
            ]
        )

    def find(self, query: dict) -> FakeCursor:
        return FakeCursor(
            [
                document
                for document in self.documents
                if all(document.get(key) == value for key, value in query.items())
            ]
        )


class FakeMongo:
    def __init__(self) -> None:
        self.db = {INGESTION_TASKS: FakeCollection()}


async def test_create_task_with_queued_status():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)

    task_id = await repo.create_task("upload", 1, status="queued")

    document = mongo.db[INGESTION_TASKS].documents[0]
    assert str(document["_id"]) == task_id
    assert document["task_type"] == "upload"
    assert document["status"] == "queued"
    assert document["total"] == 1


async def test_get_task_accepts_valid_id_and_rejects_invalid():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)
    task_id = await repo.create_task("scan", 2)

    assert (await repo.get_task(task_id))["total"] == 2
    assert await repo.get_task("not-an-object-id") is None


async def test_list_tasks_filters_and_paginates():
    mongo = FakeMongo()
    repo = IngestionRepository(mongo)
    await repo.create_task("scan", 1, status="success")
    await repo.create_task("scan", 2, status="failed")
    await repo.create_task("scan", 3, status="failed")

    total, tasks = await repo.list_tasks(status="failed", limit=1, offset=1)

    assert total == 2
    assert len(tasks) == 1
    assert tasks[0]["total"] == 2
