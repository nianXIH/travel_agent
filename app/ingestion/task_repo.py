from datetime import UTC, datetime

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING, ReturnDocument

from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS, MongoRepository


def utc_now() -> datetime:
    return datetime.now(UTC)


class IngestionRepository:
    def __init__(self, mongo: MongoRepository) -> None:
        self._db = mongo.db

    async def create_task(
        self,
        task_type: str,
        total: int,
        *,
        status: str = "running",
    ) -> str:
        now = utc_now()
        result = await self._db[INGESTION_TASKS].insert_one(
            {
                "task_type": task_type,
                "status": status,
                "total": total,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
                "current_file": "",
                "errors": [],
                "created_at": now,
                "started_at": now if status == "running" else None,
                "finished_at": None,
            }
        )
        return str(result.inserted_id)

    async def update_task(self, task_id, **fields) -> None:
        await self._db[INGESTION_TASKS].update_one(
            {"_id": ObjectId(task_id)},
            {"$set": fields},
        )

    async def get_task(self, task_id: str) -> dict | None:
        try:
            object_id = ObjectId(task_id)
        except (InvalidId, TypeError):
            return None
        return await self._db[INGESTION_TASKS].find_one({"_id": object_id})

    async def list_tasks(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[int, list[dict]]:
        query = {"status": status} if status else {}
        total = await self._db[INGESTION_TASKS].count_documents(query)
        cursor = (
            self._db[INGESTION_TASKS]
            .find(query)
            .sort("created_at", DESCENDING)
            .skip(offset)
            .limit(limit)
        )
        return total, await cursor.to_list(length=limit)

    async def get_document(self, source_path: str) -> dict | None:
        return await self._db[DOCUMENTS].find_one({"source_path": source_path})

    async def upsert_document(self, document: dict) -> dict:
        now = utc_now()
        record = {**document, "updated_at": now}
        return await self._db[DOCUMENTS].find_one_and_update(
            {"source_path": record["source_path"]},
            {
                "$set": record,
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    async def mark_document_failed(
        self,
        *,
        source_path: str,
        doc_id: str,
        source_file: str,
        error: str,
    ) -> None:
        now = utc_now()
        await self._db[DOCUMENTS].update_one(
            {"source_path": source_path},
            {
                "$set": {
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "source_file": source_file,
                    "status": "failed",
                    "error": error,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
