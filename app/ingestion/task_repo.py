from datetime import UTC, datetime

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING, ReturnDocument

from app.ingestion.metadata import make_doc_id
from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS, MongoRepository


def utc_now() -> datetime:
    return datetime.now(UTC)


def _scope(tenant_id: str | None, query: dict | None = None) -> dict:
    scoped = dict(query or {})
    if tenant_id is not None:
        scoped["tenant_id"] = tenant_id
    return scoped


def _require_tenant(tenant_id: str | None) -> None:
    if not tenant_id:
        raise ValueError("tenant_id is required")


class IngestionRepository:
    def __init__(self, mongo: MongoRepository) -> None:
        self._db = mongo.db

    async def create_task(
        self,
        tenant_id: str,
        task_type: str,
        total: int,
        *,
        status: str = "running",
    ) -> str:
        now = utc_now()
        result = await self._db[INGESTION_TASKS].insert_one(
            {
                "tenant_id": tenant_id,
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

    async def update_task(self, tenant_id: str, task_id: str, **fields) -> None:
        await self._db[INGESTION_TASKS].update_one(
            {"_id": ObjectId(task_id), "tenant_id": tenant_id},
            {"$set": fields},
        )

    async def get_task(self, tenant_id: str | None, task_id: str) -> dict | None:
        try:
            object_id = ObjectId(task_id)
        except (InvalidId, TypeError):
            return None
        return await self._db[INGESTION_TASKS].find_one(
            _scope(tenant_id, {"_id": object_id})
        )

    async def list_tasks(
        self,
        tenant_id: str | None,
        *,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[int, list[dict]]:
        query = _scope(tenant_id, {"status": status} if status else {})
        total = await self._db[INGESTION_TASKS].count_documents(query)
        cursor = (
            self._db[INGESTION_TASKS]
            .find(query)
            .sort("created_at", DESCENDING)
            .skip(offset)
            .limit(limit)
        )
        return total, await cursor.to_list(length=limit)

    async def get_document(self, tenant_id: str, source_path: str) -> dict | None:
        return await self._db[DOCUMENTS].find_one(
            {"tenant_id": tenant_id, "source_path": source_path}
        )

    async def list_documents(
        self,
        tenant_id: str,
        *,
        status: str | None = None,
    ) -> list[dict]:
        query = _scope(tenant_id, {"status": status} if status else {})
        cursor = self._db[DOCUMENTS].find(query)
        return await cursor.to_list(length=None)

    async def upsert_document(self, tenant_id: str, document: dict) -> dict:
        _require_tenant(tenant_id)
        now = utc_now()
        source_path = document["source_path"]
        record = {
            **document,
            "tenant_id": tenant_id,
            "doc_id": make_doc_id(tenant_id, source_path),
            "updated_at": now,
        }
        return await self._db[DOCUMENTS].find_one_and_update(
            {"tenant_id": tenant_id, "source_path": source_path},
            {
                "$set": record,
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    async def mark_document_deleted(
        self,
        tenant_id: str,
        *,
        source_path: str,
        doc_id: str,
    ) -> None:
        _require_tenant(tenant_id)
        now = utc_now()
        fields = {
            "status": "deleted",
            "chunk_count": 0,
            "error": "",
            "deleted_at": now,
            "updated_at": now,
        }
        if doc_id:
            fields["doc_id"] = doc_id
        await self._db[DOCUMENTS].update_one(
            {"tenant_id": tenant_id, "source_path": source_path},
            {"$set": fields},
        )

    async def mark_document_failed(
        self,
        tenant_id: str,
        *,
        source_path: str,
        doc_id: str,
        source_file: str,
        error: str,
    ) -> None:
        _require_tenant(tenant_id)
        now = utc_now()
        await self._db[DOCUMENTS].update_one(
            {"tenant_id": tenant_id, "source_path": source_path},
            {
                "$set": {
                    "tenant_id": tenant_id,
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
