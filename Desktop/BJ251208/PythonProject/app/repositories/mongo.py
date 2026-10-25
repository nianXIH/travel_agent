from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, IndexModel

from app.core.config import Settings

DOCUMENTS = "documents"
INGESTION_TASKS = "ingestion_tasks"
CHAT_SESSIONS = "chat_sessions"
CHAT_MESSAGES = "chat_messages"

COLLECTIONS = (DOCUMENTS, INGESTION_TASKS, CHAT_SESSIONS, CHAT_MESSAGES)

INDEXES: dict[str, list[IndexModel]] = {
    DOCUMENTS: [
        IndexModel([("doc_id", ASCENDING)], unique=True, name="uq_doc_id"),
        IndexModel(
            [("tenant_id", ASCENDING), ("source_path", ASCENDING)],
            unique=True,
            name="uq_tenant_source",
        ),
        IndexModel([("tenant_id", ASCENDING), ("sha256", ASCENDING)], name="ix_tenant_sha256"),
        IndexModel(
            [("tenant_id", ASCENDING), ("city", ASCENDING), ("content_type", ASCENDING)],
            name="ix_tenant_city_type",
        ),
        IndexModel(
            [("tenant_id", ASCENDING), ("status", ASCENDING), ("imported_at", DESCENDING)],
            name="ix_tenant_status_imported",
        ),
    ],
    INGESTION_TASKS: [
        IndexModel(
            [("tenant_id", ASCENDING), ("status", ASCENDING), ("created_at", DESCENDING)],
            name="ix_tenant_status_created",
        ),
        IndexModel(
            [("tenant_id", ASCENDING), ("created_at", DESCENDING)],
            name="ix_tenant_created",
        ),
    ],
    CHAT_SESSIONS: [
        IndexModel(
            [("tenant_id", ASCENDING), ("user_id", ASCENDING), ("updated_at", DESCENDING)],
            name="ix_tenant_user_updated",
        ),
        IndexModel(
            [("tenant_id", ASCENDING), ("updated_at", DESCENDING)],
            name="ix_tenant_updated",
        ),
    ],
    CHAT_MESSAGES: [
        IndexModel(
            [("tenant_id", ASCENDING), ("session_id", ASCENDING), ("created_at", ASCENDING)],
            name="ix_tenant_session_created",
        ),
    ],
}


class MongoRepository:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncMongoClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=5000,
            tz_aware=True,
        )
        self._db = self._client[settings.mongo_db]

    @property
    def client(self) -> AsyncMongoClient:
        return self._client

    @property
    def db(self):
        return self._db

    async def ping(self) -> bool:
        result = await self._client.admin.command("ping")
        return bool(result.get("ok"))

    async def ensure_schema(self) -> dict[str, list[str]]:
        existing = set(await self._db.list_collection_names())
        created: dict[str, list[str]] = {}

        for name in COLLECTIONS:
            if name not in existing:
                await self._db.create_collection(name)
            # create_indexes is idempotent: identical specs are a no-op.
            indexes = await self._db[name].create_indexes(INDEXES[name])
            created[name] = indexes

        return created

    async def close(self) -> None:
        await self._client.close()
