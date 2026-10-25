from app.ingestion.metadata import make_doc_id
from app.repositories.mongo import (
    CHAT_MESSAGES,
    CHAT_SESSIONS,
    DOCUMENTS,
    INGESTION_TASKS,
)
from scripts.migrate_multitenant import (
    LEGACY_INDEXES,
    MISSING_TENANT,
    TENANT_COLLECTIONS,
    backfill_tenant_ids,
    drop_legacy_indexes,
    recompute_doc_ids,
    should_recreate,
)


def _matches(document: dict, query: dict) -> bool:
    for key, expected in query.items():
        if key == "$or":
            if not any(_matches(document, clause) for clause in expected):
                return False
            continue
        actual = document.get(key)
        if actual != expected:
            return False
    return True


class FakeCursor:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = [dict(document) for document in documents]
        self._position = 0

    def __aiter__(self) -> "FakeCursor":
        return self

    async def __anext__(self) -> dict:
        if self._position >= len(self.documents):
            raise StopAsyncIteration
        document = self.documents[self._position]
        self._position += 1
        return dict(document)


class FakeCollection:
    def __init__(
        self,
        documents: list[dict] | None = None,
        indexes: list[str] | None = None,
    ) -> None:
        self.documents = [dict(document) for document in documents or []]
        self.indexes = list(indexes or [])
        self.update_many_calls: list[tuple[dict, dict]] = []
        self.update_one_calls: list[tuple[dict, dict]] = []
        self.dropped_indexes: list[str] = []
        self.count_queries: list[dict] = []

    async def count_documents(self, query: dict) -> int:
        self.count_queries.append(query)
        return len([document for document in self.documents if _matches(document, query)])

    async def update_many(self, query: dict, update: dict) -> None:
        self.update_many_calls.append((query, update))
        for document in self.documents:
            if _matches(document, query):
                document.update(update.get("$set", {}))

    async def update_one(self, query: dict, update: dict) -> None:
        self.update_one_calls.append((query, update))
        for document in self.documents:
            if _matches(document, query):
                document.update(update.get("$set", {}))
                return

    def find(self, query: dict, projection: dict | None = None) -> FakeCursor:
        return FakeCursor([document for document in self.documents if _matches(document, query)])

    async def index_information(self) -> dict:
        return {name: {"v": 2} for name in self.indexes}

    async def drop_index(self, name: str) -> None:
        self.dropped_indexes.append(name)
        self.indexes = [index for index in self.indexes if index != name]


class FakeDb:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {
            name: FakeCollection() for name in TENANT_COLLECTIONS
        }

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongo:
    def __init__(self) -> None:
        self.db = FakeDb()

    def writes(self) -> list[str]:
        names: list[str] = []
        for name, collection in self.db.collections.items():
            if (
                collection.update_many_calls
                or collection.update_one_calls
                or collection.dropped_indexes
            ):
                names.append(name)
        return names


def _document(**overrides) -> dict:
    base = {
        "_id": "doc-1",
        "tenant_id": "default",
        "source_path": "景点攻略/杭州.md",
        "doc_id": make_doc_id("default", "景点攻略/杭州.md"),
    }
    base.update(overrides)
    return base


async def test_dry_run_only_reads():
    mongo = FakeMongo()
    mongo.db[DOCUMENTS].documents = [
        _document(),
        {"_id": "doc-2", "source_path": "旧.md", "doc_id": "doc_legacy"},
    ]
    mongo.db[DOCUMENTS].indexes = ["uq_doc_id", "uq_source_path", "ix_sha256"]

    backfill = await backfill_tenant_ids(mongo, apply=False)
    changed, skipped = await recompute_doc_ids(mongo, apply=False)
    legacy = await drop_legacy_indexes(mongo, apply=False)

    assert mongo.writes() == []
    assert backfill[DOCUMENTS] == 1
    assert backfill[INGESTION_TASKS] == 0
    assert backfill[CHAT_SESSIONS] == 0
    assert backfill[CHAT_MESSAGES] == 0
    assert changed == 1
    assert skipped == []
    assert legacy[DOCUMENTS] == ["uq_source_path", "ix_sha256"]
    assert all(
        collection.count_queries == [MISSING_TENANT] for collection in mongo.db.collections.values()
    )


async def test_apply_backfills_missing_null_and_empty_tenants():
    mongo = FakeMongo()
    mongo.db[DOCUMENTS].documents = [
        {"_id": "missing", "source_path": "a.md"},
        {"_id": "null", "tenant_id": None, "source_path": "b.md"},
        {"_id": "empty", "tenant_id": "", "source_path": "c.md"},
        {"_id": "owned", "tenant_id": "sanya", "source_path": "d.md"},
    ]
    mongo.db[CHAT_SESSIONS].documents = [{"_id": "session", "tenant_id": None}]

    counts = await backfill_tenant_ids(mongo, apply=True)

    assert counts[DOCUMENTS] == 3
    assert counts[INGESTION_TASKS] == 0
    assert counts[CHAT_SESSIONS] == 1
    assert mongo.db[DOCUMENTS].update_many_calls == [
        (MISSING_TENANT, {"$set": {"tenant_id": "default"}})
    ]
    assert mongo.db[CHAT_SESSIONS].update_many_calls == [
        (MISSING_TENANT, {"$set": {"tenant_id": "default"}})
    ]
    assert mongo.db[INGESTION_TASKS].update_many_calls == []
    assert mongo.db[CHAT_MESSAGES].update_many_calls == []
    tenants = {document["_id"]: document["tenant_id"] for document in mongo.db[DOCUMENTS].documents}
    assert tenants == {
        "missing": "default",
        "null": "default",
        "empty": "default",
        "owned": "sanya",
    }


async def test_recompute_doc_ids_rewrites_and_skips():
    mongo = FakeMongo()
    mongo.db[DOCUMENTS].documents = [
        {"_id": "legacy", "source_path": "景点攻略/杭州.md", "doc_id": "doc_legacy"},
        {"_id": "blank-tenant", "tenant_id": "", "source_path": "三亚.md", "doc_id": ""},
        _document(_id="current"),
        {"_id": "no-path", "tenant_id": "default", "doc_id": "doc_keep"},
    ]

    changed, skipped = await recompute_doc_ids(mongo, apply=True)

    assert changed == 2
    assert skipped == ["no-path"]
    expected_hangzhou = make_doc_id("default", "景点攻略/杭州.md")
    expected_sanya = make_doc_id("default", "三亚.md")
    documents = {document["_id"]: document for document in mongo.db[DOCUMENTS].documents}
    assert documents["legacy"]["doc_id"] == expected_hangzhou
    assert documents["legacy"]["tenant_id"] == "default"
    assert documents["blank-tenant"]["doc_id"] == expected_sanya
    assert documents["blank-tenant"]["tenant_id"] == "default"
    assert documents["current"]["doc_id"] == make_doc_id("default", "景点攻略/杭州.md")
    assert documents["no-path"]["doc_id"] == "doc_keep"
    assert [query for query, _ in mongo.db[DOCUMENTS].update_one_calls] == [
        {"_id": "legacy"},
        {"_id": "blank-tenant"},
    ]


async def test_drop_legacy_indexes_only_drops_present_names():
    mongo = FakeMongo()
    mongo.db[DOCUMENTS].indexes = ["uq_doc_id", "uq_source_path", "ix_city_type"]
    mongo.db[CHAT_MESSAGES].indexes = ["ix_session_created"]

    present = await drop_legacy_indexes(mongo, apply=True)

    assert present[DOCUMENTS] == ["uq_source_path", "ix_city_type"]
    assert present[INGESTION_TASKS] == []
    assert present[CHAT_SESSIONS] == []
    assert present[CHAT_MESSAGES] == ["ix_session_created"]
    assert mongo.db[DOCUMENTS].dropped_indexes == ["uq_source_path", "ix_city_type"]
    assert mongo.db[CHAT_MESSAGES].dropped_indexes == ["ix_session_created"]
    assert mongo.db[INGESTION_TASKS].dropped_indexes == []
    assert set(LEGACY_INDEXES) == set(TENANT_COLLECTIONS)


def test_should_recreate_truth_table():
    assert should_recreate(healthy=False, recreate_requested=True, force=False) is True
    assert should_recreate(healthy=True, recreate_requested=True, force=False) is False
    assert should_recreate(healthy=True, recreate_requested=True, force=True) is True
    assert should_recreate(healthy=False, recreate_requested=False, force=False) is False
