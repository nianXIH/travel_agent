import re
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.api.routes import admin
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.tenancy import TenantCredential, TenantRegistry

ADMIN_TOKEN = "admin-secret-token-123456"


class FakeCursor:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = [dict(document) for document in documents]

    def sort(self, field: str, direction: int) -> "FakeCursor":
        self.documents.sort(
            key=lambda document: (
                document.get(field) is None,
                document.get(field) or 0,
            ),
            reverse=direction < 0,
        )
        return self

    def skip(self, offset: int) -> "FakeCursor":
        self.documents = self.documents[offset:]
        return self

    def limit(self, limit: int) -> "FakeCursor":
        self.documents = self.documents[:limit]
        return self

    async def to_list(self, length: int | None = None) -> list[dict]:
        return list(self.documents)


class FakeCollection:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

    async def aggregate(self, pipeline: list[dict]) -> FakeCursor:
        documents = self.documents
        for stage in pipeline:
            if "$match" in stage:
                match = stage["$match"]
                documents = [
                    document
                    for document in documents
                    if all(document.get(key) == value for key, value in match.items())
                ]
            elif "$group" in stage:
                group = stage["$group"]
                field = group["_id"].lstrip("$")
                counts: dict = {}
                for document in documents:
                    key = document.get(field)
                    counts[key] = counts.get(key, 0) + 1
                return FakeCursor(
                    [{"_id": key, "count": count} for key, count in counts.items()]
                )
        return FakeCursor(documents)

    async def count_documents(self, query: dict) -> int:
        return len([document for document in self.documents if _matches(document, query)])

    def find(self, query: dict, projection: dict | None = None) -> FakeCursor:
        documents = [
            dict(document)
            for document in self.documents
            if _matches(document, query)
        ]
        if projection and projection.get("_id") == 0:
            for document in documents:
                document.pop("_id", None)
        return FakeCursor(documents)

    async def find_one(self, query: dict, projection: dict | None = None) -> dict | None:
        cursor = self.find(query, projection)
        return cursor.documents[0] if cursor.documents else None


class FakeMongo:
    def __init__(self, documents: list[dict], tasks: list[dict]) -> None:
        self.db = {
            "documents": FakeCollection(documents),
            "ingestion_tasks": FakeCollection(tasks),
        }


class FakeMilvus:
    def __init__(self, chunks: list[dict] | None = None, error: Exception | None = None) -> None:
        self.chunks = chunks or []
        self.error = error
        self.queries: list[tuple[str, str, int]] = []

    def count_chunks(self, tenant_id: str | None = None) -> int:
        if self.error:
            raise self.error
        if tenant_id is None:
            return len(self.chunks)
        return sum(1 for chunk in self.chunks if chunk.get("tenant_id") == tenant_id)

    def list_document_chunks(
        self, tenant_id: str, doc_id: str, limit: int = 200
    ) -> list[dict]:
        self.queries.append((tenant_id, doc_id, limit))
        if self.error:
            raise self.error
        return [
            chunk
            for chunk in self.chunks
            if chunk.get("doc_id") == doc_id
            and chunk.get("tenant_id") == tenant_id
        ][:limit]


def _matches(document: dict, query: dict) -> bool:
    for key, expected in query.items():
        if key == "$or":
            if not any(_matches(document, clause) for clause in expected):
                return False
            continue
        actual = document.get(key)
        if isinstance(expected, dict):
            flags = re.IGNORECASE if "i" in expected.get("$options", "") else 0
            if not re.search(expected["$regex"], str(actual or ""), flags):
                return False
            continue
        if actual != expected:
            return False
    return True


def _document(**overrides) -> dict:
    base = {
        "doc_id": "d1",
        "tenant_id": "default",
        "title": "三亚亲子攻略",
        "source_file": "sanya.md",
        "source_path": "sanya.md",
        "city": "三亚",
        "content_type": "景点介绍",
        "status": "imported",
        "size": 1024,
        "chunk_count": 3,
        "sha256": "a" * 64,
        "error": "",
        "imported_at": datetime(2026, 10, 24, 10, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 10, 24, 10, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return base


DOCUMENTS = [
    _document(),
    _document(
        doc_id="d2",
        title="三亚海鲜指南",
        source_file="food.md",
        source_path="food.md",
        content_type="美食介绍",
        chunk_count=2,
        updated_at=datetime(2026, 10, 24, 9, 0, tzinfo=UTC),
    ),
    _document(
        doc_id="d3",
        title="杭州两日游",
        source_file="hangzhou.md",
        source_path="hangzhou.md",
        city="杭州",
        content_type="线路推荐",
        status="failed",
        error="RuntimeError: boom",
        updated_at=datetime(2026, 10, 24, 8, 0, tzinfo=UTC),
    ),
    _document(
        doc_id="d4",
        title="旧版三亚攻略",
        source_file="old.md",
        source_path="old.md",
        status="deleted",
        chunk_count=0,
        updated_at=datetime(2026, 10, 24, 7, 0, tzinfo=UTC),
    ),
]

TASKS = [
    {"tenant_id": "default", "status": "success"},
    {"tenant_id": "default", "status": "failed"},
]

CHUNKS = [
    {
        "tenant_id": "default",
        "doc_id": "d1",
        "chunk_id": "d1:1",
        "chunk_index": 1,
        "content": "第二段",
    },
    {
        "tenant_id": "default",
        "doc_id": "d1",
        "chunk_id": "d1:0",
        "chunk_index": 0,
        "content": "第一段",
    },
    {
        "tenant_id": "default",
        "doc_id": "d2",
        "chunk_id": "d2:0",
        "chunk_index": 0,
        "content": "海鲜",
    },
    {
        "tenant_id": "default",
        "doc_id": "d2",
        "chunk_id": "d2:1",
        "chunk_index": 1,
        "content": "餐厅",
    },
    {
        "tenant_id": "default",
        "doc_id": "d3",
        "chunk_id": "d3:0",
        "chunk_index": 0,
        "content": "线路",
    },
]


def _app(
    tmp_path,
    *,
    mongo: FakeMongo | None = None,
    milvus: FakeMilvus | None = None,
    admin_token: str = "",
    tenants: list[TenantCredential] | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        admin_token=admin_token,
        tenants=tenants or [],
        llm_api_key="",
    )
    app.state.tenants = TenantRegistry(
        app.state.settings.tenants, app.state.settings.admin_token
    )
    app.state.mongo = mongo or FakeMongo(DOCUMENTS, TASKS)
    app.state.milvus = milvus or FakeMilvus(CHUNKS)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    app.include_router(admin.router, prefix="/api/v1")
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_overview_reports_document_task_and_chunk_counts(tmp_path):
    app = _app(tmp_path)

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["documents_by_status"] == {"imported": 2, "failed": 1, "deleted": 1}
    assert body["total_documents"] == 4
    assert body["tasks_by_status"] == {"success": 1, "failed": 1}
    assert body["total_tasks"] == 2
    assert body["cities"] == {"三亚": 2}
    assert body["content_types"] == {"景点介绍": 1, "美食介绍": 1}
    assert body["chunks"] == 5
    assert body["chunk_error"] == ""
    assert body["llm_configured"] is False


async def test_overview_degrades_when_milvus_fails(tmp_path):
    app = _app(tmp_path, milvus=FakeMilvus(error=RuntimeError("milvus down")))

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] is None
    assert body["chunk_error"] == "RuntimeError: milvus down"


async def test_documents_support_filters_sort_and_pagination(tmp_path):
    app = _app(tmp_path)

    async with _client(app) as client:
        response = await client.get(
            "/api/v1/admin/documents",
            params={"status": "imported", "city": "三亚", "limit": 1, "offset": 1},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert [item["doc_id"] for item in body["items"]] == ["d2"]


async def test_documents_keyword_is_escaped(tmp_path):
    app = _app(tmp_path)

    async with _client(app) as client:
        literal = await client.get("/api/v1/admin/documents", params={"keyword": "sanya"})
        regex = await client.get("/api/v1/admin/documents", params={"keyword": ".*"})

    assert literal.json()["total"] == 1
    assert regex.json()["total"] == 0


async def test_document_detail_returns_sorted_chunks(tmp_path):
    milvus = FakeMilvus(CHUNKS)
    app = _app(tmp_path, milvus=milvus)

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/documents/d1")

    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == "d1"
    assert [chunk["chunk_index"] for chunk in body["chunks"]] == [0, 1]
    assert body["chunk_error"] == ""
    assert milvus.queries == [("default", "d1", admin.MAX_CHUNK_PREVIEW)]


async def test_document_detail_degrades_when_milvus_fails(tmp_path):
    app = _app(tmp_path, milvus=FakeMilvus(error=RuntimeError("down")))

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/documents/d1")

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == []
    assert body["chunk_error"] == "RuntimeError: down"


async def test_document_detail_returns_404_for_unknown_doc(tmp_path):
    app = _app(tmp_path)

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/documents/missing")

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


async def test_admin_requires_token_when_configured(tmp_path):
    app = _app(tmp_path, admin_token="s3cret")

    async with _client(app) as client:
        denied = await client.get("/api/v1/admin/overview")
        wrong = await client.get(
            "/api/v1/admin/overview",
            headers={"X-Admin-Token": "nope"},
        )
        allowed = await client.get(
            "/api/v1/admin/overview",
            headers={"X-Admin-Token": "s3cret"},
        )

    assert denied.status_code == 401
    assert denied.json()["code"] == "UNAUTHORIZED"
    assert wrong.status_code == 401
    assert allowed.status_code == 200


async def test_admin_tenants_lists_default_in_open_mode(tmp_path):
    app = _app(tmp_path)

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/tenants")

    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "tenant"
    assert body["tenant_id"] == "default"
    assert body["tenants"] == [{"tenant_id": "default", "tenant_name": "默认租户"}]


def _sanya_document() -> dict:
    return _document(
        doc_id="d5",
        source_path="sanya-only.md",
        source_file="sanya-only.md",
        title="三亚专属攻略",
        tenant_id="sanya",
        updated_at=datetime(2026, 10, 24, 11, 0, tzinfo=UTC),
    )


SANYA_ADMIN = TenantCredential(
    tenant_id="sanya", tenant_name="三亚运营", api_key="a" * 32, role="admin"
)


async def test_global_admin_can_filter_documents_and_list_all_tenants(tmp_path):
    viewer = TenantCredential(
        tenant_id="sanya", tenant_name="三亚运营", api_key="v" * 32, role="viewer"
    )
    mongo = FakeMongo(DOCUMENTS + [_sanya_document()], TASKS)
    app = _app(tmp_path, mongo=mongo, admin_token=ADMIN_TOKEN, tenants=[viewer])
    headers = {"X-Admin-Token": ADMIN_TOKEN}

    async with _client(app) as client:
        all_docs = await client.get("/api/v1/admin/documents", headers=headers)
        sanya = await client.get(
            "/api/v1/admin/documents",
            params={"tenant_id": "sanya"},
            headers=headers,
        )
        tenants = await client.get("/api/v1/admin/tenants", headers=headers)

    assert all_docs.json()["total"] == 5
    assert sanya.json()["total"] == 1
    assert sanya.json()["items"][0]["tenant_id"] == "sanya"
    assert tenants.json()["scope"] == "all"
    assert tenants.json()["tenant_id"] is None
    assert {item["tenant_id"] for item in tenants.json()["tenants"]} == {
        "default",
        "sanya",
    }
    assert all("api_key" not in item for item in tenants.json()["tenants"])


async def test_global_admin_unknown_tenant_filter_is_404(tmp_path):
    mongo = FakeMongo(DOCUMENTS + [_sanya_document()], TASKS)
    app = _app(
        tmp_path,
        mongo=mongo,
        admin_token=ADMIN_TOKEN,
        tenants=[SANYA_ADMIN],
    )
    headers = {"X-Admin-Token": ADMIN_TOKEN}

    async with _client(app) as client:
        overview = await client.get(
            "/api/v1/admin/overview", params={"tenant_id": "mars"}, headers=headers
        )
        documents = await client.get(
            "/api/v1/admin/documents", params={"tenant_id": "mars"}, headers=headers
        )

    assert overview.status_code == 404
    assert overview.json()["code"] == "TENANT_NOT_FOUND"
    assert documents.status_code == 404
    assert documents.json()["code"] == "TENANT_NOT_FOUND"


async def test_global_admin_detail_previews_chunks_for_document_tenant(tmp_path):
    milvus = FakeMilvus(
        CHUNKS
        + [
            {
                "tenant_id": "sanya",
                "doc_id": "d5",
                "chunk_id": "d5:0",
                "chunk_index": 0,
                "content": "三亚专属",
            },
            {
                "tenant_id": "default",
                "doc_id": "d5",
                "chunk_id": "d5:1",
                "chunk_index": 1,
                "content": "不应出现",
            },
        ]
    )
    mongo = FakeMongo(DOCUMENTS + [_sanya_document()], TASKS)
    app = _app(
        tmp_path,
        mongo=mongo,
        milvus=milvus,
        admin_token=ADMIN_TOKEN,
        tenants=[SANYA_ADMIN],
    )
    headers = {"X-Admin-Token": ADMIN_TOKEN}

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/documents/d5", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "sanya"
    assert [chunk["content"] for chunk in body["chunks"]] == ["三亚专属"]
    assert milvus.queries == [("sanya", "d5", admin.MAX_CHUNK_PREVIEW)]


async def test_tenant_admin_is_scoped_to_own_tenant(tmp_path):
    milvus = FakeMilvus(
        CHUNKS
        + [
            {
                "tenant_id": "sanya",
                "doc_id": "d5",
                "chunk_id": "d5:0",
                "chunk_index": 0,
                "content": "三亚专属",
            }
        ]
    )
    mongo = FakeMongo(DOCUMENTS + [_sanya_document()], TASKS)
    app = _app(tmp_path, mongo=mongo, milvus=milvus, tenants=[SANYA_ADMIN])
    headers = {"X-API-Key": "a" * 32}

    async with _client(app) as client:
        overview = await client.get("/api/v1/admin/overview", headers=headers)
        documents = await client.get("/api/v1/admin/documents", headers=headers)
        tenants = await client.get("/api/v1/admin/tenants", headers=headers)
        raise_other = await client.get(
            "/api/v1/admin/documents",
            params={"tenant_id": "default"},
            headers=headers,
        )
        other_detail = await client.get("/api/v1/admin/documents/d1", headers=headers)
        own_detail = await client.get("/api/v1/admin/documents/d5", headers=headers)

    assert overview.status_code == 200
    assert overview.json()["total_documents"] == 1
    assert overview.json()["documents_by_status"] == {"imported": 1}
    assert overview.json()["total_tasks"] == 0
    assert overview.json()["chunks"] == 1

    assert documents.json()["total"] == 1
    assert documents.json()["items"][0]["tenant_id"] == "sanya"

    assert tenants.json()["scope"] == "tenant"
    assert tenants.json()["tenant_id"] == "sanya"
    assert tenants.json()["tenants"] == [
        {"tenant_id": "sanya", "tenant_name": "三亚运营"}
    ]

    assert raise_other.status_code == 403
    assert raise_other.json()["code"] == "TENANT_MISMATCH"
    assert other_detail.status_code == 404
    assert other_detail.json()["code"] == "NOT_FOUND"
    assert own_detail.status_code == 200
    assert own_detail.json()["chunks"][0]["content"] == "三亚专属"
