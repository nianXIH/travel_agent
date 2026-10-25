from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.api.routes import chat as chat_routes
from app.api.routes import search as search_routes
from app.api.routes import sessions as sessions_routes
from app.core.config import Settings
from app.core.exceptions import AppError, NotFoundError
from app.core.tenancy import TenantCredential, TenantRegistry
from app.repositories.mongo import DOCUMENTS
from app.schemas.chat import ChatResponse

VIEWER_KEY = "v" * 32
ADMIN_TOKEN = "admin-secret-token-123456"
DEFAULT_SESSION_ID = "64b000000000000000000001"

VIEWER = TenantCredential(
    tenant_id="sanya", tenant_name="三亚运营", api_key=VIEWER_KEY, role="viewer"
)


class FakeChatService:
    def __init__(self) -> None:
        self.tenants: list[str] = []

    async def chat(self, payload, tenant_id: str) -> ChatResponse:
        self.tenants.append(tenant_id)
        return ChatResponse(
            session_id="session-1",
            message_id="message-1",
            answer=f"{payload.query}/{tenant_id}",
            rewritten_query=payload.query,
            intent="qa",
            citations=[],
            elapsed_ms=1,
        )


class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._sessions = {
            "default": {
                "session_id": DEFAULT_SESSION_ID,
                "title": "默认会话",
                "summary": "",
                "message_count": 1,
                "user_id": None,
            }
        }

    def _session(self, tenant_id: str, session_id: str) -> dict:
        session = self._sessions.get(tenant_id)
        if session is None or session_id != DEFAULT_SESSION_ID:
            raise NotFoundError("session not found")
        return session

    async def get_session(self, tenant_id: str, session_id: str) -> dict:
        self.calls.append(("get_session", tenant_id, session_id))
        return dict(self._session(tenant_id, session_id))

    async def list_messages(self, tenant_id: str, session_id: str, limit: int = 200):
        self.calls.append(("list_messages", tenant_id, session_id))
        self._session(tenant_id, session_id)
        return []

    async def delete_session(self, tenant_id: str, session_id: str) -> bool:
        self.calls.append(("delete_session", tenant_id, session_id))
        if tenant_id not in self._sessions or session_id != DEFAULT_SESSION_ID:
            return False
        return True


class FakeRetrieval:
    def __init__(self) -> None:
        self.tenants: list[str] = []

    def search(self, query: str, *, tenant_id: str, **kwargs: Any) -> list[dict]:
        self.tenants.append(tenant_id)
        return []


class FakeResultCursor:
    async def to_list(self, length: int | None = None) -> list[dict]:
        return [{"_id": {"city": "三亚", "content_type": "景点介绍"}, "count": 2}]


class FakeDocumentCollection:
    def __init__(self) -> None:
        self.pipelines: list[list[dict]] = []

    async def aggregate(self, pipeline: list[dict]) -> FakeResultCursor:
        self.pipelines.append(pipeline)
        return FakeResultCursor()


class FakeMongo:
    def __init__(self) -> None:
        self.documents = FakeDocumentCollection()
        self.db = {DOCUMENTS: self.documents}


def _app(*, memory: FakeMemory | None = None) -> tuple[FastAPI, dict[str, Any]]:
    settings = Settings(
        _env_file=None,
        tenants=[VIEWER],
        admin_token=ADMIN_TOKEN,
        llm_api_key="",
    )
    chat_service = FakeChatService()
    retrieval = FakeRetrieval()
    mongo = FakeMongo()
    memory = memory or FakeMemory()

    app = FastAPI()
    app.state.settings = settings
    app.state.tenants = TenantRegistry(settings.tenants, settings.admin_token)
    app.state.chat_service = chat_service
    app.state.memory = memory
    app.state.retrieval = retrieval
    app.state.mongo = mongo

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    app.include_router(chat_routes.router, prefix="/api/v1")
    app.include_router(sessions_routes.router, prefix="/api/v1")
    app.include_router(search_routes.router, prefix="/api/v1")

    return app, {
        "chat_service": chat_service,
        "retrieval": retrieval,
        "mongo": mongo,
        "memory": memory,
    }


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_viewer_chat_uses_own_tenant():
    app, fakes = _app()

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/chat",
            json={"query": "三亚亲子攻略"},
            headers={"X-API-Key": VIEWER_KEY},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "三亚亲子攻略/sanya"
    assert fakes["chat_service"].tenants == ["sanya"]


async def test_admin_selects_tenant_for_chat():
    app, fakes = _app()
    headers = {"X-Admin-Token": ADMIN_TOKEN}

    async with _client(app) as client:
        sanya = await client.post(
            "/api/v1/chat",
            json={"query": "三亚"},
            headers={**headers, "X-Tenant-Id": "sanya"},
        )
        default = await client.post("/api/v1/chat", json={"query": "默认"}, headers=headers)

    assert sanya.status_code == 200
    assert default.status_code == 200
    assert fakes["chat_service"].tenants == ["sanya", "default"]


async def test_credentials_are_enforced():
    app, _ = _app()

    async with _client(app) as client:
        anonymous = await client.post("/api/v1/chat", json={"query": "hi"})
        mismatch = await client.post(
            "/api/v1/chat",
            json={"query": "hi"},
            headers={"X-API-Key": VIEWER_KEY, "X-Tenant-Id": "default"},
        )

    assert anonymous.status_code == 401
    assert anonymous.json()["code"] == "UNAUTHORIZED"
    assert mismatch.status_code == 403
    assert mismatch.json()["code"] == "TENANT_MISMATCH"


async def test_sessions_are_tenant_scoped():
    app, fakes = _app()
    memory: FakeMemory = fakes["memory"]
    viewer = {"X-API-Key": VIEWER_KEY}
    admin = {"X-Admin-Token": ADMIN_TOKEN}

    async with _client(app) as client:
        sanya_get = await client.get(f"/api/v1/sessions/{DEFAULT_SESSION_ID}", headers=viewer)
        sanya_delete = await client.delete(f"/api/v1/sessions/{DEFAULT_SESSION_ID}", headers=viewer)
        admin_get = await client.get(f"/api/v1/sessions/{DEFAULT_SESSION_ID}", headers=admin)
        admin_delete = await client.delete(f"/api/v1/sessions/{DEFAULT_SESSION_ID}", headers=admin)

    assert sanya_get.status_code == 404
    assert sanya_get.json()["code"] == "NOT_FOUND"
    assert sanya_delete.status_code == 404
    assert admin_get.status_code == 200
    assert admin_get.json()["session"]["session_id"] == DEFAULT_SESSION_ID
    assert admin_delete.status_code == 200
    assert admin_delete.json()["deleted"] is True
    assert ("get_session", "sanya", DEFAULT_SESSION_ID) in memory.calls
    assert ("delete_session", "sanya", DEFAULT_SESSION_ID) in memory.calls
    assert ("get_session", "default", DEFAULT_SESSION_ID) in memory.calls
    assert ("delete_session", "default", DEFAULT_SESSION_ID) in memory.calls


async def test_search_uses_viewer_tenant():
    app, fakes = _app()

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/search",
            json={"query": "亚龙湾"},
            headers={"X-API-Key": VIEWER_KEY},
        )

    assert response.status_code == 200
    assert fakes["retrieval"].tenants == ["sanya"]


async def test_catalog_filters_by_tenant():
    app, fakes = _app()

    async with _client(app) as client:
        response = await client.get("/api/v1/catalog", headers={"X-API-Key": VIEWER_KEY})

    assert response.status_code == 200
    assert response.json()["document_count"] == 2
    pipelines = fakes["mongo"].documents.pipelines
    assert pipelines
    assert pipelines[0][0]["$match"]["tenant_id"] == "sanya"
