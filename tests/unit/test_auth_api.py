from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.api.deps import require_admin, require_role, target_tenant
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.tenancy import TenantContext, TenantCredential, TenantRegistry

VIEWER = TenantCredential(
    tenant_id="sanya", tenant_name="三亚运营", api_key="v" * 32, role="viewer"
)
EDITOR = TenantCredential(
    tenant_id="default", tenant_name="默认租户", api_key="e" * 32, role="editor"
)
ADMIN = TenantCredential(
    tenant_id="default", tenant_name="默认租户", api_key="a" * 32, role="admin"
)
ADMIN_TOKEN = "admin-secret-token-123456"


def _app(
    *,
    tenants: list[TenantCredential] | None = None,
    admin_token: str = "",
) -> FastAPI:
    settings = Settings(_env_file=None, tenants=tenants or [], admin_token=admin_token)
    app = FastAPI()
    app.state.settings = settings
    app.state.tenants = TenantRegistry(settings.tenants, settings.admin_token)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    @app.get("/viewer")
    async def viewer(ctx: Annotated[TenantContext, Depends(require_role("viewer"))]):
        return {"tenant_id": target_tenant(ctx), "role": ctx.role}

    @app.get("/editor")
    async def editor(ctx: Annotated[TenantContext, Depends(require_role("editor"))]):
        return {"tenant_id": target_tenant(ctx), "role": ctx.role}

    @app.get("/admin")
    async def admin(ctx: Annotated[TenantContext, Depends(require_admin)]):
        return {
            "tenant_id": target_tenant(ctx),
            "all": target_tenant(ctx, allow_all=True),
            "tenant_name": ctx.tenant_name,
        }

    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_tenant_mode_requires_credentials():
    app = _app(tenants=[VIEWER, EDITOR])

    async with _client(app) as client:
        response = await client.get("/viewer")

    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


async def test_api_key_resolves_tenant_and_role():
    app = _app(tenants=[VIEWER, EDITOR])

    async with _client(app) as client:
        viewer = await client.get("/viewer", headers={"X-API-Key": "v" * 32})
        denied = await client.get("/editor", headers={"X-API-Key": "v" * 32})
        editor = await client.get("/editor", headers={"X-API-Key": "e" * 32})

    assert viewer.status_code == 200
    assert viewer.json() == {"tenant_id": "sanya", "role": "viewer"}
    assert denied.status_code == 403
    assert denied.json()["code"] == "FORBIDDEN"
    assert editor.status_code == 200
    assert editor.json()["tenant_id"] == "default"


async def test_tenant_user_cannot_request_other_tenant():
    app = _app(tenants=[VIEWER, EDITOR])

    async with _client(app) as client:
        response = await client.get(
            "/viewer",
            headers={"X-API-Key": "v" * 32, "X-Tenant-Id": "default"},
        )

    assert response.status_code == 403
    assert response.json()["code"] == "TENANT_MISMATCH"


async def test_global_admin_can_select_tenant():
    app = _app(tenants=[VIEWER, EDITOR], admin_token=ADMIN_TOKEN)

    async with _client(app) as client:
        default = await client.get("/admin", headers={"X-Admin-Token": ADMIN_TOKEN})
        sanya = await client.get(
            "/admin",
            headers={"X-Admin-Token": ADMIN_TOKEN, "X-Tenant-Id": "sanya"},
        )
        missing = await client.get(
            "/admin",
            headers={"X-Admin-Token": ADMIN_TOKEN, "X-Tenant-Id": "mars"},
        )

    assert default.status_code == 200
    assert default.json() == {
        "tenant_id": "default",
        "all": None,
        "tenant_name": "全局管理员",
    }
    assert sanya.json() == {
        "tenant_id": "sanya",
        "all": "sanya",
        "tenant_name": "全局管理员",
    }
    assert missing.status_code == 404
    assert missing.json()["code"] == "TENANT_NOT_FOUND"


async def test_tenant_admin_key_can_access_admin_routes():
    app = _app(tenants=[VIEWER, ADMIN])

    async with _client(app) as client:
        response = await client.get("/admin", headers={"X-API-Key": "a" * 32})

    assert response.status_code == 200
    assert response.json()["tenant_id"] == "default"


async def test_open_mode_defaults_to_default_and_ignores_bad_api_key():
    app = _app()

    async with _client(app) as client:
        plain = await client.get("/viewer")
        bogus = await client.get("/viewer", headers={"X-API-Key": "whatever"})
        other = await client.get("/viewer", headers={"X-Tenant-Id": "sanya"})

    assert plain.status_code == 200
    assert plain.json()["tenant_id"] == "default"
    assert bogus.status_code == 200
    assert other.status_code == 404


async def test_open_mode_preserves_admin_token_guard():
    app = _app(admin_token=ADMIN_TOKEN)

    async with _client(app) as client:
        denied = await client.get("/admin")
        allowed = await client.get("/admin", headers={"X-Admin-Token": ADMIN_TOKEN})

    assert denied.status_code == 401
    assert allowed.status_code == 200
