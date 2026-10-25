# 权限与多租户实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为旅游知识库增加 API Key 租户识别、三级角色、文档/分块/任务/会话的完全租户隔离，并提供存量数据迁移与跨租户管理后台。

**Architecture:** 在 FastAPI 依赖层解析 `X-API-Key` / `X-Admin-Token` / `X-Tenant-Id`，生成 `TenantContext`；MongoDB 所有业务集合增加 `tenant_id` 并重建租户前缀索引；Milvus 单集合增加 `tenant_id` 标量字段，所有过滤表达式强制带租户；`doc_id` 改为 `sha1(tenant_id:relative_path)`；数据目录按 `DATA_DIR/tenants/{tenant_id}` 隔离。未配置 `TENANTS` 时保持现有开放行为。

**Tech Stack:** Python 3.12, FastAPI, Pydantic Settings, MongoDB (AsyncMongoClient), Milvus (pymilvus), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-25-multi-tenancy-design.md`

## Global Constraints

- Python 版本下限：`>=3.12`；ruff `target-version = py312`、`line-length = 100`，规则 `E,F,I,UP,B,ASYNC`。
- 单元测试默认命令：`pytest`（`addopts = "-ra -m 'not integration'"`）；集成测试：`pytest -m integration`。
- 现有单元测试基线：**118 passed, 10 deselected**。任务 3–9 会短期修改共享签名；在这些任务之间只运行本任务指定的目标测试，**任务 9 与任务 13 必须恢复全量绿**。
- 所有密钥比较使用 `secrets.compare_digest`；任何日志、响应、管理页不得返回 API Key。
- `ADMIN_TOKEN` 不得与任何租户 `api_key` 相同，否则同一凭证会被解析为全局管理员。
- 未配置 `TENANTS` 时为兼容模式：匿名请求视为 `default` 租户管理员；若同时配置了 `ADMIN_TOKEN`，沿用旧行为，`/admin/*` 必须携带匹配的 `X-Admin-Token`。
- 租户 ID 必须匹配 `^[a-z0-9][a-z0-9_-]{0,63}$`；跨租户资源统一返回 404，租户用户显式选择他人租户返回 403 `TENANT_MISMATCH`。
- 不得读取或提交 `.env` 中的真实密钥；示例一律使用占位字符串。
- 只有用户明确授权后才能执行 `git commit`；计划中的 Commit 步骤默认跳过。

---

## 文件结构

新增：

```text
app/core/tenancy.py
scripts/migrate_multitenant.py
tests/unit/test_tenancy.py
tests/unit/test_auth_api.py
tests/unit/test_filters.py
tests/integration/test_tenant_isolation.py
docs/superpowers/plans/2026-10-25-multi-tenancy.md
```

修改：

```text
app/core/config.py
app/api/deps.py
app/main.py
app/api/routes/{admin,chat,ingest,planning,search,sessions}.py
app/schemas/{admin,ingestion}.py
app/repositories/mongo.py
app/ingestion/{metadata,scanner,pipeline,task_repo}.py
app/workers/ingest_worker.py
app/retrieval/{filters,milvus_repo,service,three_way}.py
app/chat/{state,memory,nodes,service}.py
app/planner/service.py
app/static/admin/{index.html,admin.css,admin.js}
scripts/{ingest_dir,eval_retrieval,eval_qa}.py
tests/unit/{test_admin_api,test_chat_nodes,test_ingest_api,test_ingest_worker,test_ingestion_pipeline,test_ingestion_repo,test_planner_service,test_planning_api,test_three_way}.py
tests/integration/{test_ingest_retrieval_loop,test_milvus_search,test_mongo_ingestion_repo}.py
.env.example
项目架构.txt
任务进度.md
```

---

### Task 1: 租户模型、注册表与配置

**Files:**
- Create: `app/core/tenancy.py`
- Modify: `app/core/config.py:1-30, 95-118`
- Test: `tests/unit/test_tenancy.py`

**Interfaces:**
- Produces: `DEFAULT_TENANT_ID`, `DEFAULT_TENANT_NAME`, `TENANTS_DIR`, `TENANT_ID_RE`, `Role`, `ROLE_RANK`, `TenantCredential`, `TenantMeta`, `TenantContext`, `TenantRegistry`.
- Produces: `Settings.tenants`, `Settings.tenant_data_path(tenant_id) -> Path`, `Settings.tenant_upload_path(tenant_id) -> Path`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_tenancy.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.tenancy import (
    DEFAULT_TENANT_ID,
    ROLE_RANK,
    TenantCredential,
    TenantRegistry,
)
from app.ingestion.metadata import make_doc_id


def _credential(**overrides) -> TenantCredential:
    values = {
        "tenant_id": "sanya",
        "tenant_name": "三亚运营",
        "api_key": "k" * 32,
        "role": "viewer",
    }
    values.update(overrides)
    return TenantCredential(**values)


def test_tenant_id_must_be_slug():
    for bad in ("Sanya", "三亚", "-sanya", "a" * 65):
        with pytest.raises(ValidationError):
            _credential(tenant_id=bad)


def test_api_key_min_length():
    with pytest.raises(ValidationError):
        _credential(api_key="short")


def test_registry_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate"):
        TenantRegistry([_credential(), _credential(tenant_id="xiamen", tenant_name="厦门")])


def test_registry_rejects_inconsistent_names():
    with pytest.raises(ValueError, match="tenant_name"):
        TenantRegistry(
            [
                _credential(role="viewer"),
                _credential(
                    tenant_id="sanya",
                    tenant_name="另一个名字",
                    api_key="z" * 32,
                    role="editor",
                ),
            ]
        )


def test_registry_always_contains_default_and_accepts_configured_name():
    registry = TenantRegistry(
        [_credential(tenant_id=DEFAULT_TENANT_ID, tenant_name="总部", role="editor")]
    )

    assert registry.open_mode is False
    assert registry.get_tenant(DEFAULT_TENANT_ID).tenant_name == "总部"
    assert {meta.tenant_id for meta in registry.list_tenants()} == {"default"}


def test_registry_uses_last_matching_key_without_early_return():
    first = _credential(tenant_id="a", tenant_name="A")
    second = _credential(tenant_id="b", tenant_name="B", api_key="z" * 32)
    registry = TenantRegistry([first, second])

    assert registry.verify_api_key("z" * 32).tenant_id == "b"
    assert registry.verify_api_key("missing") is None


def test_open_mode_and_admin_token():
    registry = TenantRegistry([], "admin-secret")

    assert registry.open_mode is True
    assert registry.verify_admin_token("admin-secret") is True
    assert registry.verify_admin_token("wrong") is False
    assert TenantRegistry().verify_admin_token("") is False


def test_role_rank_is_ordered():
    assert ROLE_RANK["viewer"] < ROLE_RANK["editor"] < ROLE_RANK["admin"]


def test_tenant_paths_are_scoped_and_reject_traversal(tmp_path: Path):
    settings = Settings(_env_file=None, data_dir=tmp_path)

    assert settings.tenant_data_path(DEFAULT_TENANT_ID) == settings.data_path
    assert settings.tenant_data_path("sanya") == (tmp_path / "tenants" / "sanya").resolve()
    assert settings.tenant_upload_path("sanya") == (
        tmp_path / "tenants" / "sanya" / "uploads"
    ).resolve()
    with pytest.raises(ValueError):
        settings.tenant_data_path("../escape")


def test_doc_id_is_scoped_by_tenant():
    assert make_doc_id("default", "景点攻略/杭州.md") != make_doc_id(
        "sanya", "景点攻略/杭州.md"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_tenancy.py -q`

Expected: FAIL with `ModuleNotFoundError: app.core.tenancy` (or `make_doc_id` arity error).

- [ ] **Step 3: Implement `app/core/tenancy.py`**

```python
import re
import secrets
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_TENANT_ID = "default"
DEFAULT_TENANT_NAME = "默认租户"
TENANTS_DIR = "tenants"
TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

Role = Literal["viewer", "editor", "admin"]
ROLE_RANK = {"viewer": 1, "editor": 2, "admin": 3}


class TenantCredential(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=64)
    tenant_name: str = Field(min_length=1, max_length=64)
    api_key: str = Field(min_length=16)
    role: Role = "viewer"

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: str) -> str:
        if TENANT_ID_RE.fullmatch(value) is None:
            raise ValueError("tenant_id must match ^[a-z0-9][a-z0-9_-]{0,63}$")
        return value


@dataclass(frozen=True)
class TenantMeta:
    tenant_id: str
    tenant_name: str


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str | None
    tenant_name: str
    role: str
    is_global_admin: bool = False
    requested_tenant_id: str | None = None
    open_mode: bool = False
    admin_verified: bool = False


class TenantRegistry:
    def __init__(
        self,
        credentials: list[TenantCredential] | None = None,
        admin_token: str = "",
    ) -> None:
        self._credentials = list(credentials or [])
        self._admin_token = admin_token
        self._validate_keys()

        names: dict[str, str] = {}
        for credential in self._credentials:
            existing = names.get(credential.tenant_id)
            if existing is not None and existing != credential.tenant_name:
                raise ValueError(
                    f"tenant {credential.tenant_id} has inconsistent tenant_name"
                )
            names[credential.tenant_id] = credential.tenant_name
        names.setdefault(DEFAULT_TENANT_ID, DEFAULT_TENANT_NAME)
        self._tenants = {
            tenant_id: TenantMeta(tenant_id=tenant_id, tenant_name=name)
            for tenant_id, name in names.items()
        }

    def _validate_keys(self) -> None:
        seen: set[str] = set()
        for credential in self._credentials:
            if credential.api_key in seen:
                raise ValueError("duplicate tenant api_key")
            seen.add(credential.api_key)

    @property
    def open_mode(self) -> bool:
        return not self._credentials

    def verify_api_key(self, api_key: str | None) -> TenantCredential | None:
        provided = (api_key or "").encode("utf-8")
        matched: TenantCredential | None = None
        for credential in self._credentials:
            if secrets.compare_digest(provided, credential.api_key.encode("utf-8")):
                matched = credential
        return matched

    def verify_admin_token(self, token: str | None) -> bool:
        if not self._admin_token:
            return False
        provided = (token or "").encode("utf-8")
        return secrets.compare_digest(provided, self._admin_token.encode("utf-8"))

    def get_tenant(self, tenant_id: str) -> TenantMeta | None:
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> list[TenantMeta]:
        return list(self._tenants.values())
```

- [ ] **Step 4: Update `app/core/config.py`**

Add imports:

```python
from app.core.tenancy import (
    DEFAULT_TENANT_ID,
    TENANT_ID_RE,
    TENANTS_DIR,
    TenantCredential,
)
```

Add field after `admin_token`:

```python
    tenants: list[TenantCredential] = Field(default_factory=list)
```

Add methods after `data_path`:

```python
    def tenant_data_path(self, tenant_id: str) -> Path:
        if tenant_id == DEFAULT_TENANT_ID:
            return self.data_path
        if TENANT_ID_RE.fullmatch(tenant_id) is None:
            raise ValueError(f"invalid tenant id: {tenant_id}")
        path = (self.data_path / TENANTS_DIR / tenant_id).resolve()
        if not path.is_relative_to(self.data_path):
            raise ValueError(f"invalid tenant id: {tenant_id}")
        return path

    def tenant_upload_path(self, tenant_id: str) -> Path:
        return self.tenant_data_path(tenant_id) / "uploads"
```

- [ ] **Step 5: Update `make_doc_id` in `app/ingestion/metadata.py`**

```python
def make_doc_id(tenant_id: str, relative_path: str) -> str:
    digest = hashlib.sha1(f"{tenant_id}:{relative_path}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:16]}"
```

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_tenancy.py -q`

Expected: PASS. (`tests/unit/test_ingestion_pipeline.py` will fail on `make_doc_id` arity until Task 5; that is expected.)

- [ ] **Step 7: Commit (only if authorized)**

```bash
git add app/core/tenancy.py app/core/config.py app/ingestion/metadata.py tests/unit/test_tenancy.py
git commit -m "feat: add tenant registry and scoped paths"
```

---

### Task 2: 鉴权依赖与启动注册

**Files:**
- Modify: `app/api/deps.py:1-69`
- Modify: `app/main.py:89-99`
- Modify: `tests/unit/test_admin_api.py` (only `_app` setup; tenant cases arrive in Task 10)
- Test: `tests/unit/test_auth_api.py`

**Interfaces:**
- Consumes: `TenantRegistry`, `TenantContext`, `ROLE_RANK`, `DEFAULT_TENANT_ID`.
- Produces: `get_tenant_registry(request) -> TenantRegistry`, `get_tenant(...) -> TenantContext`, `require_role(minimum) -> dependency`, `require_admin(...) -> TenantContext`, `target_tenant(ctx, allow_all=False) -> str | None`.
- Produces: `app.state.tenants` set in `create_app()`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_auth_api.py`:

```python
from typing import Annotated

import pytest
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_auth_api.py -q`

Expected: FAIL with `ImportError: cannot import name 'require_role'`.

- [ ] **Step 3: Implement dependencies in `app/api/deps.py`**

Replace the imports and `require_admin` with:

```python
import secrets
from typing import Annotated

from fastapi import Header, Request

from app.chat.llm import ChatLLM
from app.chat.memory import ChatMemory
from app.chat.service import ChatService
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.tenancy import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    ROLE_RANK,
    TenantContext,
    TenantCredential,
    TenantRegistry,
)
from app.ingestion.task_repo import IngestionRepository
from app.planner.service import PlannerService
from app.repositories.mongo import MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService
from app.workers.ingest_worker import IngestionWorker
```

Append after the existing getters:

```python
def get_tenant_registry(request: Request) -> TenantRegistry:
    return request.app.state.tenants


def get_tenant(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> TenantContext:
    registry: TenantRegistry = request.app.state.tenants
    requested = (x_tenant_id or "").strip() or None

    if registry.open_mode:
        if requested and requested != DEFAULT_TENANT_ID:
            raise AppError("租户不存在", status_code=404, code="TENANT_NOT_FOUND")
        return TenantContext(
            tenant_id=DEFAULT_TENANT_ID,
            tenant_name=DEFAULT_TENANT_NAME,
            role="admin",
            requested_tenant_id=requested,
            open_mode=True,
            admin_verified=registry.verify_admin_token(x_admin_token),
        )

    if registry.verify_admin_token(x_admin_token):
        if requested and registry.get_tenant(requested) is None:
            raise AppError("租户不存在", status_code=404, code="TENANT_NOT_FOUND")
        return TenantContext(
            tenant_id=None,
            tenant_name="全局管理员",
            role="admin",
            is_global_admin=True,
            requested_tenant_id=requested,
            admin_verified=True,
        )

    credential: TenantCredential | None = registry.verify_api_key(x_api_key)
    if credential is None:
        raise AppError("需要有效的访问凭证", status_code=401, code="UNAUTHORIZED")
    if requested and requested != credential.tenant_id:
        raise AppError("无权访问其他租户", status_code=403, code="TENANT_MISMATCH")
    return TenantContext(
        tenant_id=credential.tenant_id,
        tenant_name=credential.tenant_name,
        role=credential.role,
        requested_tenant_id=requested,
    )


def require_role(minimum: str):
    def dependency(
        ctx: Annotated[TenantContext, Depends(get_tenant)],
    ) -> TenantContext:
        if ctx.is_global_admin or ctx.open_mode:
            return ctx
        if ROLE_RANK.get(ctx.role, 0) < ROLE_RANK[minimum]:
            raise AppError("权限不足", status_code=403, code="FORBIDDEN")
        return ctx

    return dependency


def require_admin(
    request: Request,
    ctx: Annotated[TenantContext, Depends(get_tenant)],
) -> TenantContext:
    if ctx.open_mode:
        expected = request.app.state.settings.admin_token
        if expected and not ctx.admin_verified:
            raise AppError("管理员令牌无效", status_code=401, code="UNAUTHORIZED")
        return ctx
    if not ctx.is_global_admin and ROLE_RANK.get(ctx.role, 0) < ROLE_RANK["admin"]:
        raise AppError("权限不足", status_code=403, code="FORBIDDEN")
    return ctx


def target_tenant(ctx: TenantContext, *, allow_all: bool = False) -> str | None:
    if not ctx.is_global_admin:
        return ctx.tenant_id
    if ctx.requested_tenant_id:
        return ctx.requested_tenant_id
    return None if allow_all else DEFAULT_TENANT_ID
```

Add `from fastapi import Depends` to the FastAPI import line:

```python
from fastapi import Depends, Header, Request
```

- [ ] **Step 4: Register the registry in `app/main.py`**

Import:

```python
from app.core.tenancy import TenantRegistry
```

In `create_app()` immediately after `app.state.settings = settings`:

```python
    app.state.tenants = TenantRegistry(settings.tenants, settings.admin_token)
```

- [ ] **Step 5: Install tenant state in the existing admin API test app**

`require_admin` now depends on `get_tenant`, which reads `request.app.state.tenants`. Update `_app()` in `tests/unit/test_admin_api.py` so the existing suite keeps working before Task 10 expands it:

```python
from app.core.tenancy import TenantRegistry


def _app(
    tmp_path,
    *,
    mongo: FakeMongo | None = None,
    milvus: FakeMilvus | None = None,
    admin_token: str = "",
) -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        admin_token=admin_token,
        llm_api_key="",
    )
    app.state.tenants = TenantRegistry(
        app.state.settings.tenants, app.state.settings.admin_token
    )
    app.state.mongo = mongo or FakeMongo(DOCUMENTS, TASKS)
    app.state.milvus = milvus or FakeMilvus(CHUNKS)
    ...
```

Do not add the tenant-specific assertions yet; Task 10 rewrites this helper with the `tenants` parameter.

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_auth_api.py tests/unit/test_admin_api.py -q`

Expected: PASS. `test_admin_requires_token_when_configured` still passes because open-mode `require_admin` checks `admin_token`.

- [ ] **Step 7: Commit (only if authorized)**

```bash
git add app/api/deps.py app/main.py tests/unit/test_auth_api.py
git commit -m "feat: add tenant-aware auth dependencies"
```

---

### Task 3: MongoDB 租户索引、仓储与 doc_id

**Files:**
- Modify: `app/repositories/mongo.py:12-34`
- Modify: `app/ingestion/task_repo.py:14-135`
- Test: `tests/unit/test_ingestion_repo.py`
- Test: `tests/integration/test_mongo_ingestion_repo.py` (index expectations only)

**Interfaces:**
- Produces: `IngestionRepository.create_task(tenant_id, task_type, total, *, status="running")`
- Produces: `update_task(tenant_id, task_id, **fields)`
- Produces: `get_task(tenant_id: str | None, task_id)`
- Produces: `list_tasks(tenant_id: str | None, *, status=None, limit=20, offset=0)`
- Produces: `get_document(tenant_id, source_path)`, `list_documents(tenant_id, *, status=None)`, `upsert_document(tenant_id, document)`, `mark_document_deleted(tenant_id, *, source_path, doc_id)`, `mark_document_failed(tenant_id, *, source_path, doc_id, source_file, error)`.
- `tenant_id=None` is allowed only for task reads and means “all tenants” (global admin only).

- [ ] **Step 1: Write failing tests**

Replace `tests/unit/test_ingestion_repo.py` test bodies with tenant-aware calls and add an isolation test:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_ingestion_repo.py -q`

Expected: FAIL with `TypeError: create_task() takes 3 positional arguments but 4 were given`.

- [ ] **Step 3: Replace index definitions in `app/repositories/mongo.py`**

```python
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
```

- [ ] **Step 4: Replace `app/ingestion/task_repo.py`**

Keep `utc_now()` and imports. Replace the class body with:

```python
def _scope(tenant_id: str | None, query: dict | None = None) -> dict:
    scoped = dict(query or {})
    if tenant_id is not None:
        scoped["tenant_id"] = tenant_id
    return scoped


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
        now = utc_now()
        record = {**document, "tenant_id": tenant_id, "updated_at": now}
        return await self._db[DOCUMENTS].find_one_and_update(
            {"tenant_id": tenant_id, "source_path": record["source_path"]},
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
```

- [ ] **Step 5: Update integration index assertions**

In `tests/integration/test_mongo_ingestion_repo.py`:

```python
    assert "uq_doc_id" in document_indexes
    assert "uq_tenant_source" in document_indexes
    assert document_indexes["uq_doc_id"].get("unique") is True
    assert document_indexes["uq_tenant_source"].get("unique") is True

    task_indexes = await mongo.db[INGESTION_TASKS].index_information()
    assert "ix_tenant_status_created" in task_indexes
```

Update all repository calls in that file to pass `"default"` as the first argument, e.g.:

```python
task_id = await repo.create_task("default", task_type, 2)
created = await repo.get_task("default", task_id)
await repo.update_task("default", task_id, status="partial", ...)
total, tasks = await repo.list_tasks("default", status="partial", limit=20)
assert await repo.get_task("default", "not-an-object-id") is None
await repo.upsert_document("default", {...})
first = await repo.get_document("default", source_path)
await repo.mark_document_failed("default", source_path=..., doc_id=..., ...)
```

Also scope cleanup deletes:

```python
await mongo.db[INGESTION_TASKS].delete_one({"_id": ObjectId(task_id), "tenant_id": "default"})
await mongo.db[DOCUMENTS].delete_one({"tenant_id": "default", "source_path": source_path})
```

- [ ] **Step 6: Run targeted tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_ingestion_repo.py -q`

Expected: PASS. Pipeline/worker/route tests still fail on the new signatures until Tasks 5–6.

- [ ] **Step 7: Commit (only if authorized)**

```bash
git add app/repositories/mongo.py app/ingestion/task_repo.py tests/unit/test_ingestion_repo.py tests/integration/test_mongo_ingestion_repo.py
git commit -m "feat: scope mongo persistence by tenant"
```

---

### Task 4: Milvus 租户字段与强制过滤

**Files:**
- Modify: `app/retrieval/filters.py:1-30`
- Modify: `app/retrieval/milvus_repo.py:17-42, 45-69, 221-265, 261-318`
- Test: `tests/unit/test_filters.py`

**Interfaces:**
- Produces: `build_filter(*, tenant_id: str, ...) -> str` (tenant expression always first; empty tenant raises `ValueError`).
- Produces: `delete_stale_chunks(tenant_id, doc_id, keep_chunk_ids)`, `delete_document_chunks(tenant_id, doc_id)`, `count_chunks(tenant_id=None)`, `list_document_chunks(tenant_id, doc_id, limit=200)`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_filters.py`:

```python
import pytest

from app.retrieval.filters import build_filter


def test_filter_always_starts_with_tenant():
    expression = build_filter(tenant_id="sanya", city="三亚", content_type="景点介绍")

    assert expression.startswith('tenant_id == "sanya"')
    assert 'city == "三亚"' in expression
    assert 'content_type == "景点介绍"' in expression


def test_filter_uses_tenant_even_without_other_filters():
    assert build_filter(tenant_id="default") == 'tenant_id == "default"'


def test_filter_escapes_quotes_and_backslashes():
    expression = build_filter(tenant_id='a"b\\c')

    assert expression == 'tenant_id == "a\\"b\\\\c"'


def test_filter_rejects_empty_tenant():
    with pytest.raises(ValueError, match="tenant_id"):
        build_filter(tenant_id="")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_filters.py -q`

Expected: FAIL with `TypeError: build_filter() missing 1 required keyword-only argument: 'tenant_id'`.

- [ ] **Step 3: Update `app/retrieval/filters.py`**

```python
def build_filter(
    *,
    tenant_id: str,
    city: str | None = None,
    content_type: str | None = None,
    scenic_name: str | None = None,
    route_name: str | None = None,
    hotel_name: str | None = None,
    doc_ids: list[str] | None = None,
) -> str:
    if not tenant_id:
        raise ValueError("tenant_id is required")
    expressions: list[str] = [f"tenant_id == {quote(tenant_id)}"]
    for field, value in (
        ("city", city),
        ("content_type", content_type),
        ("scenic_name", scenic_name),
        ("route_name", route_name),
        ("hotel_name", hotel_name),
    ):
        if value:
            expressions.append(f"{field} == {quote(value)}")

    if doc_ids:
        joined = ", ".join(quote(doc_id) for doc_id in doc_ids)
        expressions.append(f"doc_id in [{joined}]")

    return " and ".join(expressions)
```

- [ ] **Step 4: Update Milvus schema and repository**

In `app/retrieval/milvus_repo.py`:

```python
OUTPUT_FIELDS = [
    "tenant_id",
    "doc_id",
    ...
]

SCALAR_INDEXES = (
    "tenant_id",
    "doc_id",
    ...
)
```

In `build_schema()` add immediately after `doc_id`:

```python
            _varchar("tenant_id", 64),
```

Replace the deletion/count/preview methods:

```python
    def delete_stale_chunks(
        self,
        tenant_id: str,
        doc_id: str,
        keep_chunk_ids: list[str],
    ) -> None:
        expression = f"tenant_id == {quote(tenant_id)} and doc_id == {quote(doc_id)}"
        if keep_chunk_ids:
            joined = ", ".join(quote(chunk_id) for chunk_id in keep_chunk_ids)
            expression += f" and chunk_id not in [{joined}]"
        self._client.delete(self._collection, filter=expression)

    def delete_document_chunks(self, tenant_id: str, doc_id: str) -> None:
        self._client.delete(
            self._collection,
            filter=f"tenant_id == {quote(tenant_id)} and doc_id == {quote(doc_id)}",
        )

    def count_chunks(self, tenant_id: str | None = None) -> int:
        expression = 'chunk_id != ""'
        if tenant_id is not None:
            expression = f"tenant_id == {quote(tenant_id)} and {expression}"
        result = self._client.query(
            self._collection,
            filter=expression,
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"])

    def list_document_chunks(
        self,
        tenant_id: str | None,
        doc_id: str,
        limit: int = 200,
    ) -> list[dict]:
        expression = f"doc_id == {quote(doc_id)}"
        if tenant_id is not None:
            expression = f"tenant_id == {quote(tenant_id)} and {expression}"
        return self._client.query(
            self._collection,
            filter=expression,
            output_fields=[
                "chunk_id",
                "chunk_index",
                "section_path",
                "content",
                "content_type",
                "city",
                "updated_at",
            ],
            limit=limit,
        )
```

In `validate()` add `"tenant_id": False,` to the `expected` dict after `"doc_id": False`.

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_filters.py -q`

Expected: PASS.

- [ ] **Step 6: Commit (only if authorized)**

```bash
git add app/retrieval/filters.py app/retrieval/milvus_repo.py tests/unit/test_filters.py
git commit -m "feat: enforce tenant filters in milvus"
```

---

### Task 5: 导入链路、扫描器与 Worker

**Files:**
- Modify: `app/ingestion/scanner.py:1-44`
- Modify: `app/ingestion/pipeline.py:51-306`
- Modify: `app/workers/ingest_worker.py:28-101`
- Test: `tests/unit/test_ingestion_pipeline.py`
- Test: `tests/unit/test_ingest_worker.py`

**Interfaces:**
- Produces: `scan_directory(directory, suffixes=None, *, exclude_dirs=None)`.
- Produces: `scan_tenant_directory(settings, tenant_id) -> list[ScannedFile]`.
- Produces: `IngestionPipeline.run(..., *, tenant_id=DEFAULT_TENANT_ID, ...)` and `run_task(task_id, files, *, tenant_id=DEFAULT_TENANT_ID, ...)`.
- Produces: `IngestionWorker.submit(tenant_id, task_id, files, *, force=False, prune=False)`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_ingestion_pipeline.py`, update the fakes:

```python
class FakeRepo:
    def __init__(self) -> None:
        self.tasks: list[tuple[str, str, dict]] = []
        self.documents: dict[tuple[str, str], dict] = {}
        self.failures: list[dict] = []
        self.deleted: list[dict] = []

    async def get_document(self, tenant_id: str, source_path: str) -> dict | None:
        return self.documents.get((tenant_id, source_path))

    async def upsert_document(self, tenant_id: str, document: dict) -> dict:
        self.documents[(tenant_id, document["source_path"])] = document
        return document

    async def mark_document_failed(self, tenant_id: str, **fields) -> None:
        self.failures.append({"tenant_id": tenant_id, **fields})

    async def list_documents(self, tenant_id: str) -> list[dict]:
        return [
            document
            for (owner, _), document in self.documents.items()
            if owner == tenant_id
        ]

    async def mark_document_deleted(
        self, tenant_id: str, *, source_path: str, doc_id: str
    ) -> None:
        self.deleted.append(
            {"tenant_id": tenant_id, "source_path": source_path, "doc_id": doc_id}
        )
        if (tenant_id, source_path) in self.documents:
            self.documents[(tenant_id, source_path)]["status"] = "deleted"

    async def update_task(self, tenant_id: str, task_id: str, **fields) -> None:
        self.tasks.append((tenant_id, task_id, fields))


class FakeMilvus:
    ...
    def delete_stale_chunks(
        self, tenant_id: str, doc_id: str, chunk_ids: list[str]
    ) -> None:
        self.deleted.append((tenant_id, doc_id, chunk_ids))

    def delete_document_chunks(self, tenant_id: str, doc_id: str) -> None:
        self.deleted_docs.append((tenant_id, doc_id))
        self.rows = [
            row
            for row in self.rows
            if not (row.get("tenant_id") == tenant_id and row.get("doc_id") == doc_id)
        ]
```

Update every `run_task(...)` call to pass `tenant_id="default"`:

```python
summary = await pipeline.run_task("task-1", [_scanned(path, root)], tenant_id="default")
```

Update assertions:

```python
    assert repo.documents[("default", "景点攻略/杭州景点推荐.md")]["status"] == "imported"
    assert repo.documents[("default", "景点攻略/杭州景点推荐.md")]["tenant_id"] == "default"
    assert repo.tasks[-1][2]["status"] == "success"
    assert milvus.rows[0]["tenant_id"] == "default"
    assert repo.deleted == [
        {"tenant_id": "default", "source_path": "景点攻略/删除.md", "doc_id": "doc_gone"}
    ]
    assert milvus.deleted_docs == [("default", "doc_gone")]
```

Also update the stale-chunk assertion in `test_run_task_reimports_when_chunks_changed` (currently `milvus.deleted[-1][0] == doc_id`):

```python
    assert milvus.deleted[-1][0] == "default"
    assert milvus.deleted[-1][1] == doc_id
```

Update the prune boundary regex from `"complete DATA_DIR"` to `"complete data directory"`.

Add a tenant isolation test:

```python
async def test_tenant_imports_are_isolated(tmp_path):
    root = tmp_path / "数据"
    path = _write_document(
        root,
        "景点攻略/杭州景点推荐.md",
        "# 杭州景点推荐\n\n## 西湖\n西湖适合清晨慢游。\n",
    )
    repo = FakeRepo()
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        _settings(root), embedder=FakeEmbedder(), mongo=FakeMongo(), repo=repo, milvus=milvus
    )
    scanned = _scanned(path, root)

    await pipeline.run_task("task-a", [scanned], tenant_id="default")
    await pipeline.run_task("task-b", [scanned], tenant_id="sanya")

    first = repo.documents[("default", "景点攻略/杭州景点推荐.md")]
    second = repo.documents[("sanya", "景点攻略/杭州景点推荐.md")]
    assert first["doc_id"] != second["doc_id"]
    assert first["tenant_id"] == "default"
    assert second["tenant_id"] == "sanya"
    assert [row["tenant_id"] for row in milvus.rows] == ["default", "sanya"]
```

In `tests/unit/test_ingest_worker.py`, update fakes:

```python
class FakeRepo:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str, dict]] = []

    async def update_task(self, tenant_id: str, task_id: str, **fields) -> None:
        self.updates.append((tenant_id, task_id, fields))


class FakePipeline:
    ...
    async def run_task(
        self,
        task_id: str,
        files,
        *,
        tenant_id: str = "default",
        force: bool = False,
        prune: bool = False,
        progress=None,
    ) -> None:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "task_id": task_id,
                "files": list(files),
                "force": force,
                "prune": prune,
            }
        )
        ...
```

Update `submit` calls and assertions:

```python
worker.submit("default", "task-1", [_file()])
...
assert pipeline.calls[0]["tenant_id"] == "default"
assert repo.updates[0][0] == "default"
assert repo.updates[0][1] == "task-1"
assert repo.updates[0][2]["status"] == "running"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_ingestion_pipeline.py tests/unit/test_ingest_worker.py -q`

Expected: FAIL with `TypeError` on `get_document` / `submit` arity.

- [ ] **Step 3: Update `app/ingestion/scanner.py`**

```python
from pathlib import Path

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, TENANTS_DIR

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}


@dataclass(frozen=True)
class ScannedFile:
    path: Path
    relative_path: str
    size: int
    mtime: float
    sha256: str


def scan_directory(
    directory: Path,
    suffixes: set[str] | None = None,
    *,
    exclude_dirs: set[Path] | None = None,
) -> list[ScannedFile]:
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"data directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    allowed = suffixes or SUPPORTED_SUFFIXES
    excluded = tuple(path.resolve() for path in (exclude_dirs or set()))
    files: list[ScannedFile] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if any(path.is_relative_to(item) for item in excluded):
            continue
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.suffix.lower() not in allowed:
            continue
        files.append(build_scanned_file(path, directory))
    return files


def scan_tenant_directory(settings: Settings, tenant_id: str) -> list[ScannedFile]:
    root = settings.tenant_data_path(tenant_id)
    exclude = (
        {settings.data_path / TENANTS_DIR}
        if tenant_id == DEFAULT_TENANT_ID
        else set()
    )
    return scan_directory(root, exclude_dirs=exclude)
```

Keep `build_scanned_file` and `file_sha256` unchanged.

- [ ] **Step 4: Update `app/ingestion/pipeline.py`**

Imports:

```python
from app.core.tenancy import DEFAULT_TENANT_ID
from app.ingestion.metadata import make_doc_id, parse_document_meta
from app.ingestion.scanner import ScannedFile, scan_directory, scan_tenant_directory
```

Replace `run()`:

```python
    async def run(
        self,
        directory: Path | None = None,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        force: bool = False,
        limit: int | None = None,
        prune: bool = False,
        dry_run: bool = False,
        progress=print,
    ) -> IngestionSummary:
        root = (directory or self._settings.tenant_data_path(tenant_id)).resolve()
        if prune:
            self._validate_prune(root, limit, tenant_id)
        try:
            if directory is None:
                files = await asyncio.to_thread(
                    scan_tenant_directory, self._settings, tenant_id
                )
            else:
                files = await asyncio.to_thread(scan_directory, root)
            if limit is not None:
                files = files[:limit]
            if dry_run:
                if not prune:
                    raise ValueError("dry_run requires prune")
                stale = await self._find_stale_documents(files, tenant_id)
                return IngestionSummary(
                    task_id="",
                    total=len(files),
                    stale_files=[doc["source_path"] for doc in stale],
                    dry_run=True,
                )
            if prune and not files:
                raise ValueError(
                    "refusing to prune: scan returned no files (empty scan)"
                )
            task_id = await self._repo.create_task(tenant_id, "scan", len(files))
            return await self.run_task(
                task_id,
                files,
                tenant_id=tenant_id,
                force=force,
                prune=prune,
                progress=progress,
            )
        finally:
            await self._mongo.close()
            await self._milvus.close()

    def _validate_prune(
        self,
        root: Path,
        limit: int | None,
        tenant_id: str,
    ) -> None:
        if limit is not None:
            raise ValueError("prune cannot be combined with limit")
        canonical = self._settings.tenant_data_path(tenant_id).resolve()
        if os.path.normcase(str(root)) != os.path.normcase(str(canonical)):
            raise ValueError("prune is only supported when scanning the tenant data directory")
```

Replace `run_task()`:

```python
    async def run_task(
        self,
        task_id: str,
        files: Sequence[ScannedFile],
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        force: bool = False,
        prune: bool = False,
        progress=print,
    ) -> IngestionSummary:
        started = time.perf_counter()
        summary = IngestionSummary(task_id=task_id, total=len(files))

        try:
            for position, file in enumerate(files, start=1):
                await self._process_file(
                    file,
                    position=position,
                    total=len(files),
                    tenant_id=tenant_id,
                    force=force,
                    progress=progress,
                    summary=summary,
                )
                await self._repo.update_task(
                    tenant_id,
                    task_id,
                    current_file=file.relative_path,
                    succeeded=summary.succeeded,
                    failed=summary.failed,
                    skipped=summary.skipped,
                    errors=summary.errors,
                )
            if prune:
                await self._prune_documents(
                    files, tenant_id=tenant_id, progress=progress, summary=summary
                )
            await asyncio.to_thread(self._milvus.flush)
        finally:
            summary.duration_seconds = round(time.perf_counter() - started, 2)
            status = _task_status(summary)
            await self._repo.update_task(
                tenant_id,
                task_id,
                status=status,
                current_file="",
                succeeded=summary.succeeded,
                failed=summary.failed,
                skipped=summary.skipped,
                pruned=summary.pruned,
                prune_failed=summary.prune_failed,
                errors=summary.errors,
                finished_at=utc_now(),
                duration_seconds=summary.duration_seconds,
            )

        return summary
```

Replace `_find_stale_documents`, `_prune_documents`, `_ensure_complete_scan`:

```python
    async def _find_stale_documents(
        self,
        files: Sequence[ScannedFile],
        tenant_id: str,
    ) -> list[dict]:
        scanned = {file.relative_path for file in files}
        documents = await self._repo.list_documents(tenant_id)
        return [
            document
            for document in documents
            if document.get("status") != "deleted"
            and document.get("source_path")
            and document["source_path"] not in scanned
        ]

    async def _prune_documents(
        self,
        files: Sequence[ScannedFile],
        *,
        tenant_id: str,
        progress,
        summary: IngestionSummary,
    ) -> None:
        await self._ensure_complete_scan(files, tenant_id)
        stale = await self._find_stale_documents(files, tenant_id)
        summary.stale_files = [document["source_path"] for document in stale]
        for document in stale:
            source_path = document["source_path"]
            doc_id = document.get("doc_id") or ""
            try:
                if doc_id:
                    await asyncio.to_thread(
                        self._milvus.delete_document_chunks, tenant_id, doc_id
                    )
                await self._repo.mark_document_deleted(
                    tenant_id,
                    source_path=source_path,
                    doc_id=doc_id,
                )
                summary.pruned += 1
                progress(f"prune {source_path}")
            except Exception as exc:
                summary.prune_failed += 1
                summary.errors.append(
                    {"file": source_path, "error": f"PruneError: {exc}"}
                )
                progress(f"FAILED to prune {source_path}: {exc}")

    async def _ensure_complete_scan(
        self,
        files: Sequence[ScannedFile],
        tenant_id: str,
    ) -> None:
        scanned = {file.relative_path for file in files}
        if not scanned:
            raise ValueError("refusing to prune: scan returned no files (empty scan)")
        expected = await asyncio.to_thread(
            scan_tenant_directory, self._settings, tenant_id
        )
        expected_paths = {file.relative_path for file in expected}
        if scanned != expected_paths:
            raise ValueError(
                "refusing to prune: task files do not match a complete data directory scan"
            )
```

Replace `_process_file()` with tenant-aware calls:

```python
    async def _process_file(
        self,
        file: ScannedFile,
        *,
        position: int,
        total: int,
        tenant_id: str,
        force: bool,
        progress,
        summary: IngestionSummary,
    ) -> None:
        existing = await self._repo.get_document(tenant_id, file.relative_path)
        if (
            existing
            and existing.get("sha256") == file.sha256
            and existing.get("status") == "imported"
            and not force
        ):
            summary.skipped += 1
            progress(f"[{position}/{total}] skip {file.relative_path} (unchanged)")
            return

        doc_id = (existing or {}).get("doc_id") or make_doc_id(
            tenant_id, file.relative_path
        )
        try:
            text = await asyncio.to_thread(load_text, file.path)
            meta = parse_document_meta(text, file.path, file.relative_path)
            sections = parse_sections(text)
            chunks = chunk_sections(sections)
            if not chunks:
                raise ValueError("no chunks parsed from document")

            embedding = await asyncio.to_thread(
                self._embedder.encode,
                [chunk.text for chunk in chunks],
            )
            now = int(time.time())
            rows = [
                {
                    "tenant_id": tenant_id,
                    "chunk_id": f"{doc_id}:{chunk.index}",
                    "doc_id": doc_id,
                    ...
                }
                for index, chunk in enumerate(chunks)
            ]

            await asyncio.to_thread(self._milvus.upsert_chunks, rows)
            await asyncio.to_thread(
                self._milvus.delete_stale_chunks,
                tenant_id,
                doc_id,
                [row["chunk_id"] for row in rows],
            )
            await self._repo.upsert_document(
                tenant_id,
                {
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                    ...
                },
            )
            ...
        except Exception as exc:
            ...
            await self._repo.mark_document_failed(
                tenant_id,
                source_path=file.relative_path,
                doc_id=doc_id,
                source_file=file.path.name,
                error=error["error"],
            )
            ...
```

Do not omit the existing row/document fields: keep `content`, `content_type`, `city`, all name fields, `source_file`, `source_path`, `section_path`, `chunk_index`, `extra`, `updated_at`, `DENSE_FIELD`, `SPARSE_FIELD`, and document `title`, `sha256`, `status`, `size`, `mtime`, `chunk_count`, `imported_at`, `error`.

- [ ] **Step 5: Update `app/workers/ingest_worker.py`**

```python
    def submit(
        self,
        tenant_id: str,
        task_id: str,
        files: Sequence[ScannedFile],
        *,
        force: bool = False,
        prune: bool = False,
    ) -> None:
        self._active.add(task_id)

        def _cleanup(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            self._active.discard(task_id)

        task = asyncio.create_task(
            self._run(tenant_id, task_id, list(files), force=force, prune=prune)
        )
        self._tasks.add(task)
        task.add_done_callback(_cleanup)

    async def _run(
        self,
        tenant_id: str,
        task_id: str,
        files: list[ScannedFile],
        *,
        force: bool,
        prune: bool,
    ) -> None:
        async with self._semaphore:
            await self._repo.update_task(
                tenant_id,
                task_id,
                status="running",
                started_at=utc_now(),
            )
            try:
                await self._pipeline.run_task(
                    task_id,
                    files,
                    tenant_id=tenant_id,
                    force=force,
                    prune=prune,
                    progress=logger.debug,
                )
            except asyncio.CancelledError:
                await self._repo.update_task(
                    tenant_id,
                    task_id,
                    status="cancelled",
                    current_file="",
                    finished_at=utc_now(),
                )
                raise
            except Exception as exc:
                logger.exception("ingestion task {} failed", task_id)
                await self._repo.update_task(
                    tenant_id,
                    task_id,
                    status="failed",
                    current_file="",
                    finished_at=utc_now(),
                    errors=[{"file": "", "error": f"{type(exc).__name__}: {exc}"}],
                )
```

- [ ] **Step 6: Run targeted tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_ingestion_pipeline.py tests/unit/test_ingest_worker.py -q`

Expected: PASS.

- [ ] **Step 7: Commit (only if authorized)**

```bash
git add app/ingestion/scanner.py app/ingestion/pipeline.py app/workers/ingest_worker.py tests/unit/test_ingestion_pipeline.py tests/unit/test_ingest_worker.py
git commit -m "feat: make ingestion pipeline tenant aware"
```

---

### Task 6: 导入路由的租户与角色

**Files:**
- Modify: `app/api/routes/ingest.py:1-273`
- Modify: `app/schemas/ingestion.py:17-37`
- Test: `tests/unit/test_ingest_api.py`

**Interfaces:**
- Consumes: `require_role`, `target_tenant`, `scan_tenant_directory`, tenant-aware `IngestionRepository` / `IngestionWorker`.
- Produces: all ingest routes enforce `viewer`/`editor`; `IngestionTaskSummary.tenant_id`.

- [ ] **Step 1: Write failing tests**

Update `tests/unit/test_ingest_api.py` fakes:

```python
class FakeRepo:
    def __init__(self, tasks: dict[str, dict] | None = None) -> None:
        self.tasks = tasks or {}
        self.created: list[dict] = []
        self.updates: list[tuple[str, str, dict]] = []
        self._counter = len(self.tasks)

    async def update_task(self, tenant_id: str, task_id: str, **fields) -> None:
        self.updates.append((tenant_id, task_id, fields))
        self.tasks[task_id].update(fields)

    async def create_task(
        self, tenant_id: str, task_type: str, total: int, *, status: str = "running"
    ) -> str:
        self._counter += 1
        task_id = f"task-{self._counter}"
        self.tasks[task_id] = _task(
            task_id, status=status, task_type=task_type, total=total, tenant_id=tenant_id
        )
        self.created.append(
            {
                "tenant_id": tenant_id,
                "task_type": task_type,
                "total": total,
                "status": status,
            }
        )
        return task_id

    async def get_task(self, tenant_id: str | None, task_id: str) -> dict | None:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if tenant_id is not None and task.get("tenant_id") != tenant_id:
            return None
        return task

    async def list_tasks(
        self, tenant_id: str | None, *, status=None, limit=20, offset=0
    ) -> tuple[int, list[dict]]:
        items = [
            item
            for item in self.tasks.values()
            if tenant_id is None or item.get("tenant_id") == tenant_id
        ]
        if status:
            items = [item for item in items if item["status"] == status]
        return len(items), items[offset : offset + limit]


class FakeWorker:
    ...
    def submit(
        self, tenant_id: str, task_id: str, files, *, force: bool = False, prune: bool = False
    ) -> None:
        self.submitted.append(
            {
                "tenant_id": tenant_id,
                "task_id": task_id,
                "files": list(files),
                "force": force,
                "prune": prune,
            }
        )
```

Add `tenant_id: str = "default"` to `_task()` and include `"tenant_id": tenant_id`.

`_app()` must set auth state:

```python
from app.core.tenancy import TenantRegistry


def _app(tmp_path, repo: FakeRepo, worker: FakeWorker) -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        max_upload_mb=1,
        allowed_upload_suffixes=".md,.markdown,.txt,.pdf,.docx",
    )
    app.state.tenants = TenantRegistry()
    app.state.ingest_repo = repo
    app.state.ingest_worker = worker
    ...
```

Change monkeypatches from `scan_directory` to `scan_tenant_directory`:

```python
monkeypatch.setattr(ingest, "scan_tenant_directory", lambda settings, tenant_id: [scanned])
```

Update expected created entries:

```python
assert repo.created[-1] == {
    "tenant_id": "default",
    "task_type": "scan",
    "total": 1,
    "status": "queued",
}
```

Update update assertions:

```python
assert repo.updates[0][0] == "default"
assert repo.updates[0][1] == "task-1"
assert repo.updates[0][2]["status"] == "failed"
```

Add role and tenant tests:

```python
from app.core.config import Settings
from app.core.tenancy import TenantCredential


def _tenant_app(tmp_path, repo, worker, *, tenants, admin_token=""):
    app = _app(tmp_path, repo, worker)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        max_upload_mb=1,
        tenants=tenants,
        admin_token=admin_token,
    )
    app.state.settings = settings
    app.state.tenants = TenantRegistry(settings.tenants, settings.admin_token)
    return app


async def test_viewer_cannot_scan(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "scan_tenant_directory", lambda settings, tenant_id: [])
    viewer = TenantCredential(
        tenant_id="sanya", tenant_name="三亚运营", api_key="v" * 32, role="viewer"
    )
    app = _tenant_app(tmp_path, FakeRepo(), FakeWorker(), tenants=[viewer])

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/scan", json={}, headers={"X-API-Key": "v" * 32}
        )

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"


async def test_editor_scan_and_task_are_scoped(monkeypatch, tmp_path):
    scanned = ScannedFile(
        path=tmp_path / "a.md", relative_path="a.md", size=1, mtime=0.0, sha256="abc"
    )
    monkeypatch.setattr(
        ingest, "scan_tenant_directory", lambda settings, tenant_id: [scanned]
    )
    editor = TenantCredential(
        tenant_id="sanya", tenant_name="三亚运营", api_key="e" * 32, role="editor"
    )
    repo = FakeRepo()
    worker = FakeWorker()
    app = _tenant_app(tmp_path, repo, worker, tenants=[editor])

    async with _client(app) as client:
        created = await client.post(
            "/api/v1/ingest/scan", json={}, headers={"X-API-Key": "e" * 32}
        )
        listing = await client.get(
            "/api/v1/ingest/tasks", headers={"X-API-Key": "e" * 32}
        )

    assert created.status_code == 200
    assert repo.created[-1]["tenant_id"] == "sanya"
    assert worker.submitted[0]["tenant_id"] == "sanya"
    assert listing.json()["total"] == 1
    assert listing.json()["tasks"][0]["tenant_id"] == "sanya"


async def test_global_admin_lists_all_tasks_and_filters(tmp_path):
    repo = FakeRepo(
        {
            "task-1": _task("task-1", tenant_id="default"),
            "task-2": _task("task-2", tenant_id="sanya"),
        }
    )
    admin_token = "admin-secret-token-123456"
    editor = TenantCredential(
        tenant_id="default", tenant_name="默认租户", api_key="e" * 32, role="editor"
    )
    viewer = TenantCredential(
        tenant_id="sanya", tenant_name="三亚运营", api_key="v" * 32, role="viewer"
    )
    app = _tenant_app(
        tmp_path,
        repo,
        FakeWorker(),
        tenants=[editor, viewer],
        admin_token=admin_token,
    )

    async with _client(app) as client:
        all_tasks = await client.get(
            "/api/v1/ingest/tasks", headers={"X-Admin-Token": admin_token}
        )
        scoped = await client.get(
            "/api/v1/ingest/tasks",
            headers={"X-Admin-Token": admin_token, "X-Tenant-Id": "sanya"},
        )

    assert all_tasks.json()["total"] == 2
    assert scoped.json()["total"] == 1
    assert scoped.json()["tasks"][0]["tenant_id"] == "sanya"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_ingest_api.py -q`

Expected: FAIL with `TypeError` on repo/worker arity or 401/403 assertions.

- [ ] **Step 3: Update schemas**

In `app/schemas/ingestion.py`, add to `IngestionTaskSummary` after `task_id`:

```python
    tenant_id: str = ""
```

- [ ] **Step 4: Update `app/api/routes/ingest.py`**

Imports:

```python
from app.api.deps import (
    get_ingest_repo,
    get_ingest_worker,
    get_settings_dep,
    require_role,
    target_tenant,
)
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.ingestion.scanner import (
    build_scanned_file,
    scan_tenant_directory,
)
```

Scan route:

```python
@router.post("/ingest/scan", response_model=IngestStartResponse)
async def scan(
    payload: IngestScanRequest,
    repo: Annotated[IngestionRepository, Depends(get_ingest_repo)],
    worker: Annotated[IngestionWorker, Depends(get_ingest_worker)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    ctx: Annotated[TenantContext, Depends(require_role("editor"))],
) -> IngestStartResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    try:
        files = await asyncio.to_thread(
            scan_tenant_directory, settings, tenant_id
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise AppError(str(exc), status_code=400, code="DATA_DIR_UNAVAILABLE") from exc
    except OSError as exc:
        raise AppError(
            f"无法扫描数据目录: {exc}",
            status_code=400,
            code="DATA_DIR_UNAVAILABLE",
        ) from exc

    if payload.prune:
        if payload.limit is not None:
            raise AppError(
                "prune 不能与 limit 同时使用",
                status_code=400,
                code="INVALID_PRUNE",
            )
        if not files:
            raise AppError(
                "扫描结果为空，拒绝执行 prune",
                status_code=400,
                code="PRUNE_EMPTY_SCAN",
            )
    if payload.limit is not None:
        files = files[: payload.limit]
    return await _queue(
        repo,
        worker,
        tenant_id,
        "scan",
        files,
        force=payload.force,
        prune=payload.prune,
    )
```

Upload route: add `ctx` dependency and tenant paths:

```python
    ctx: Annotated[TenantContext, Depends(require_role("editor"))],
) -> IngestStartResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    ...
    upload_dir = settings.tenant_upload_path(tenant_id)
    ...
    scanned = await asyncio.to_thread(
        build_scanned_file,
        target,
        settings.tenant_data_path(tenant_id),
    )
    ...
    return await _queue(repo, worker, tenant_id, "upload", [scanned])
```

List route:

```python
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> IngestionTaskListResponse:
    tenant_id = target_tenant(ctx, allow_all=True)
    total, documents = await repo.list_tasks(
        tenant_id, status=status, limit=limit, offset=offset
    )
```

Detail route:

```python
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> IngestionTaskSummary:
    tenant_id = target_tenant(ctx, allow_all=True)
    document = await repo.get_task(tenant_id, task_id)
```

Retry route:

```python
    ctx: Annotated[TenantContext, Depends(require_role("editor"))],
) -> IngestStartResponse:
    scope = target_tenant(ctx, allow_all=True)
    document = await repo.get_task(scope, task_id)
    if document is None:
        raise NotFoundError("task not found")

    task_tenant = str(document.get("tenant_id") or DEFAULT_TENANT_ID)
    if not ctx.is_global_admin and task_tenant != ctx.tenant_id:
        raise NotFoundError("task not found")

    ...
        await repo.update_task(
            task_tenant,
            task_id,
            status="failed",
            ...
        )
    ...
    root = settings.tenant_data_path(task_tenant).resolve()
    ...
        files = await asyncio.to_thread(
            scan_tenant_directory, settings, task_tenant
        )
    ...
    return await _queue(
        repo, worker, task_tenant, "retry", files, force=bool(failed_files)
    )
```

Queue helper:

```python
async def _queue(
    repo: IngestionRepository,
    worker: IngestionWorker,
    tenant_id: str,
    task_type: str,
    files,
    *,
    force: bool = False,
    prune: bool = False,
) -> IngestStartResponse:
    task_id = await repo.create_task(
        tenant_id, task_type, len(files), status="queued"
    )
    worker.submit(tenant_id, task_id, files, force=force, prune=prune)
    return IngestStartResponse(task_id=task_id, status="queued")
```

`_task_summary` adds:

```python
        tenant_id=document.get("tenant_id") or DEFAULT_TENANT_ID,
```

- [ ] **Step 5: Run targeted tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_ingest_api.py -q`

Expected: PASS.

- [ ] **Step 6: Commit (only if authorized)**

```bash
git add app/api/routes/ingest.py app/schemas/ingestion.py tests/unit/test_ingest_api.py
git commit -m "feat: scope ingest routes by tenant and role"
```

---

### Task 7: 检索引擎的租户参数

**Files:**
- Modify: `app/retrieval/service.py:28-92`
- Modify: `app/retrieval/three_way.py:37-70, 140-170`
- Test: `tests/unit/test_three_way.py`

**Interfaces:**
- Produces: `RetrievalService.search(..., tenant_id: str, ...)`.
- Produces: `RetrievalService.search_all(..., tenant_id: str, ...)`.
- Produces: `ThreeWayRetriever.search(..., tenant_id: str, ...)`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_three_way.py`, update `FakeRetrieval.search`:

```python
    def search(
        self,
        text: str,
        *,
        mode: str,
        top_k: int,
        rerank: bool,
        city,
        content_type,
        tenant_id: str,
    ):
        self.search_calls.append(
            {"text": text, "mode": mode, "top_k": top_k, "tenant_id": tenant_id}
        )
        ...
```

Update all `retriever.search(...)` calls to include `tenant_id="default"`. Add an assertion:

```python
    assert retrieval.search_calls[0]["tenant_id"] == "default"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_three_way.py -q`

Expected: FAIL with `TypeError: search() missing 1 required keyword-only argument: 'tenant_id'`.

- [ ] **Step 3: Update `app/retrieval/service.py`**

Add `tenant_id: str` after the `*` in `search()` and `search_all()`:

```python
    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        mode: str = "hybrid",
        ...
    ) -> list[dict]:
        ...
        filter_expr = build_filter(
            tenant_id=tenant_id,
            city=city,
            ...
        )
```

```python
    def search_all(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        rerank: bool = False,
        **filters,
    ) -> dict[str, list[dict]]:
        ...
        filter_expr = build_filter(tenant_id=tenant_id, **filters)
```

- [ ] **Step 4: Update `app/retrieval/three_way.py`**

`search()` signature:

```python
    async def search(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        city: str | None = None,
        content_type: str | None = None,
        on_stage: StageCallback | None = None,
    ) -> RetrievalOutcome:
```

Pass tenant to both local searches:

```python
        hybrid_task = asyncio.create_task(
            self._search_local(
                query, "hybrid", candidate_k, tenant_id, city, content_type
            )
        )
        ...
            hyde_hits, hyde_search_info = await self._search_local(
                hyde_text, "dense", candidate_k, tenant_id, city, content_type
            )
```

`_search_local()`:

```python
    async def _search_local(
        self,
        text: str,
        mode: str,
        top_k: int,
        tenant_id: str,
        city: str | None,
        content_type: str | None,
    ) -> tuple[list[dict], dict]:
        ...
                self._retrieval.search,
                text,
                tenant_id=tenant_id,
                mode=mode,
                ...
```

- [ ] **Step 5: Run tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_three_way.py tests/unit/test_filters.py -q`

Expected: PASS.

- [ ] **Step 6: Commit (only if authorized)**

```bash
git add app/retrieval/service.py app/retrieval/three_way.py tests/unit/test_three_way.py
git commit -m "feat: require tenant in retrieval paths"
```

---

### Task 8: 会话、问答与 Chat 状态

**Files:**
- Modify: `app/chat/state.py:4-27`
- Modify: `app/chat/memory.py:24-174`
- Modify: `app/chat/nodes.py:41-187`
- Modify: `app/chat/service.py:36-205`
- Modify: `app/api/routes/chat.py`
- Modify: `app/api/routes/sessions.py`
- Test: `tests/unit/test_chat_nodes.py`

**Interfaces:**
- Produces: `ChatState.tenant_id: str`.
- Produces: all `ChatMemory` methods take `tenant_id` as first positional argument (keyword-only methods also require it).
- Produces: `ChatService.chat(request, tenant_id)` / `ChatService.stream(request, tenant_id)`.
- Produces: chat and session routes require `viewer`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_chat_nodes.py`, update fakes:

```python
class FakeThreeWay:
    async def search(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int,
        city,
        content_type,
        on_stage=None,
    ) -> RetrievalOutcome:
        self.calls.append(
            {
                "query": query,
                "tenant_id": tenant_id,
                "top_k": top_k,
                "city": city,
                "content_type": content_type,
            }
        )
        ...
```

Add `tenant_id` to the state and expected calls:

```python
        {
            "query": "西湖怎么玩",
            "tenant_id": "default",
            "session_id": "s1",
            ...
        }
```

```python
    assert three_way.calls == [
        {
            "query": "杭州西湖怎么玩",
            "tenant_id": "default",
            "top_k": 5,
            "city": "杭州",
            "content_type": None,
        }
    ]
```

Add an assertion that memory receives the tenant:

```python
    assert memory.messages[0]["tenant_id"] == "default"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_chat_nodes.py -q`

Expected: FAIL with `KeyError: 'tenant_id'` or fake signature mismatch.

- [ ] **Step 3: Update `app/chat/state.py`**

Add after `user_id`:

```python
    tenant_id: str
```

- [ ] **Step 4: Update `app/chat/memory.py`**

Every method gains the tenant and scopes every query/insert:

```python
    async def get_or_create_session(
        self,
        tenant_id: str,
        *,
        session_id: str | None,
        user_id: str | None,
        title: str,
    ) -> dict:
        if session_id:
            document = await self._db[CHAT_SESSIONS].find_one(
                {"_id": parse_object_id(session_id), "tenant_id": tenant_id}
            )
            ...
        document = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            ...
        }

    async def get_session(self, tenant_id: str, session_id: str) -> dict:
        document = await self._db[CHAT_SESSIONS].find_one(
            {"_id": parse_object_id(session_id), "tenant_id": tenant_id}
        )
        ...

    async def load_history(
        self,
        tenant_id: str,
        session_id: str,
        limit: int,
        *,
        exclude_message_id: str | None = None,
    ) -> list[dict]:
        query: dict = {"tenant_id": tenant_id, "session_id": session_id}
        ...

    async def add_message(self, *, tenant_id: str, session_id: str, ...) -> str:
        document = {
            "tenant_id": tenant_id,
            ...
        }
        await self._db[CHAT_SESSIONS].update_one(
            {"_id": parse_object_id(session_id), "tenant_id": tenant_id},
            ...
        )

    async def update_message(self, tenant_id: str, message_id: str, **fields) -> None:
        await self._db[CHAT_MESSAGES].update_one(
            {"_id": parse_object_id(message_id, resource="message"), "tenant_id": tenant_id},
            {"$set": updates},
        )

    async def message_status(self, tenant_id: str, message_id: str) -> str | None:
        document = await self._db[CHAT_MESSAGES].find_one(
            {"_id": parse_object_id(message_id, resource="message"), "tenant_id": tenant_id},
            {"status": 1},
        )
        ...

    async def list_sessions(
        self,
        *,
        tenant_id: str,
        user_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[int, list[dict]]:
        query = {"tenant_id": tenant_id}
        if user_id is not None:
            query["user_id"] = user_id
        ...

    async def list_messages(self, tenant_id: str, session_id: str, limit: int = 200) -> list[dict]:
        cursor = (
            self._db[CHAT_MESSAGES]
            .find({"tenant_id": tenant_id, "session_id": session_id})
            ...
        )

    async def delete_session(self, tenant_id: str, session_id: str) -> bool:
        session_object_id = parse_object_id(session_id)
        session = await self._db[CHAT_SESSIONS].find_one_and_delete(
            {"_id": session_object_id, "tenant_id": tenant_id},
            ...
        )
        await self._db[CHAT_MESSAGES].delete_many(
            {"tenant_id": tenant_id, "session_id": session_id}
        )
        return session is not None

    async def refresh_summary(self, tenant_id: str, session_id: str, summary: str) -> None:
        await self._db[CHAT_SESSIONS].update_one(
            {"_id": parse_object_id(session_id), "tenant_id": tenant_id},
            {"$set": {"summary": summary, "updated_at": utc_now()}},
        )
```

- [ ] **Step 5: Update `app/chat/nodes.py`**

`load_history`:

```python
        tenant_id = state["tenant_id"]
        session = await self._memory.get_session(tenant_id, session_id)
        history = await self._memory.load_history(
            tenant_id,
            session_id,
            self._settings.chat_history_max_messages,
            exclude_message_id=state.get("user_message_id"),
        )
```

`retrieve`:

```python
        outcome = await self._three_way.search(
            state.get("rewritten_query") or state["query"],
            tenant_id=state["tenant_id"],
            top_k=state.get("top_k", self._settings.rerank_top_n),
            ...
        )
        ...
        message_id = await self._memory.add_message(
            tenant_id=state["tenant_id"],
            session_id=state["session_id"],
            ...
        )
```

`persist`:

```python
            await self._memory.update_message(
                state["tenant_id"],
                message_id,
                content=answer,
                ...
            )
```

- [ ] **Step 6: Update `app/chat/service.py`**

```python
    async def chat(self, request: ChatRequest, tenant_id: str) -> ChatResponse:
        ...
        session_id, user_message_id = await self._prepare(request, tenant_id)
        ...
                self._initial_state(request, session_id, user_message_id, tenant_id),
        ...
        self._schedule_summary(tenant_id, session_id)

    async def stream(self, request: ChatRequest, tenant_id: str) -> AsyncIterator:
        ...
        session_id, user_message_id = await self._prepare(request, tenant_id)
        ...
        state = self._initial_state(request, session_id, user_message_id, tenant_id)
        ...
                if not runtime.persisted:
                    await self._mark_incomplete(
                        tenant_id, runtime, "partial" if runtime.answer else "failed"
                    )
                if completed:
                    self._schedule_summary(tenant_id, session_id)

    async def _prepare(
        self, request: ChatRequest, tenant_id: str
    ) -> tuple[str, str]:
        session = await self._memory.get_or_create_session(
            tenant_id,
            session_id=request.session_id,
            user_id=request.user_id,
            title=request.query,
        )
        ...
        user_message_id = await self._memory.add_message(
            tenant_id=tenant_id,
            session_id=session_id,
            role="user",
            content=request.query,
        )
        return session_id, user_message_id

    def _initial_state(
        self,
        request: ChatRequest,
        session_id: str,
        user_message_id: str,
        tenant_id: str,
    ) -> ChatState:
        return {
            "tenant_id": tenant_id,
            ...
        }

    async def _mark_incomplete(
        self, tenant_id: str, runtime: ChatRuntime, status: str
    ) -> None:
        ...
            await self._memory.update_message(
                tenant_id,
                runtime.assistant_message_id,
                content=runtime.answer,
                status=status,
            )

    def _schedule_summary(self, tenant_id: str, session_id: str) -> None:
        ...
        task = asyncio.create_task(self._refresh_summary(tenant_id, session_id))
        ...

    async def _refresh_summary(self, tenant_id: str, session_id: str) -> None:
        session = await self._memory.get_session(tenant_id, session_id)
        ...
        history = await self._memory.load_history(tenant_id, session_id, limit=40)
        ...
            await self._memory.refresh_summary(tenant_id, session_id, summary[:800])
```

- [ ] **Step 7: Update chat and session routes**

`app/api/routes/chat.py`:

```python
from app.api.deps import get_chat_service, require_role, target_tenant
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> ChatResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    return await service.chat(payload, tenant_id)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> EventSourceResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    events = await service.stream(payload, tenant_id)
    return EventSourceResponse(events, headers={"X-Accel-Buffering": "no"})
```

`target_tenant(ctx)` returns `str | None`; the `or DEFAULT_TENANT_ID` fallback satisfies `ChatService`'s `str` parameter for the global-admin-without-tenant case.

`app/api/routes/sessions.py`: add `ctx` to all three routes and `tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID`; pass it to every memory call:

```python
    total, sessions = await memory.list_sessions(
        tenant_id=tenant_id, user_id=user_id, limit=limit, offset=offset
    )
```

```python
    session = await memory.get_session(tenant_id, session_id)
    messages = await memory.list_messages(tenant_id, session_id)
```

```python
    deleted = await memory.delete_session(tenant_id, session_id)
```

- [ ] **Step 8: Run targeted tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_chat_nodes.py tests/unit/test_auth_api.py -q`

Expected: PASS.

- [ ] **Step 9: Commit (only if authorized)**

```bash
git add app/chat/state.py app/chat/memory.py app/chat/nodes.py app/chat/service.py app/api/routes/chat.py app/api/routes/sessions.py tests/unit/test_chat_nodes.py
git commit -m "feat: isolate chat sessions and messages by tenant"
```

---

### Task 9: 规划、检索与目录接口

**Files:**
- Modify: `app/planner/service.py:64-259`
- Modify: `app/api/routes/search.py`
- Modify: `app/api/routes/planning.py`
- Test: `tests/unit/test_planner_service.py`
- Test: `tests/unit/test_planning_api.py`

**Interfaces:**
- Produces: `PlannerService.plan(request, tenant_id)` / `recommend(request, tenant_id)`.
- Produces: `/search`, `/catalog`, `/itineraries`, `/recommendations` require `viewer` and pass the target tenant.
- `catalog` filters MongoDB by `tenant_id`.

- [ ] **Step 1: Write failing tests**

In `tests/unit/test_planner_service.py`, add `tenant_id: str` to `FakeRetrieval.search` and record it:

```python
    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        mode: str = "hybrid",
        ...
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "tenant_id": tenant_id,
                ...
            }
        )
```

Update every service call:

```python
response = await service.plan(ItineraryRequest(destination="三亚", days=2), "default")
```

```python
response = await service.recommend(
    RecommendationRequest(destination="三亚", category="美食", top_k=2), "default"
)
```

Add an assertion:

```python
    assert all(call["tenant_id"] == "default" for call in retrieval.calls)
```

In `tests/unit/test_planning_api.py`, update `FakePlanner`:

```python
    async def plan(self, request, tenant_id: str):
        self.requests.append(
            {"type": "plan", "payload": request, "tenant_id": tenant_id}
        )
        ...

    async def recommend(self, request, tenant_id: str):
        self.requests.append(
            {"type": "recommend", "payload": request, "tenant_id": tenant_id}
        )
        ...
```

`_app()` sets auth state:

```python
from app.core.config import Settings
from app.core.tenancy import TenantRegistry


def _app(planner: FakePlanner) -> FastAPI:
    app = FastAPI()
    app.state.settings = Settings(_env_file=None)
    app.state.tenants = TenantRegistry()
    app.state.planner = planner
    ...
```

Add assertions:

```python
    assert planner.requests[0]["tenant_id"] == "default"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_planner_service.py tests/unit/test_planning_api.py -q`

Expected: FAIL with `TypeError` on `plan()` / `recommend()` arity.

- [ ] **Step 3: Update `app/planner/service.py`**

```python
    async def plan(self, request: ItineraryRequest, tenant_id: str) -> ItineraryResponse:
        ...
        citations, retrieval_info = await self._collect_context(
            request.destination,
            city,
            ITINERARY_CONTENT_TYPES,
            tenant_id=tenant_id,
            per_type=self._settings.planner_context_per_type,
        )
        ...

    async def recommend(
        self, request: RecommendationRequest, tenant_id: str
    ) -> RecommendationResponse:
        ...
        citations, retrieval_info = await self._collect_context(
            request.destination,
            city,
            content_types,
            tenant_id=tenant_id,
            per_type=request.top_k,
        )
        ...

    async def _collect_context(
        self,
        destination: str,
        city: str | None,
        content_types: tuple[str, ...],
        *,
        tenant_id: str,
        per_type: int,
    ) -> tuple[list[dict], dict]:
        results = await asyncio.gather(
            *(
                self._search_content_type(
                    destination, city, content_type, tenant_id=tenant_id, per_type=per_type
                )
                for content_type in content_types
            )
        )
        ...

    async def _search_content_type(
        self,
        destination: str,
        city: str | None,
        content_type: str,
        *,
        tenant_id: str,
        per_type: int,
    ) -> tuple[str, list[dict], dict]:
        ...
            hits = await asyncio.to_thread(
                self._retrieval.search,
                query,
                tenant_id=tenant_id,
                mode="hybrid",
                top_k=per_type,
                city=city,
                content_type=content_type,
            )
```

- [ ] **Step 4: Update `app/api/routes/planning.py`**

```python
from app.api.deps import get_planner, require_role, target_tenant
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext


@router.post("/itineraries", response_model=ItineraryResponse)
async def plan_itinerary(
    payload: ItineraryRequest,
    service: Annotated[PlannerService, Depends(get_planner)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> ItineraryResponse:
    return await service.plan(payload, target_tenant(ctx) or DEFAULT_TENANT_ID)


@router.post("/recommendations", response_model=RecommendationResponse)
async def recommend(
    payload: RecommendationRequest,
    service: Annotated[PlannerService, Depends(get_planner)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> RecommendationResponse:
    return await service.recommend(
        payload, target_tenant(ctx) or DEFAULT_TENANT_ID
    )
```

- [ ] **Step 5: Update `app/api/routes/search.py`**

```python
from app.api.deps import (
    get_mongo,
    get_retrieval,
    get_settings_dep,
    require_role,
    target_tenant,
)
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    retrieval: Annotated[RetrievalService, Depends(get_retrieval)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> SearchResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    ...
            retrieval.search,
            payload.query,
            tenant_id=tenant_id,
            mode=payload.mode,
            ...
```

Catalog:

```python
@router.get("/catalog", response_model=CatalogResponse)
async def catalog(
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> CatalogResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    pipeline = [
        {"$match": {"tenant_id": tenant_id, "status": "imported"}},
        ...
    ]
```

- [ ] **Step 6: Run targeted tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_planner_service.py tests/unit/test_planning_api.py -q`

Expected: PASS.

- [ ] **Step 7: Run the full unit suite**

Run: `.venv/Scripts/python -m pytest -q`

Expected: PASS with 118 + new tests. If failures remain, they are missed tenant call sites; fix before continuing.

- [ ] **Step 8: Commit (only if authorized)**

```bash
git add app/planner/service.py app/api/routes/search.py app/api/routes/planning.py tests/unit/test_planner_service.py tests/unit/test_planning_api.py
git commit -m "feat: scope planning and search by tenant"
```

---

### Task 10: 管理 API 与租户清单

**Files:**
- Modify: `app/schemas/admin.py:27-68`
- Modify: `app/api/routes/admin.py:1-194`
- Test: `tests/unit/test_admin_api.py`

**Interfaces:**
- Consumes: `require_admin`, `get_tenant_registry`, `TenantContext`, `TenantMeta`.
- Produces: `GET /api/v1/admin/tenants` returning `{scope, tenant_id, tenants}`.
- Produces: `AdminDocumentSummary.tenant_id`.
- Overview/documents/detail accept optional `tenant_id` for global admin.

- [ ] **Step 1: Write failing tests**

Update `_app()` in `tests/unit/test_admin_api.py` to install `app.state.tenants = TenantRegistry()` and accept optional tenant settings:

```python
from app.core.tenancy import TenantRegistry, TenantCredential


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
    ...
```

Add `"tenant_id": "default"` to `_document()` base, and update the fake Milvus signatures:

```python
    def count_chunks(self, tenant_id: str | None = None) -> int:
        if self.error:
            raise self.error
        if tenant_id is None:
            return len(self.chunks)
        return sum(1 for chunk in self.chunks if chunk.get("tenant_id") == tenant_id)

    def list_document_chunks(
        self, tenant_id: str | None, doc_id: str, limit: int = 200
    ) -> list[dict]:
        self.queries.append((tenant_id, doc_id, limit))
        if self.error:
            raise self.error
        return [
            chunk
            for chunk in self.chunks
            if chunk.get("doc_id") == doc_id
            and (tenant_id is None or chunk.get("tenant_id") == tenant_id)
        ][:limit]
```

Add `"tenant_id": "default"` to `CHUNKS`. Update the detail assertion:

```python
    assert milvus.queries == [("default", "d1", admin.MAX_CHUNK_PREVIEW)]
```

Add two tests:

```python
async def test_admin_tenants_lists_default_in_open_mode(tmp_path):
    app = _app(tmp_path)

    async with _client(app) as client:
        response = await client.get("/api/v1/admin/tenants")

    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "tenant"
    assert body["tenant_id"] == "default"
    assert body["tenants"] == [{"tenant_id": "default", "tenant_name": "默认租户"}]


async def test_global_admin_can_filter_documents_and_get_cross_tenant_tasks(tmp_path):
    sanya_doc = _document(
        doc_id="d5",
        source_path="sanya-only.md",
        source_file="sanya-only.md",
        tenant_id="sanya",
        updated_at=datetime(2026, 10, 24, 11, 0, tzinfo=UTC),
    )
    mongo = FakeMongo(DOCUMENTS + [sanya_doc], TASKS)
    viewer = TenantCredential(
        tenant_id="sanya", tenant_name="三亚运营", api_key="v" * 32, role="viewer"
    )
    app = _app(
        tmp_path,
        mongo=mongo,
        admin_token="admin-secret-token-123456",
        tenants=[viewer],
    )
    headers = {"X-Admin-Token": "admin-secret-token-123456"}

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
    assert {item["tenant_id"] for item in tenants.json()["tenants"]} == {
        "default",
        "sanya",
    }
    assert all("api_key" not in item for item in tenants.json()["tenants"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/unit/test_admin_api.py -q`

Expected: FAIL (`/admin/tenants` 404, fake signature mismatch, or missing `tenant_id`).

- [ ] **Step 3: Update `app/schemas/admin.py`**

Add to `AdminDocumentSummary` after `doc_id`:

```python
    tenant_id: str = ""
```

Add at the end:

```python
class AdminTenant(BaseModel):
    tenant_id: str
    tenant_name: str


class AdminTenantListResponse(BaseModel):
    scope: str
    tenant_id: str | None = None
    tenants: list[AdminTenant] = Field(default_factory=list)
```

- [ ] **Step 4: Update `app/api/routes/admin.py`**

Imports:

```python
from app.api.deps import (
    get_milvus,
    get_mongo,
    get_settings_dep,
    get_tenant_registry,
    require_admin,
)
from app.core.tenancy import (
    DEFAULT_TENANT_ID,
    TenantContext,
    TenantRegistry,
)
from app.schemas.admin import (
    ...
    AdminTenant,
    AdminTenantListResponse,
)
```

Router line becomes:

```python
router = APIRouter(tags=["admin"])
```

Add:

```python
@router.get("/admin/tenants", response_model=AdminTenantListResponse)
async def list_tenants(
    ctx: Annotated[TenantContext, Depends(require_admin)],
    registry: Annotated[TenantRegistry, Depends(get_tenant_registry)],
) -> AdminTenantListResponse:
    if ctx.is_global_admin:
        return AdminTenantListResponse(
            scope="all",
            tenant_id=None,
            tenants=[
                AdminTenant(tenant_id=meta.tenant_id, tenant_name=meta.tenant_name)
                for meta in registry.list_tenants()
            ],
        )
    tenant_id = ctx.tenant_id or DEFAULT_TENANT_ID
    meta = registry.get_tenant(tenant_id)
    return AdminTenantListResponse(
        scope="tenant",
        tenant_id=tenant_id,
        tenants=[
            AdminTenant(
                tenant_id=tenant_id,
                tenant_name=meta.tenant_name if meta else ctx.tenant_name,
            )
        ],
    )
```

Add helper:

```python
def _resolve_admin_tenant(
    ctx: TenantContext,
    requested: str | None,
    registry: TenantRegistry,
) -> str | None:
    if ctx.open_mode:
        return DEFAULT_TENANT_ID
    if ctx.is_global_admin:
        if requested and registry.get_tenant(requested) is None:
            raise AppError("租户不存在", status_code=404, code="TENANT_NOT_FOUND")
        return requested or None
    if requested and requested != ctx.tenant_id:
        raise AppError("无权访问其他租户", status_code=403, code="TENANT_MISMATCH")
    return ctx.tenant_id
```

Overview route:

```python
@router.get("/admin/overview", response_model=AdminOverview)
async def overview(
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
    milvus: Annotated[MilvusRepository, Depends(get_milvus)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    ctx: Annotated[TenantContext, Depends(require_admin)],
    registry: Annotated[TenantRegistry, Depends(get_tenant_registry)],
    tenant_id: str | None = Query(default=None, max_length=64),
) -> AdminOverview:
    tenant_scope = _resolve_admin_tenant(ctx, tenant_id, registry)
    db = mongo.db
    documents_by_status, total_documents = await _status_counts(
        db[DOCUMENTS], tenant_scope
    )
    tasks_by_status, total_tasks = await _status_counts(
        db[INGESTION_TASKS], tenant_scope
    )
    cities, content_types = await asyncio.gather(
        _field_counts(db[DOCUMENTS], "city", tenant_scope),
        _field_counts(db[DOCUMENTS], "content_type", tenant_scope),
    )

    chunks: int | None = None
    chunk_error = ""
    try:
        chunks = await asyncio.to_thread(milvus.count_chunks, tenant_scope)
    ...
```

Documents list:

```python
    ctx: Annotated[TenantContext, Depends(require_admin)],
    registry: Annotated[TenantRegistry, Depends(get_tenant_registry)],
    tenant_id: str | None = Query(default=None, max_length=64),
) -> AdminDocumentListResponse:
    tenant_scope = _resolve_admin_tenant(ctx, tenant_id, registry)
    query = _document_query(
        tenant_id=tenant_scope,
        status=status,
        ...
    )
```

Detail route:

```python
async def get_document(
    doc_id: str,
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
    milvus: Annotated[MilvusRepository, Depends(get_milvus)],
    ctx: Annotated[TenantContext, Depends(require_admin)],
    registry: Annotated[TenantRegistry, Depends(get_tenant_registry)],
    tenant_id: str | None = Query(default=None, max_length=64),
) -> AdminDocumentDetail:
    ...
    tenant_scope = _resolve_admin_tenant(ctx, tenant_id, registry)
    query: dict = {"doc_id": doc_id}
    if tenant_scope is not None:
        query["tenant_id"] = tenant_scope
    document = await mongo.db[DOCUMENTS].find_one(query, {"_id": 0})
    if document is None:
        raise NotFoundError("document not found")

    document_tenant = document.get("tenant_id") or DEFAULT_TENANT_ID
    ...
        rows = await asyncio.to_thread(
            milvus.list_document_chunks,
            document_tenant,
            doc_id,
            MAX_CHUNK_PREVIEW,
        )
```

Helpers:

```python
async def _status_counts(collection, tenant_id: str | None) -> tuple[dict[str, int], int]:
    pipeline: list[dict] = []
    if tenant_id is not None:
        pipeline.append({"$match": {"tenant_id": tenant_id}})
    pipeline.append({"$group": {"_id": "$status", "count": {"$sum": 1}}})
    cursor = await collection.aggregate(pipeline)
    ...


async def _field_counts(
    collection, field: str, tenant_id: str | None
) -> dict[str, int]:
    match: dict = {"status": "imported"}
    if tenant_id is not None:
        match["tenant_id"] = tenant_id
    cursor = await collection.aggregate(
        [
            {"$match": match},
            {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
        ]
    )
    ...


def _document_query(
    *,
    tenant_id: str | None,
    status: str | None,
    city: str | None,
    content_type: str | None,
    keyword: str | None,
) -> dict:
    query: dict = {}
    if tenant_id is not None:
        query["tenant_id"] = tenant_id
    ...
```

- [ ] **Step 5: Run targeted tests**

Run: `.venv/Scripts/python -m pytest tests/unit/test_admin_api.py -q`

Expected: PASS.

- [ ] **Step 6: Commit (only if authorized)**

```bash
git add app/schemas/admin.py app/api/routes/admin.py tests/unit/test_admin_api.py
git commit -m "feat: add tenant-aware admin API"
```

---

### Task 11: 管理后台前端

**Files:**
- Modify: `app/static/admin/index.html`
- Modify: `app/static/admin/admin.css`
- Modify: `app/static/admin/admin.js`

**Interfaces:**
- Consumes: `GET /api/v1/admin/tenants`, `X-Admin-Token` / `X-API-Key`, `X-Tenant-Id`.
- Produces: tenant selector, tenant columns, global-admin safeguards.

- [ ] **Step 1: Update `index.html`**

Credential block:

```html
      <label class="field-label" for="admin-token">访问凭证</label>
      <div class="token-row">
        <input id="admin-token" type="password" placeholder="管理令牌或 API Key" autocomplete="off">
        <button id="save-token" class="ghost small" type="button">保存</button>
      </div>
```

Topbar:

```html
      <div class="topbar-actions">
        <label id="tenant-picker" class="tenant-picker" hidden>
          <span>租户</span>
          <select id="tenant-select"></select>
        </label>
        <span id="readiness" class="pill">检查中…</span>
        <button id="refresh" class="primary" type="button">刷新</button>
      </div>
```

Docs table: add `<th>租户</th>` after `<th>文档</th>`, change empty colspan to `8`.

Tasks table: add `<th>租户</th>` after `<th>任务</th>`, change empty colspan to `12`.

Ingest section: add before `<div class="grid-2">`:

```html
      <p id="tenant-guard" class="hint" hidden>
        全局管理员需先在顶部选择具体租户，才能上传或扫描。
      </p>
```

Search section: add after the toolbar:

```html
      <p id="search-tenant-guard" class="hint" hidden>
        全局管理员需先在顶部选择具体租户，才能测试检索。
      </p>
```

- [ ] **Step 2: Update `admin.css`**

Append:

```css
.tenant-picker {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--ink-soft);
}

.tenant-picker select {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  padding: 6px 10px;
}

.tenant-picker[hidden] { display: none; }
```

- [ ] **Step 3: Update `admin.js`**

现有 `api()` 已对 401 提示并聚焦凭证输入，其他非 2xx（含 403）抛出 `Error` 并由调用方 `toast(error.message, true)` 呈现；保留该行为即可满足错误提示要求，无需新增处理。

State:

```js
const state = {
  tab: "overview",
  token: sessionStorage.getItem("adminToken") || "",
  tenantId: sessionStorage.getItem("adminTenant") || "",
  isGlobalAdmin: false,
  docs: { limit: 20, offset: 0, total: 0 },
  tasks: { limit: 20, offset: 0, total: 0, status: "" },
  detailDocId: "",
  toastTimer: null,
};
```

`api()` headers:

```js
  if (state.token) {
    headers["X-Admin-Token"] = state.token;
    headers["X-API-Key"] = state.token;
  }
  if (state.tenantId) headers["X-Tenant-Id"] = state.tenantId;
```

Add helpers after `loadReadiness()`:

```js
async function loadTenants() {
  const picker = $("tenant-picker");
  const select = $("tenant-select");
  try {
    const data = await api("/admin/tenants");
    state.isGlobalAdmin = data.scope === "all";
    if (state.isGlobalAdmin) {
      const options = ['<option value="">全部租户</option>'].concat(
        data.tenants.map(
          (tenant) =>
            `<option value="${escapeHtml(tenant.tenant_id)}">${escapeHtml(tenant.tenant_name)}</option>`
        )
      );
      select.innerHTML = options.join("");
      if (!data.tenants.some((tenant) => tenant.tenant_id === state.tenantId)) {
        state.tenantId = "";
      }
      select.value = state.tenantId;
      picker.hidden = false;
    } else {
      state.tenantId = data.tenant_id || "";
      select.innerHTML = "";
      picker.hidden = true;
    }
  } catch {
    state.isGlobalAdmin = false;
    picker.hidden = true;
  }
  updateTenantControls();
}

function tenantParamString() {
  return state.tenantId ? `tenant_id=${encodeURIComponent(state.tenantId)}` : "";
}

function updateTenantControls() {
  const blocked = state.isGlobalAdmin && !state.tenantId;
  [
    "upload-file",
    "upload-btn",
    "scan-force",
    "scan-btn",
    "search-query",
    "search-mode",
    "search-topk",
    "search-rerank",
    "search-city",
    "search-type",
    "search-btn",
  ].forEach((id) => {
    const element = $(id);
    if (element) element.disabled = blocked;
  });
  $("tenant-guard").hidden = !blocked;
  $("search-tenant-guard").hidden = !blocked;
}
```

Overview request:

```js
async function loadOverview() {
  const cards = $("overview-cards");
  cards.innerHTML = '<div class="card muted">加载中…</div>';
  try {
    const suffix = tenantParamString();
    const data = await api(`/admin/overview${suffix ? `?${suffix}` : ""}`);
    ...
```

`documentParams()`:

```js
  if (state.tenantId) params.set("tenant_id", state.tenantId);
```

Docs row: add tenant cell after the title cell:

```js
      <td>${escapeHtml(item.tenant_id || "—")}</td>
```

Tasks row: add tenant cell after the task id cell:

```js
        <td>${escapeHtml(task.tenant_id || "—")}</td>
```

`showIngestResult`: include tenant:

```js
  container.innerHTML = `<p>${escapeHtml(label)}：租户 <strong>${escapeHtml(
    state.tenantId || "default"
  )}</strong>，任务 <strong>${escapeHtml(task.task_id)}</strong> 已排队（${escapeHtml(
    task.status
  )}）。</p>
    <button class="ghost small" type="button" id="goto-task">查看任务</button>`;
```

Guard checks in actions (defensive, in addition to disabled controls):

```js
async function uploadDocument() {
  if (state.isGlobalAdmin && !state.tenantId) {
    toast("请先选择具体租户", true);
    return;
  }
  ...
}
```

Add the same guard at the top of `scanDirectory()` and `runSearch()`.

Wiring:

```js
  $("tenant-select").addEventListener("change", () => {
    state.tenantId = $("tenant-select").value;
    if (state.tenantId) sessionStorage.setItem("adminTenant", state.tenantId);
    else sessionStorage.removeItem("adminTenant");
    state.docs.offset = 0;
    state.tasks.offset = 0;
    updateTenantControls();
    loadReadiness();
    loadTab(state.tab);
  });
```

Token save handler:

```js
  $("save-token").addEventListener("click", async () => {
    state.token = $("admin-token").value.trim();
    if (state.token) sessionStorage.setItem("adminToken", state.token);
    else sessionStorage.removeItem("adminToken");
    toast("访问凭证已保存");
    await loadTenants();
    loadTab(state.tab);
  });
```

Startup:

```js
bindEvents();
updateTenantControls();
loadReadiness();
loadTenants();
const initialTab = window.location.hash.replace("#", "");
switchTab(PAGE_TITLES[initialTab] ? initialTab : "overview", false);
```

- [ ] **Step 4: Add the tenant example to `.env.example`**

After the `ADMIN_TOKEN` lines:

```env
# Multi-tenancy (optional). Empty means open mode: anonymous requests act as
# the default tenant admin. Each api_key must be unique and at least 16 chars.
# TENANTS=[{"tenant_id":"default","tenant_name":"默认租户","api_key":"replace-with-32-char-key","role":"admin"}]
```

- [ ] **Step 5: Manual verification**

1. Start the service with no `TENANTS`: `curl http://localhost:8000/readyz` must be ready; `/admin/` shows no tenant selector and all tabs work.
2. Configure two tenants and restart (generate keys locally, never paste them into the repo):
   - global admin token: tenant selector shows “全部租户 + default + sanya”;
   - select “全部租户”: upload/scan/search disabled with a hint;
   - select `sanya`: docs/tasks filtered, upload/scan enabled;
   - viewer key: no selector, own tenant only, upload/scan controls disabled by 403 toast.
3. Confirm the credential input value is never echoed in network responses and no key appears in the tenant dropdown.

- [ ] **Step 6: Commit (only if authorized)**

```bash
git add app/static/admin/index.html app/static/admin/admin.css app/static/admin/admin.js .env.example
git commit -m "feat: add tenant controls to admin console"
```

---

### Task 12: CLI 与迁移脚本

**Files:**
- Modify: `scripts/ingest_dir.py`
- Modify: `scripts/eval_retrieval.py`
- Modify: `scripts/eval_qa.py`
- Create: `scripts/migrate_multitenant.py`
- Test: `tests/unit/test_eval_qa.py` (fake signature)

**Interfaces:**
- Produces: `--tenant` on all three scripts (default `default`).
- Produces: `scripts/migrate_multitenant.py` with `--apply` and `--recreate-milvus`.

- [ ] **Step 1: Update the evaluation test fake**

In `tests/unit/test_eval_qa.py`:

```python
    class FakeRetriever:
        async def search(self, query, *, tenant_id, top_k, city, content_type):
            self.tenant_id = tenant_id
            ...
```

Update the direct call so the required keyword is supplied, and assert it reaches the retriever:

```python
    row = await ev.evaluate_case(
        _case(),
        retriever=retriever,
        llm=FakeLLM(),
        routing="off",
        top_k=5,
        tenant_id="default",
    )

    assert retriever.tenant_id == "default"
```

The existing test constructs the fake inline (`retriever=FakeRetriever()`); hoist it to a local variable first so the assertion can inspect it.

- [ ] **Step 2: Update `scripts/ingest_dir.py`**

```python
from app.core.tenancy import DEFAULT_TENANT_ID


    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT_ID,
        help="tenant id (default: default)",
    )
    ...
        summary = await pipeline.run(
            directory=args.directory,
            tenant_id=args.tenant,
            ...
        )
    ...
        print(f"milvus live chunks: {milvus.count_chunks(args.tenant)}")
```

- [ ] **Step 3: Update `scripts/eval_retrieval.py`**

Add `--tenant` to the parser, thread it through `_run_local`, `_evaluate_three_way`, and `_run_three_way`:

```python
from app.core.tenancy import DEFAULT_TENANT_ID
```

```python
def _run_local(cases, top_k: int, tenant_id: str) -> int:
    ...
            results = service.search_all(query, tenant_id=tenant_id, top_k=top_k)
            results["hybrid_rerank"] = service.search(
                query, tenant_id=tenant_id, top_k=top_k
            )
```

```python
async def _evaluate_three_way(
    cases,
    top_k: int,
    tenant_id: str,
    show_hits: int = 0,
    routing: str = "off",
) -> tuple[list[int | None], list[dict]]:
    ...
            outcome = await retriever.search(
                query,
                tenant_id=tenant_id,
                top_k=top_k,
                ...
            )
```

```python
def _run_three_way(cases, top_k: int, tenant_id: str, show_hits=0, routing="off") -> int:
    ranks, infos = asyncio.run(
        _evaluate_three_way(cases, top_k, tenant_id, show_hits, routing)
    )
```

```python
    parser.add_argument("--tenant", default=DEFAULT_TENANT_ID)
    ...
    if args.pipeline == "three-way":
        return _run_three_way(
            cases, args.top_k, args.tenant, args.show_hits, args.routing
        )
    return _run_local(cases, args.top_k, args.tenant)
```

- [ ] **Step 4: Update `scripts/eval_qa.py`**

```python
from app.core.tenancy import DEFAULT_TENANT_ID
```

Add `tenant_id: str` to `evaluate_case`, `run_cases`, and `_run` keyword parameters, and pass to the retriever:

```python
    outcome = await retriever.search(
        question,
        tenant_id=tenant_id,
        top_k=top_k,
        city=city,
        content_type=content_type,
    )
```

Add the CLI flag and thread it:

```python
    parser.add_argument("--tenant", default=DEFAULT_TENANT_ID)
    ...
                tenant_id=args.tenant,
```

Include `"tenant": args.tenant` in the JSON report config.

- [ ] **Step 5: Write `scripts/migrate_multitenant.py`**

```python
"""Migrate existing collections and vectors to the multi-tenant schema.

Dry-run by default. Use --apply to write MongoDB changes. Recreating the Milvus
collection is destructive and requires --apply --recreate-milvus.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID
from app.ingestion.metadata import make_doc_id
from app.repositories.mongo import (
    CHAT_MESSAGES,
    CHAT_SESSIONS,
    DOCUMENTS,
    INGESTION_TASKS,
    MongoRepository,
)
from app.retrieval.milvus_repo import MilvusRepository

TENANT_COLLECTIONS = (DOCUMENTS, INGESTION_TASKS, CHAT_SESSIONS, CHAT_MESSAGES)

LEGACY_INDEXES = {
    DOCUMENTS: [
        "uq_source_path",
        "ix_sha256",
        "ix_city_type",
        "ix_status_imported",
    ],
    INGESTION_TASKS: ["ix_status_created", "ix_created"],
    CHAT_SESSIONS: ["ix_user_updated", "ix_updated"],
    CHAT_MESSAGES: ["ix_session_created"],
}


async def backfill_tenant_ids(mongo: MongoRepository, *, apply: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in TENANT_COLLECTIONS:
        query = {"tenant_id": {"$exists": False}}
        counts[name] = await mongo.db[name].count_documents(query)
        if apply and counts[name]:
            await mongo.db[name].update_many(
                query, {"$set": {"tenant_id": DEFAULT_TENANT_ID}}
            )
    return counts


async def recompute_doc_ids(
    mongo: MongoRepository, *, apply: bool
) -> tuple[int, list[str]]:
    changed = 0
    skipped: list[str] = []
    cursor = mongo.db[DOCUMENTS].find(
        {},
        {"_id": 1, "tenant_id": 1, "source_path": 1, "doc_id": 1},
    )
    async for document in cursor:
        source_path = document.get("source_path")
        if not source_path:
            skipped.append(str(document["_id"]))
            continue
        tenant_id = document.get("tenant_id") or DEFAULT_TENANT_ID
        doc_id = make_doc_id(tenant_id, source_path)
        if doc_id == document.get("doc_id"):
            continue
        changed += 1
        if apply:
            await mongo.db[DOCUMENTS].update_one(
                {"_id": document["_id"]},
                {"$set": {"tenant_id": tenant_id, "doc_id": doc_id}},
            )
    return changed, skipped


async def drop_legacy_indexes(
    mongo: MongoRepository, *, apply: bool
) -> dict[str, list[str]]:
    present: dict[str, list[str]] = {}
    for collection, names in LEGACY_INDEXES.items():
        existing = await mongo.db[collection].index_information()
        found = [name for name in names if name in existing]
        present[collection] = found
        if apply:
            for name in found:
                await mongo.db[collection].drop_index(name)
    return present


def milvus_health(milvus: MilvusRepository) -> tuple[bool, list[str]]:
    if not milvus.has_collection():
        return False, ["collection missing"]
    problems = milvus.validate()
    return not problems, problems


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes")
    parser.add_argument(
        "--recreate-milvus",
        action="store_true",
        help="drop and recreate the Milvus collection (destructive)",
    )
    args = parser.parse_args()

    settings = get_settings()
    mongo = MongoRepository(settings)
    milvus = MilvusRepository(settings)
    try:
        await mongo.ping()
        backfill = await backfill_tenant_ids(mongo, apply=args.apply)
        changed, skipped = await recompute_doc_ids(mongo, apply=args.apply)
        legacy = await drop_legacy_indexes(mongo, apply=args.apply)

        if args.apply:
            await mongo.ensure_schema()

        healthy, problems = await asyncio.to_thread(milvus_health, milvus)
        recreated = False
        if args.apply and args.recreate_milvus:
            milvus.create_collection(
                recreate=True,
                consistency_level=settings.milvus_consistency_level,
            )
            milvus.ensure_loaded()
            healthy, problems = True, []
            recreated = True

        print(f"mode: {'apply' if args.apply else 'dry-run'}")
        for name, count in backfill.items():
            print(f"backfill {name}: {count} document(s) without tenant_id")
        print(f"doc ids to rewrite: {changed}")
        if skipped:
            print(f"documents without source_path skipped: {len(skipped)}")
            for object_id in skipped:
                print(f"  skipped: {object_id}")
        for collection, names in legacy.items():
            print(f"legacy indexes {collection}: {', '.join(names) or 'none'}")
        print(f"milvus tenant schema: {'ok' if healthy else 'not ready'}")
        for problem in problems:
            print(f"  problem: {problem}")
        if recreated:
            print("milvus collection recreated: all vectors were removed")

        if not args.apply:
            print("dry-run: no changes written; re-run with --apply")
            return 0

        print("next: .venv/Scripts/python scripts/ingest_dir.py --tenant default --force")
        if not healthy:
            print(
                "milvus is missing the tenant schema; "
                "re-run with --apply --recreate-milvus"
            )
            return 2
        return 0
    finally:
        milvus.close()
        await mongo.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 6: Verify scripts parse and targeted tests pass**

Run:

```bash
.venv/Scripts/python scripts/migrate_multitenant.py --help
.venv/Scripts/python scripts/ingest_dir.py --help
.venv/Scripts/python -m pytest tests/unit/test_eval_qa.py tests/unit/test_eval_retrieval.py -q
```

Expected: help text contains `--tenant` / `--apply`; tests pass.

- [ ] **Step 7: Commit (only if authorized)**

```bash
git add scripts/ingest_dir.py scripts/eval_retrieval.py scripts/eval_qa.py scripts/migrate_multitenant.py tests/unit/test_eval_qa.py
git commit -m "feat: add tenant CLI options and migration script"
```

---

### Task 13: 集成测试、迁移与全量验收

**Files:**
- Modify: `tests/integration/test_ingest_retrieval_loop.py`
- Modify: `tests/integration/test_milvus_search.py`
- Create: `tests/integration/test_tenant_isolation.py`
- Modify: `项目架构.txt`
- Modify: `任务进度.md`

**Interfaces:**
- Consumes: all tenant-aware interfaces above.
- Produces: cross-tenant isolation coverage and updated project documentation.

- [ ] **Step 1: Update `tests/integration/test_ingest_retrieval_loop.py`**

Use `"default"` for the task, pipeline, retrieval, and cleanup:

```python
    task_id = await repo.create_task("default", "integration", 1)
...
        summary = await pipeline.run_task(
            task_id,
            [scanned],
            tenant_id="default",
            force=True,
            progress=lambda *_args: None,
        )
...
        document = await repo.get_document("default", relative_path)
...
        task = await repo.get_task("default", task_id)
...
        hits = retrieval.search(
            "西湖适合骑行还是步行游览",
            tenant_id="default",
            mode="hybrid",
            top_k=5,
            rerank=False,
            doc_ids=[doc_id],
        )
...
        pruned = await pipeline.run(
            tenant_id="default", prune=True, progress=lambda *_args: None
        )
```

Update cleanup:

```python
            deleted = await IngestionRepository(verify_mongo).get_document(
                "default", relative_path
            )
...
        cleanup_milvus.delete_document_chunks("default", doc_id)
...
            await cleanup_mongo.db[DOCUMENTS].delete_one(
                {"tenant_id": "default", "source_path": relative_path}
            )
            ...
                await cleanup_mongo.db[INGESTION_TASKS].delete_one(
                    {"_id": ObjectId(task), "tenant_id": "default"}
                )
```

Also update the direct Milvus count query:

```python
                filter=f'tenant_id == "default" and doc_id == "{doc_id}"',
```

- [ ] **Step 2: Update `tests/integration/test_milvus_search.py`**

Include `tenant_id` in the seed fields and assert schema:

```python
        output_fields=[
            "tenant_id",
            "chunk_id",
            "doc_id",
            "city",
            "content",
            DENSE_FIELD,
            SPARSE_FIELD,
        ],
```

Add:

```python
def test_tenant_filter_is_applied(milvus):
    seed = _seed(milvus, 'tenant_id != ""')
    tenant_id = seed["tenant_id"]

    hits = milvus.search_dense(
        seed[DENSE_FIELD],
        limit=10,
        filter_expr=f'tenant_id == "{tenant_id}"',
    )

    assert hits
    assert all((hit.get("entity") or {}).get("tenant_id") == tenant_id for hit in hits)
```

- [ ] **Step 3: Add cross-tenant integration test**

Create `tests/integration/test_tenant_isolation.py`:

```python
import shutil
import uuid

import pytest
from bson import ObjectId

from app.core.tenancy import DEFAULT_TENANT_ID
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import build_scanned_file
from app.ingestion.task_repo import IngestionRepository
from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS, MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService

pytestmark = pytest.mark.integration

SAMPLE = """# 租户隔离测试

## 元数据
- 城市：杭州
- 内容类型：景点介绍
- 景点名称：西湖

## 简介

西湖适合清晨骑行，湖岸有苏堤和白堤。
"""


def _write(root, name: str, text: str):
    path = root / "uploads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_tenants_cannot_retrieve_each_others_chunks(
    settings, mongo, milvus, embedder
):
    suffix = uuid.uuid4().hex[:8]
    tenant_b = f"it{suffix}"
    root_a = settings.tenant_data_path(DEFAULT_TENANT_ID)
    root_b = settings.tenant_data_path(tenant_b)
    file_a = _write(root_a, f"tenant-a-{suffix}.md", SAMPLE)
    file_b = _write(root_b, f"tenant-b-{suffix}.md", SAMPLE)
    scanned_a = build_scanned_file(file_a, root_a)
    scanned_b = build_scanned_file(file_b, root_b)

    repo = IngestionRepository(mongo)
    pipeline = IngestionPipeline(
        settings, embedder=embedder, mongo=mongo, milvus=milvus, repo=repo
    )
    retrieval = RetrievalService(settings, embedder=embedder, milvus=milvus)
    task_a = await repo.create_task(DEFAULT_TENANT_ID, "integration", 1)
    task_b = await repo.create_task(tenant_b, "integration", 1)
    doc_a = doc_b = ""

    try:
        summary_a = await pipeline.run_task(
            task_a,
            [scanned_a],
            tenant_id=DEFAULT_TENANT_ID,
            force=True,
            progress=lambda *_args: None,
        )
        summary_b = await pipeline.run_task(
            task_b,
            [scanned_b],
            tenant_id=tenant_b,
            force=True,
            progress=lambda *_args: None,
        )
        assert summary_a.failed == 0, summary_a.errors
        assert summary_b.failed == 0, summary_b.errors

        document_a = await repo.get_document(DEFAULT_TENANT_ID, scanned_a.relative_path)
        document_b = await repo.get_document(tenant_b, scanned_b.relative_path)
        assert document_a["tenant_id"] == DEFAULT_TENANT_ID
        assert document_b["tenant_id"] == tenant_b
        assert document_a["doc_id"] != document_b["doc_id"]
        doc_a, doc_b = document_a["doc_id"], document_b["doc_id"]

        hits_a = retrieval.search(
            "西湖适合骑行吗",
            tenant_id=DEFAULT_TENANT_ID,
            mode="hybrid",
            top_k=5,
            rerank=False,
        )
        hits_b = retrieval.search(
            "西湖适合骑行吗",
            tenant_id=tenant_b,
            mode="hybrid",
            top_k=5,
            rerank=False,
        )
        assert any(hit["doc_id"] == doc_a for hit in hits_a)
        assert all(hit["doc_id"] != doc_b for hit in hits_a)
        assert any(hit["doc_id"] == doc_b for hit in hits_b)
        assert all(hit["doc_id"] != doc_a for hit in hits_b)
    finally:
        file_a.unlink(missing_ok=True)
        file_b.unlink(missing_ok=True)
        shutil.rmtree(root_b, ignore_errors=True)

        cleanup_mongo = MongoRepository(settings)
        cleanup_milvus = MilvusRepository(settings)
        try:
            for tenant_id, doc_id in (
                (DEFAULT_TENANT_ID, doc_a),
                (tenant_b, doc_b),
            ):
                if doc_id:
                    cleanup_milvus.delete_document_chunks(tenant_id, doc_id)
            cleanup_milvus.flush()
            await cleanup_mongo.db[DOCUMENTS].delete_many(
                {
                    "$or": [
                        {
                            "tenant_id": DEFAULT_TENANT_ID,
                            "source_path": scanned_a.relative_path,
                        },
                        {"tenant_id": tenant_b},
                    ]
                }
            )
            for task_id in (task_a, task_b):
                await cleanup_mongo.db[INGESTION_TASKS].delete_one(
                    {"_id": ObjectId(task_id)}
                )
        finally:
            await cleanup_mongo.close()
            cleanup_milvus.close()
```

- [ ] **Step 4: Run the full unit suite**

Run: `.venv/Scripts/python -m pytest -q`

Expected: PASS. Record the count (118 + new tests).

- [ ] **Step 5: Real migration and re-ingest**

Integration tests use the live Milvus collection, so the tenant schema must exist before
Step 6. Back up MongoDB first. Stop the API service. Then:

```bash
.venv/Scripts/python scripts/migrate_multitenant.py
.venv/Scripts/python scripts/migrate_multitenant.py --apply --recreate-milvus
.venv/Scripts/python scripts/ingest_dir.py --tenant default --force
```

Expected:

- dry-run prints counts and changes nothing;
- apply rewrites 30 documents to `tenant_id=default`, drops legacy indexes, creates tenant indexes, recreates `travel_chunks`;
- force ingest imports 30 documents / 851 chunks with no failures.

Verify:

```bash
curl http://localhost:8000/readyz
```

Expected: `status=ready`, `chunk_count=851`.

Rollback: restore the Mongo backup and re-ingest vectors with the previous code version; the migration never modifies source files.

- [ ] **Step 6: Run integration tests**

Run: `.venv/Scripts/python -m pytest -m integration -q`

Expected: 10 existing + 1 new test pass (or skip only when Mongo/Milvus/model unavailable).

- [ ] **Step 7: Two-tenant smoke test**

Add tenant credentials to local `.env` without committing them (generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`). Restart the service, then verify:

1. no credentials → `POST /api/v1/search` returns 401;
2. default editor key → search hits default documents; upload works;
3. sanya viewer key → search returns no default documents; upload returns 403;
4. default key requesting `X-Tenant-Id: sanya` → 403 `TENANT_MISMATCH`;
5. global admin token → `/api/v1/admin/tenants` shows both tenants without keys; `/admin/documents?tenant_id=sanya` is empty; overview shows all or one tenant.
6. browser: open `/admin/`, select each tenant, confirm docs/tasks filters, tenant columns, and the “全部租户” safeguards; capture screenshots of the credential prompt, global tenant filter, tenant-admin view, and tasks page.

- [ ] **Step 8: Update project docs**

In `项目架构.txt`, replace the pending item “权限与多租户” with a completed section describing:
`X-API-Key` / `X-Admin-Token` / `X-Tenant-Id`, viewer/editor/admin, tenant-scoped Mongo/Milvus, tenant data directories, migration command, and open mode.

In `任务进度.md`, append a completed section with:
- implementation date, files, test counts;
- migration and re-ingest result (`30 documents / 851 chunks`);
- two-tenant verification result;
- remaining pending items (视频片段检索、多模态检索).

- [ ] **Step 9: Commit (only if authorized)**

```bash
git add tests/integration/test_ingest_retrieval_loop.py tests/integration/test_milvus_search.py tests/integration/test_tenant_isolation.py 项目架构.txt 任务进度.md
git commit -m "test: verify tenant isolation end to end"
```

---

## 验收清单

- [ ] 未配置 `TENANTS` 时 118 项既有单测行为不变；配置后 401/403/404 语义符合规格。
- [ ] 任意检索过滤表达式都包含 `tenant_id`（`tests/unit/test_filters.py`）。
- [ ] 两个租户导入同名文件产生不同 `doc_id`，互相检索不到（集成测试）。
- [ ] `scripts/migrate_multitenant.py` 默认 dry-run，`--apply --recreate-milvus` 才销毁向量。
- [ ] 迁移后 `/readyz` 为 `ready`，`chunk_count=851`，30 份文档均为 `tenant_id=default`。
- [ ] 管理后台：全局管理员可汇总/筛选租户；租户管理员只看到本租户；全局“全部”视图禁用写操作与检索测试。
- [ ] API Key、管理员令牌不出现在日志、响应、管理页下拉或仓库文件中。
