from datetime import UTC, datetime

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.api.routes import ingest
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.tenancy import TenantCredential, TenantRegistry
from app.ingestion.scanner import ScannedFile

EDITOR_KEY = "e" * 32
VIEWER_KEY = "v" * 32
ADMIN_TOKEN = "admin-secret-token-123456"
EDITOR = TenantCredential(
    tenant_id="sanya", tenant_name="三亚运营", api_key=EDITOR_KEY, role="editor"
)
VIEWER = TenantCredential(
    tenant_id="sanya", tenant_name="三亚运营", api_key=VIEWER_KEY, role="viewer"
)


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
    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.active: set[str] = set()

    def is_active(self, task_id: str) -> bool:
        return task_id in self.active

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


def _task(
    task_id: str,
    *,
    status: str = "success",
    task_type: str = "scan",
    total: int = 1,
    errors: list[dict] | None = None,
    tenant_id: str = "default",
) -> dict:
    now = datetime.now(UTC)
    return {
        "_id": task_id,
        "task_id": task_id,
        "tenant_id": tenant_id,
        "task_type": task_type,
        "status": status,
        "total": total,
        "succeeded": total if status == "success" else 0,
        "failed": 1 if status == "failed" else 0,
        "skipped": 0,
        "current_file": "",
        "errors": errors or [],
        "created_at": now,
        "started_at": now,
        "finished_at": now,
        "duration_seconds": 1.2,
    }


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

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    app.include_router(ingest.router, prefix="/api/v1")
    return app


def _tenant_app(
    tmp_path, repo: FakeRepo, worker: FakeWorker, *, tenants, admin_token=""
) -> FastAPI:
    app = _app(tmp_path, repo, worker)
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        max_upload_mb=1,
        allowed_upload_suffixes=".md,.markdown,.txt,.pdf,.docx",
        tenants=tenants,
        admin_token=admin_token,
    )
    app.state.settings = settings
    app.state.tenants = TenantRegistry(settings.tenants, settings.admin_token)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_scan_creates_queued_task_and_submits(monkeypatch, tmp_path):
    scanned = ScannedFile(
        path=tmp_path / "a.md",
        relative_path="a.md",
        size=1,
        mtime=0.0,
        sha256="abc",
    )
    monkeypatch.setattr(ingest, "scan_tenant_directory", lambda settings, tenant_id: [scanned])
    repo = FakeRepo()
    worker = FakeWorker()
    app = _app(tmp_path, repo, worker)

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/scan", json={"force": True})

    assert response.status_code == 200
    assert response.json()["task_id"] == "task-1"
    assert repo.created[-1] == {
        "tenant_id": "default",
        "task_type": "scan",
        "total": 1,
        "status": "queued",
    }
    assert worker.submitted[0]["files"] == [scanned]
    assert worker.submitted[0]["force"] is True


async def test_scan_prune_submits_flag(monkeypatch, tmp_path):
    scanned = ScannedFile(
        path=tmp_path / "a.md",
        relative_path="a.md",
        size=1,
        mtime=0.0,
        sha256="abc",
    )
    monkeypatch.setattr(ingest, "scan_tenant_directory", lambda settings, tenant_id: [scanned])
    worker = FakeWorker()
    app = _app(tmp_path, FakeRepo(), worker)

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/scan", json={"prune": True})

    assert response.status_code == 200
    assert worker.submitted[0]["prune"] is True


async def test_scan_rejects_prune_with_limit(monkeypatch, tmp_path):
    scanned = ScannedFile(
        path=tmp_path / "a.md",
        relative_path="a.md",
        size=1,
        mtime=0.0,
        sha256="abc",
    )
    monkeypatch.setattr(ingest, "scan_tenant_directory", lambda settings, tenant_id: [scanned])
    repo = FakeRepo()
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/scan",
            json={"prune": True, "limit": 1},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_PRUNE"
    assert repo.created == []


async def test_scan_rejects_prune_when_scan_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "scan_tenant_directory", lambda settings, tenant_id: [])
    repo = FakeRepo()
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/scan", json={"prune": True})

    assert response.status_code == 400
    assert response.json()["code"] == "PRUNE_EMPTY_SCAN"
    assert repo.created == []


async def test_scan_missing_data_dir_returns_400(monkeypatch, tmp_path):
    def fail(settings, tenant_id):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(ingest, "scan_tenant_directory", fail)
    app = _app(tmp_path, FakeRepo(), FakeWorker())

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/scan", json={})

    assert response.status_code == 400
    assert response.json()["code"] == "DATA_DIR_UNAVAILABLE"


async def test_scan_uses_selected_tenant_directory(monkeypatch, tmp_path):
    calls: list[tuple[object, str]] = []

    def fake_scan(settings, tenant_id):
        calls.append((settings, tenant_id))
        return []

    monkeypatch.setattr(ingest, "scan_tenant_directory", fake_scan)
    app = _tenant_app(tmp_path, FakeRepo(), FakeWorker(), tenants=[EDITOR])

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/scan", json={}, headers={"X-API-Key": EDITOR_KEY}
        )

    assert response.status_code == 200
    assert calls[0][0] is app.state.settings
    assert calls[0][1] == "sanya"


async def test_upload_saves_file_and_queues_task(tmp_path):
    repo = FakeRepo()
    worker = FakeWorker()
    app = _app(tmp_path, repo, worker)

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/upload",
            files={"file": ("travel.md", b"# Travel\n\n## Guide\nGood.", "text/markdown")},
        )

    assert response.status_code == 200
    saved = tmp_path / "uploads" / "travel.md"
    assert saved.read_bytes() == b"# Travel\n\n## Guide\nGood."
    assert repo.created[-1] == {
        "tenant_id": "default",
        "task_type": "upload",
        "total": 1,
        "status": "queued",
    }
    submitted = worker.submitted[0]
    assert submitted["files"][0].relative_path == "uploads/travel.md"
    assert submitted["force"] is False


async def test_upload_writes_into_tenant_uploads(tmp_path):
    repo = FakeRepo()
    worker = FakeWorker()
    app = _tenant_app(tmp_path, repo, worker, tenants=[EDITOR])

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/upload",
            files={"file": ("guide.md", b"# Guide\n\n## Body\ncontent", "text/markdown")},
            headers={"X-API-Key": EDITOR_KEY},
        )

    assert response.status_code == 200
    saved = tmp_path / "tenants" / "sanya" / "uploads" / "guide.md"
    assert saved.read_bytes() == b"# Guide\n\n## Body\ncontent"
    assert not (tmp_path / "uploads" / "guide.md").exists()
    assert repo.created[-1]["tenant_id"] == "sanya"
    assert worker.submitted[0]["tenant_id"] == "sanya"
    assert worker.submitted[0]["files"][0].relative_path == "uploads/guide.md"


async def test_upload_rejects_unsupported_suffix(tmp_path):
    repo = FakeRepo()
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/upload",
            files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "UNSUPPORTED_FILE_TYPE"
    assert repo.created == []


async def test_upload_rejects_oversized_file(tmp_path):
    repo = FakeRepo()
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/upload",
            files={"file": ("big.md", b"x" * (1024 * 1024 + 1), "text/markdown")},
        )

    assert response.status_code == 413
    assert response.json()["code"] == "FILE_TOO_LARGE"
    assert repo.created == []


async def test_list_and_get_tasks(tmp_path):
    repo = FakeRepo(
        {
            "task-1": _task("task-1", status="success"),
            "task-2": _task("task-2", status="failed"),
        }
    )
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        listing = await client.get("/api/v1/ingest/tasks", params={"status": "failed"})
        detail = await client.get("/api/v1/ingest/tasks/task-2")
        missing = await client.get("/api/v1/ingest/tasks/nope")

    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["tasks"][0]["task_id"] == "task-2"
    assert listing.json()["tasks"][0]["tenant_id"] == "default"
    assert detail.status_code == 200
    assert detail.json()["status"] == "failed"
    assert missing.status_code == 404


async def test_viewer_cannot_scan(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "scan_tenant_directory", lambda settings, tenant_id: [])
    app = _tenant_app(tmp_path, FakeRepo(), FakeWorker(), tenants=[VIEWER])

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/scan", json={}, headers={"X-API-Key": VIEWER_KEY}
        )

    assert response.status_code == 403
    assert response.json()["code"] == "FORBIDDEN"


async def test_viewer_can_list_but_not_upload_or_retry(tmp_path):
    repo = FakeRepo({"task-1": _task("task-1", tenant_id="sanya")})
    app = _tenant_app(tmp_path, repo, FakeWorker(), tenants=[VIEWER])

    async with _client(app) as client:
        listing = await client.get("/api/v1/ingest/tasks", headers={"X-API-Key": VIEWER_KEY})
        detail = await client.get("/api/v1/ingest/tasks/task-1", headers={"X-API-Key": VIEWER_KEY})
        upload = await client.post(
            "/api/v1/ingest/upload",
            files={"file": ("a.md", b"# A\n\n## B\nc", "text/markdown")},
            headers={"X-API-Key": VIEWER_KEY},
        )
        retry = await client.post(
            "/api/v1/ingest/tasks/task-1/retry", headers={"X-API-Key": VIEWER_KEY}
        )

    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert detail.status_code == 200
    assert detail.json()["tenant_id"] == "sanya"
    assert upload.status_code == 403
    assert upload.json()["code"] == "FORBIDDEN"
    assert retry.status_code == 403
    assert retry.json()["code"] == "FORBIDDEN"
    assert repo.created == []
    assert repo.updates == []


async def test_editor_scan_and_task_are_scoped(monkeypatch, tmp_path):
    scanned = ScannedFile(
        path=tmp_path / "a.md", relative_path="a.md", size=1, mtime=0.0, sha256="abc"
    )
    monkeypatch.setattr(
        ingest, "scan_tenant_directory", lambda settings, tenant_id: [scanned]
    )
    repo = FakeRepo()
    worker = FakeWorker()
    app = _tenant_app(tmp_path, repo, worker, tenants=[EDITOR])

    async with _client(app) as client:
        created = await client.post(
            "/api/v1/ingest/scan", json={}, headers={"X-API-Key": EDITOR_KEY}
        )
        listing = await client.get(
            "/api/v1/ingest/tasks", headers={"X-API-Key": EDITOR_KEY}
        )

    assert created.status_code == 200
    assert repo.created[-1]["tenant_id"] == "sanya"
    assert worker.submitted[0]["tenant_id"] == "sanya"
    assert listing.json()["total"] == 1
    assert listing.json()["tasks"][0]["tenant_id"] == "sanya"


async def test_tenant_cannot_read_or_retry_other_tenant_task(tmp_path):
    repo = FakeRepo({"task-1": _task("task-1", status="failed", tenant_id="default")})
    app = _tenant_app(tmp_path, repo, FakeWorker(), tenants=[EDITOR, VIEWER])

    async with _client(app) as client:
        detail = await client.get(
            "/api/v1/ingest/tasks/task-1", headers={"X-API-Key": VIEWER_KEY}
        )
        retry = await client.post(
            "/api/v1/ingest/tasks/task-1/retry", headers={"X-API-Key": EDITOR_KEY}
        )

    assert detail.status_code == 404
    assert retry.status_code == 404
    assert repo.created == []
    assert repo.updates == []


async def test_tenant_cannot_select_other_tenant(tmp_path):
    app = _tenant_app(tmp_path, FakeRepo(), FakeWorker(), tenants=[EDITOR])

    async with _client(app) as client:
        response = await client.get(
            "/api/v1/ingest/tasks",
            headers={"X-API-Key": EDITOR_KEY, "X-Tenant-Id": "default"},
        )

    assert response.status_code == 403
    assert response.json()["code"] == "TENANT_MISMATCH"


async def test_global_admin_lists_all_tasks_and_filters(tmp_path):
    repo = FakeRepo(
        {
            "task-1": _task("task-1", tenant_id="default"),
            "task-2": _task("task-2", tenant_id="sanya"),
        }
    )
    editor = TenantCredential(
        tenant_id="default", tenant_name="默认租户", api_key=EDITOR_KEY, role="editor"
    )
    app = _tenant_app(
        tmp_path,
        repo,
        FakeWorker(),
        tenants=[editor, VIEWER],
        admin_token=ADMIN_TOKEN,
    )

    async with _client(app) as client:
        all_tasks = await client.get(
            "/api/v1/ingest/tasks", headers={"X-Admin-Token": ADMIN_TOKEN}
        )
        scoped = await client.get(
            "/api/v1/ingest/tasks",
            headers={"X-Admin-Token": ADMIN_TOKEN, "X-Tenant-Id": "sanya"},
        )

    assert all_tasks.json()["total"] == 2
    assert scoped.json()["total"] == 1
    assert scoped.json()["tasks"][0]["tenant_id"] == "sanya"


async def test_tenant_viewer_listing_excludes_other_tenants(tmp_path):
    repo = FakeRepo(
        {
            "task-1": _task("task-1", tenant_id="default"),
            "task-2": _task("task-2", tenant_id="sanya"),
        }
    )
    default_viewer = TenantCredential(
        tenant_id="default", tenant_name="默认租户", api_key="d" * 32, role="viewer"
    )
    app = _tenant_app(tmp_path, repo, FakeWorker(), tenants=[default_viewer, VIEWER])

    async with _client(app) as client:
        sanya = await client.get(
            "/api/v1/ingest/tasks", headers={"X-API-Key": VIEWER_KEY}
        )
        default = await client.get(
            "/api/v1/ingest/tasks", headers={"X-API-Key": "d" * 32}
        )

    assert sanya.status_code == 200
    assert sanya.json()["total"] == 1
    assert [item["task_id"] for item in sanya.json()["tasks"]] == ["task-2"]
    assert all(item["tenant_id"] == "sanya" for item in sanya.json()["tasks"])
    assert default.status_code == 200
    assert default.json()["total"] == 1
    assert [item["task_id"] for item in default.json()["tasks"]] == ["task-1"]
    assert all(item["tenant_id"] == "default" for item in default.json()["tasks"])


async def test_global_admin_scan_defaults_to_default_tenant(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(
        ingest,
        "scan_tenant_directory",
        lambda settings, tenant_id: calls.append(tenant_id) or [],
    )
    repo = FakeRepo()
    app = _tenant_app(
        tmp_path, repo, FakeWorker(), tenants=[EDITOR], admin_token=ADMIN_TOKEN
    )

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/scan", json={}, headers={"X-Admin-Token": ADMIN_TOKEN}
        )

    assert response.status_code == 200
    assert calls == ["default"]
    assert repo.created[-1]["tenant_id"] == "default"


async def test_global_admin_scan_rejects_unknown_tenant(tmp_path):
    app = _tenant_app(
        tmp_path, FakeRepo(), FakeWorker(), tenants=[EDITOR], admin_token=ADMIN_TOKEN
    )

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/scan",
            json={},
            headers={"X-Admin-Token": ADMIN_TOKEN, "X-Tenant-Id": "mars"},
        )

    assert response.status_code == 404
    assert response.json()["code"] == "TENANT_NOT_FOUND"


async def test_retry_queues_failed_files(tmp_path):
    (tmp_path / "景点攻略").mkdir(parents=True)
    (tmp_path / "景点攻略" / "a.md").write_text("# A\n\n## Body\ncontent", encoding="utf-8")
    (tmp_path / "景点攻略" / "b.md").write_text("# B\n\n## Body\ncontent", encoding="utf-8")
    repo = FakeRepo(
        {
            "task-1": _task(
                "task-1",
                status="partial",
                errors=[
                    {"file": "景点攻略/a.md", "error": "boom"},
                    {"file": "景点攻略/a.md", "error": "boom again"},
                    {"file": "景点攻略/b.md", "error": "boom"},
                ],
            )
        }
    )
    worker = FakeWorker()
    app = _app(tmp_path, repo, worker)

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/tasks/task-1/retry")

    assert response.status_code == 200
    assert response.json()["task_id"] == "task-2"
    assert repo.created[-1] == {
        "tenant_id": "default",
        "task_type": "retry",
        "total": 2,
        "status": "queued",
    }
    assert worker.submitted[0]["tenant_id"] == "default"
    assert worker.submitted[0]["force"] is True


async def test_global_admin_retry_uses_task_tenant_root(tmp_path):
    sanya_root = tmp_path / "tenants" / "sanya"
    (sanya_root / "景点攻略").mkdir(parents=True)
    (sanya_root / "景点攻略" / "a.md").write_text(
        "# A\n\n## Body\ncontent", encoding="utf-8"
    )
    repo = FakeRepo(
        {
            "task-1": _task(
                "task-1",
                status="failed",
                tenant_id="sanya",
                errors=[{"file": "景点攻略/a.md", "error": "boom"}],
            )
        }
    )
    worker = FakeWorker()
    app = _tenant_app(
        tmp_path, repo, worker, tenants=[EDITOR], admin_token=ADMIN_TOKEN
    )

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/tasks/task-1/retry",
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )

    assert response.status_code == 200
    assert repo.created[-1]["tenant_id"] == "sanya"
    assert worker.submitted[0]["tenant_id"] == "sanya"
    assert worker.submitted[0]["files"][0].path == sanya_root / "景点攻略" / "a.md"


async def test_retry_rejects_paths_outside_task_tenant_root(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## Body\ncontent", encoding="utf-8")
    repo = FakeRepo(
        {
            "task-1": _task(
                "task-1",
                status="failed",
                tenant_id="sanya",
                errors=[{"file": "a.md", "error": "boom"}],
            )
        }
    )
    worker = FakeWorker()
    app = _tenant_app(
        tmp_path, repo, worker, tenants=[EDITOR], admin_token=ADMIN_TOKEN
    )

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/ingest/tasks/task-1/retry",
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )

    assert response.status_code == 404
    assert response.json()["code"] == "SOURCE_FILES_MISSING"
    assert repo.created == []
    assert worker.submitted == []


async def test_retry_without_failed_files_returns_400(tmp_path):
    repo = FakeRepo({"task-1": _task("task-1", status="success")})
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/tasks/task-1/retry")

    assert response.status_code == 400
    assert response.json()["code"] == "NO_FAILED_FILES"


async def test_retry_skips_paths_outside_data_dir(tmp_path):
    repo = FakeRepo(
        {
            "task-1": _task(
                "task-1",
                status="failed",
                errors=[{"file": "../secret.md", "error": "boom"}],
            )
        }
    )
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/tasks/task-1/retry")

    assert response.status_code == 404
    assert response.json()["code"] == "SOURCE_FILES_MISSING"
    assert repo.created == []


async def test_list_tasks_marks_orphaned_running_task_stale(tmp_path):
    repo = FakeRepo(
        {
            "task-1": _task("task-1", status="running"),
            "task-2": _task("task-2", status="running"),
        }
    )
    worker = FakeWorker()
    worker.active.add("task-2")
    app = _app(tmp_path, repo, worker)

    async with _client(app) as client:
        listing = await client.get("/api/v1/ingest/tasks")
        detail = await client.get("/api/v1/ingest/tasks/task-1")

    tasks = {item["task_id"]: item for item in listing.json()["tasks"]}
    assert tasks["task-1"]["stale"] is True
    assert tasks["task-2"]["stale"] is False
    assert detail.json()["stale"] is True


async def test_retry_stale_scan_rescans_data_dir(monkeypatch, tmp_path):
    scanned = ScannedFile(
        path=tmp_path / "a.md",
        relative_path="a.md",
        size=1,
        mtime=0.0,
        sha256="abc",
    )
    monkeypatch.setattr(
        ingest, "scan_tenant_directory", lambda settings, tenant_id: [scanned]
    )
    repo = FakeRepo({"task-1": _task("task-1", status="running", total=30)})
    worker = FakeWorker()
    app = _app(tmp_path, repo, worker)

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/tasks/task-1/retry")

    assert response.status_code == 200
    assert repo.updates[0][0] == "default"
    assert repo.updates[0][1] == "task-1"
    assert repo.updates[0][2]["status"] == "failed"
    assert repo.updates[0][2]["finished_at"] is not None
    assert repo.created[-1] == {
        "tenant_id": "default",
        "task_type": "retry",
        "total": 1,
        "status": "queued",
    }
    assert worker.submitted[0]["files"] == [scanned]
    assert worker.submitted[0]["force"] is False


async def test_retry_active_task_returns_409(tmp_path):
    repo = FakeRepo({"task-1": _task("task-1", status="running")})
    worker = FakeWorker()
    worker.active.add("task-1")
    app = _app(tmp_path, repo, worker)

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/tasks/task-1/retry")

    assert response.status_code == 409
    assert response.json()["code"] == "TASK_BUSY"
    assert repo.created == []
    assert repo.updates == []


async def test_retry_stale_non_scan_without_errors_returns_400(tmp_path):
    repo = FakeRepo({"task-1": _task("task-1", status="running", task_type="upload")})
    app = _app(tmp_path, repo, FakeWorker())

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/tasks/task-1/retry")

    assert response.status_code == 400
    assert response.json()["code"] == "NO_FAILED_FILES"
    assert repo.tasks["task-1"]["status"] == "failed"
    assert repo.created == []


@pytest.mark.parametrize("task_id", ["bad-id", "unknown"])
async def test_get_task_with_invalid_id_returns_404(tmp_path, task_id):
    app = _app(tmp_path, FakeRepo(), FakeWorker())

    async with _client(app) as client:
        response = await client.get(f"/api/v1/ingest/tasks/{task_id}")

    assert response.status_code == 404
