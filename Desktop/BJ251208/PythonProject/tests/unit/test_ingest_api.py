from datetime import UTC, datetime

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.api.routes import ingest
from app.core.config import Settings
from app.core.exceptions import AppError
from app.ingestion.scanner import ScannedFile


class FakeRepo:
    def __init__(self, tasks: dict[str, dict] | None = None) -> None:
        self.tasks = tasks or {}
        self.created: list[dict] = []
        self._counter = len(self.tasks)

    async def create_task(self, task_type: str, total: int, *, status: str = "running") -> str:
        self._counter += 1
        task_id = f"task-{self._counter}"
        self.tasks[task_id] = _task(task_id, status=status, task_type=task_type, total=total)
        self.created.append({"task_type": task_type, "total": total, "status": status})
        return task_id

    async def get_task(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)

    async def list_tasks(self, *, status=None, limit=20, offset=0) -> tuple[int, list[dict]]:
        items = list(self.tasks.values())
        if status:
            items = [item for item in items if item["status"] == status]
        return len(items), items[offset : offset + limit]


class FakeWorker:
    def __init__(self) -> None:
        self.submitted: list[dict] = []

    def submit(self, task_id: str, files, *, force: bool = False) -> None:
        self.submitted.append({"task_id": task_id, "files": list(files), "force": force})


def _task(
    task_id: str,
    *,
    status: str = "success",
    task_type: str = "scan",
    total: int = 1,
    errors: list[dict] | None = None,
) -> dict:
    now = datetime.now(UTC)
    return {
        "_id": task_id,
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
        data_dir=tmp_path,
        max_upload_mb=1,
        allowed_upload_suffixes=".md,.markdown,.txt,.pdf,.docx",
    )
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
    monkeypatch.setattr(ingest, "scan_directory", lambda directory: [scanned])
    repo = FakeRepo()
    worker = FakeWorker()
    app = _app(tmp_path, repo, worker)

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/scan", json={"force": True})

    assert response.status_code == 200
    assert response.json()["task_id"] == "task-1"
    assert repo.created[-1] == {"task_type": "scan", "total": 1, "status": "queued"}
    assert worker.submitted[0]["files"] == [scanned]
    assert worker.submitted[0]["force"] is True


async def test_scan_missing_data_dir_returns_400(monkeypatch, tmp_path):
    def fail(directory):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(ingest, "scan_directory", fail)
    app = _app(tmp_path, FakeRepo(), FakeWorker())

    async with _client(app) as client:
        response = await client.post("/api/v1/ingest/scan", json={})

    assert response.status_code == 400
    assert response.json()["code"] == "DATA_DIR_UNAVAILABLE"


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
    assert repo.created[-1] == {"task_type": "upload", "total": 1, "status": "queued"}
    submitted = worker.submitted[0]
    assert submitted["files"][0].relative_path == "uploads/travel.md"
    assert submitted["force"] is False


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
    assert detail.status_code == 200
    assert detail.json()["status"] == "failed"
    assert missing.status_code == 404


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
    assert repo.created[-1] == {"task_type": "retry", "total": 2, "status": "queued"}
    assert worker.submitted[0]["force"] is True


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


@pytest.mark.parametrize("task_id", ["bad-id", "unknown"])
async def test_get_task_with_invalid_id_returns_404(tmp_path, task_id):
    app = _app(tmp_path, FakeRepo(), FakeWorker())

    async with _client(app) as client:
        response = await client.get(f"/api/v1/ingest/tasks/{task_id}")

    assert response.status_code == 404
