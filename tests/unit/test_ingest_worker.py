import asyncio
from pathlib import Path

from app.ingestion.scanner import ScannedFile
from app.workers.ingest_worker import IngestionWorker


def _file(name: str = "travel.md") -> ScannedFile:
    return ScannedFile(
        path=Path(name),
        relative_path=name,
        size=1,
        mtime=0.0,
        sha256="abc",
    )


class FakeRepo:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str, dict]] = []

    async def update_task(self, tenant_id: str, task_id: str, **fields) -> None:
        self.updates.append((tenant_id, task_id, fields))


class FakePipeline:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

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
        if self.error is not None:
            raise self.error


class BlockingPipeline:
    def __init__(self, release: asyncio.Event, started: list[str] | None = None) -> None:
        self._release = release
        self._started = started if started is not None else []

    async def run_task(
        self,
        task_id,
        files,
        *,
        tenant_id: str = "default",
        force=False,
        prune=False,
        progress=None,
    ):
        self._started.append(task_id)
        await self._release.wait()


async def test_submit_runs_pipeline_and_marks_running():
    repo = FakeRepo()
    pipeline = FakePipeline()
    worker = IngestionWorker(pipeline, repo)

    worker.submit("default", "task-1", [_file()])
    await worker.wait_idle()

    assert pipeline.calls[0]["tenant_id"] == "default"
    assert pipeline.calls[0]["task_id"] == "task-1"
    assert repo.updates[0][0] == "default"
    assert repo.updates[0][1] == "task-1"
    assert repo.updates[0][2]["status"] == "running"


async def test_submit_propagates_tenant_and_prune_flag():
    repo = FakeRepo()
    pipeline = FakePipeline()
    worker = IngestionWorker(pipeline, repo)

    worker.submit("sanya", "task-prune", [_file()], prune=True)
    await worker.wait_idle()

    assert pipeline.calls[0]["tenant_id"] == "sanya"
    assert pipeline.calls[0]["prune"] is True
    assert all(update[0] == "sanya" for update in repo.updates)


async def test_pipeline_failure_marks_task_failed():
    repo = FakeRepo()
    pipeline = FakePipeline(error=RuntimeError("milvus down"))
    worker = IngestionWorker(pipeline, repo)

    worker.submit("default", "task-2", [_file()])
    await worker.wait_idle()

    assert repo.updates[-1][0] == "default"
    assert repo.updates[-1][1] == "task-2"
    assert repo.updates[-1][2]["status"] == "failed"
    assert repo.updates[-1][2]["errors"][0]["error"] == "RuntimeError: milvus down"


async def test_is_active_tracks_submitted_task():
    repo = FakeRepo()
    release = asyncio.Event()

    worker = IngestionWorker(BlockingPipeline(release), repo)
    assert worker.is_active("task-1") is False

    worker.submit("default", "task-1", [_file()])
    await asyncio.sleep(0.05)
    assert worker.is_active("task-1") is True

    release.set()
    await worker.wait_idle()
    await asyncio.sleep(0)
    assert worker.is_active("task-1") is False


async def test_tasks_run_serially():
    repo = FakeRepo()
    release = asyncio.Event()
    started: list[str] = []

    worker = IngestionWorker(BlockingPipeline(release, started), repo)

    worker.submit("default", "task-a", [_file("a.md")])
    worker.submit("default", "task-b", [_file("b.md")])
    await asyncio.sleep(0.05)

    assert started == ["task-a"]

    release.set()
    await worker.wait_idle()

    assert started == ["task-a", "task-b"]
