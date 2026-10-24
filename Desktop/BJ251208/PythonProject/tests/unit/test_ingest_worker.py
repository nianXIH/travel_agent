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
        self.updates: list[tuple[str, dict]] = []

    async def update_task(self, task_id: str, **fields) -> None:
        self.updates.append((task_id, fields))


class FakePipeline:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

    async def run_task(
        self,
        task_id: str,
        files,
        *,
        force: bool = False,
        progress=None,
    ) -> None:
        self.calls.append({"task_id": task_id, "files": list(files), "force": force})
        if self.error is not None:
            raise self.error


async def test_submit_runs_pipeline_and_marks_running():
    repo = FakeRepo()
    pipeline = FakePipeline()
    worker = IngestionWorker(pipeline, repo)

    worker.submit("task-1", [_file()])
    await worker.wait_idle()

    assert pipeline.calls[0]["task_id"] == "task-1"
    assert repo.updates[0][0] == "task-1"
    assert repo.updates[0][1]["status"] == "running"


async def test_pipeline_failure_marks_task_failed():
    repo = FakeRepo()
    pipeline = FakePipeline(error=RuntimeError("milvus down"))
    worker = IngestionWorker(pipeline, repo)

    worker.submit("task-2", [_file()])
    await worker.wait_idle()

    assert repo.updates[-1][1]["status"] == "failed"
    assert repo.updates[-1][1]["errors"][0]["error"] == "RuntimeError: milvus down"


async def test_tasks_run_serially():
    repo = FakeRepo()
    release = asyncio.Event()
    started: list[str] = []

    class BlockingPipeline:
        async def run_task(self, task_id, files, *, force=False, progress=None):
            started.append(task_id)
            await release.wait()

    worker = IngestionWorker(BlockingPipeline(), repo)

    worker.submit("task-a", [_file("a.md")])
    worker.submit("task-b", [_file("b.md")])
    await asyncio.sleep(0.05)

    assert started == ["task-a"]

    release.set()
    await worker.wait_idle()

    assert started == ["task-a", "task-b"]
