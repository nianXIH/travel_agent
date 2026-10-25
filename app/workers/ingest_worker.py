import asyncio
from collections.abc import Sequence

from loguru import logger

from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import ScannedFile
from app.ingestion.task_repo import IngestionRepository, utc_now


class IngestionWorker:
    """Runs ingestion tasks serially so GPU-bound embedding never overlaps."""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        repo: IngestionRepository,
    ) -> None:
        self._pipeline = pipeline
        self._repo = repo
        self._semaphore = asyncio.Semaphore(1)
        self._tasks: set[asyncio.Task] = set()
        self._active: set[str] = set()

    def is_active(self, task_id: str) -> bool:
        return task_id in self._active

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

    async def wait_idle(self) -> None:
        pending = list(self._tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def shutdown(self, *, grace_seconds: float = 10.0) -> None:
        pending = list(self._tasks)
        if not pending:
            return
        _, still_running = await asyncio.wait(pending, timeout=grace_seconds)
        for task in still_running:
            task.cancel()
        if still_running:
            await asyncio.gather(*still_running, return_exceptions=True)

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
