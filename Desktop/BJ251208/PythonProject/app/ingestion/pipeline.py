import asyncio
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, TENANTS_DIR
from app.ingestion.chunker import chunk_sections, parse_sections
from app.ingestion.loader import load_text
from app.ingestion.metadata import make_doc_id, parse_document_meta
from app.ingestion.scanner import (
    ScannedFile,
    scan_directory,
    scan_tenant_directory,
)
from app.ingestion.task_repo import IngestionRepository, utc_now
from app.repositories.mongo import MongoRepository
from app.retrieval.embedder import BGEM3Embedder
from app.retrieval.milvus_repo import DENSE_FIELD, SPARSE_FIELD, MilvusRepository


@dataclass
class IngestionSummary:
    task_id: str
    total: int
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    pruned: int = 0
    prune_failed: int = 0
    chunks_written: int = 0
    duration_seconds: float = 0.0
    errors: list[dict] = field(default_factory=list)
    stale_files: list[str] = field(default_factory=list)
    dry_run: bool = False


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: BGEM3Embedder | None = None,
        mongo: MongoRepository | None = None,
        milvus: MilvusRepository | None = None,
        repo: IngestionRepository | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder or BGEM3Embedder(settings)
        # Only close clients this pipeline constructed; injected clients are
        # owned (and closed) by the caller.
        self._owns_mongo = mongo is None
        self._owns_milvus = milvus is None
        self._mongo = mongo or MongoRepository(settings)
        self._milvus = milvus or MilvusRepository(settings)
        self._repo = repo or IngestionRepository(self._mongo)

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
                files = await asyncio.to_thread(scan_tenant_directory, self._settings, tenant_id)
            else:
                files = await asyncio.to_thread(
                    scan_directory,
                    root,
                    exclude_dirs=self._exclude_dirs(tenant_id),
                )
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
                raise ValueError("refusing to prune: scan returned no files (empty scan)")
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
            if self._owns_mongo:
                await self._mongo.close()
            if self._owns_milvus:
                self._milvus.close()

    def _exclude_dirs(self, tenant_id: str) -> set[Path]:
        # Other tenants live under DATA_DIR/tenants; the default tenant scans
        # DATA_DIR itself and must never adopt their documents.
        if tenant_id != DEFAULT_TENANT_ID:
            return set()
        return {self._settings.data_path / TENANTS_DIR}

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
                    await asyncio.to_thread(self._milvus.delete_document_chunks, tenant_id, doc_id)
                await self._repo.mark_document_deleted(
                    tenant_id,
                    source_path=source_path,
                    doc_id=doc_id,
                )
                summary.pruned += 1
                progress(f"prune {source_path}")
            except Exception as exc:
                summary.prune_failed += 1
                summary.errors.append({"file": source_path, "error": f"PruneError: {exc}"})
                progress(f"FAILED to prune {source_path}: {exc}")

    async def _ensure_complete_scan(
        self,
        files: Sequence[ScannedFile],
        tenant_id: str,
    ) -> None:
        scanned = {file.relative_path for file in files}
        if not scanned:
            raise ValueError("refusing to prune: scan returned no files (empty scan)")
        expected = await asyncio.to_thread(scan_tenant_directory, self._settings, tenant_id)
        expected_paths = {file.relative_path for file in expected}
        if scanned != expected_paths:
            raise ValueError(
                "refusing to prune: task files do not match a complete data directory scan"
            )

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

        # Always derive the id from the tenant and relative path: Mongo records,
        # chunk ids, stale deletion and failure records must agree. Legacy
        # un-scoped ids are rewritten/re-ingested by the migration task.
        doc_id = make_doc_id(tenant_id, file.relative_path)
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
                    "content": chunk.text,
                    "content_type": meta.content_type,
                    "city": meta.city,
                    "scenic_name": meta.scenic_name,
                    "route_name": meta.route_name,
                    "hotel_name": meta.hotel_name,
                    "restaurant_name": meta.restaurant_name,
                    "source_file": file.path.name,
                    "source_path": file.relative_path,
                    "section_path": chunk.section_path,
                    "chunk_index": chunk.index,
                    "extra": meta.extra,
                    "updated_at": now,
                    DENSE_FIELD: embedding.dense[index],
                    SPARSE_FIELD: embedding.sparse[index],
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
                    "source_path": file.relative_path,
                    "source_file": file.path.name,
                    "title": meta.title,
                    "sha256": file.sha256,
                    "content_type": meta.content_type,
                    "city": meta.city,
                    "status": "imported",
                    "size": file.size,
                    "mtime": file.mtime,
                    "chunk_count": len(rows),
                    "imported_at": utc_now(),
                    "error": "",
                },
            )
            summary.succeeded += 1
            summary.chunks_written += len(rows)
            progress(
                f"[{position}/{total}] import {file.relative_path} "
                f"chunks={len(rows)} city={meta.city} type={meta.content_type}"
            )
        except Exception as exc:
            summary.failed += 1
            error = {"file": file.relative_path, "error": f"{type(exc).__name__}: {exc}"}
            summary.errors.append(error)
            await self._repo.mark_document_failed(
                tenant_id,
                source_path=file.relative_path,
                doc_id=doc_id,
                source_file=file.path.name,
                error=error["error"],
            )
            progress(f"[{position}/{total}] FAILED {file.relative_path}: {error['error']}")


def _task_status(summary: IngestionSummary) -> str:
    failures = summary.failed + summary.prune_failed
    if failures == 0:
        return "success"
    if summary.succeeded == 0:
        return "failed"
    return "partial"
