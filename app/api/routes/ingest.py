import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.deps import (
    get_ingest_repo,
    get_ingest_worker,
    get_settings_dep,
    require_role,
    target_tenant,
)
from app.core.config import Settings
from app.core.exceptions import AppError, NotFoundError
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.ingestion.scanner import (
    build_scanned_file,
    scan_tenant_directory,
)
from app.ingestion.task_repo import IngestionRepository, utc_now
from app.schemas.ingestion import (
    IngestionTaskListResponse,
    IngestionTaskSummary,
    IngestScanRequest,
    IngestStartResponse,
)
from app.workers.ingest_worker import IngestionWorker

router = APIRouter(tags=["ingestion"])

UPLOAD_CHUNK_SIZE = 1024 * 1024


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
        files = await asyncio.to_thread(scan_tenant_directory, settings, tenant_id)
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


@router.post("/ingest/upload", response_model=IngestStartResponse)
async def upload(
    file: Annotated[UploadFile, File(...)],
    repo: Annotated[IngestionRepository, Depends(get_ingest_repo)],
    worker: Annotated[IngestionWorker, Depends(get_ingest_worker)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    ctx: Annotated[TenantContext, Depends(require_role("editor"))],
) -> IngestStartResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    filename = Path(file.filename or "").name.strip()
    if not filename or filename in {".", ".."}:
        raise AppError("文件名不能为空", status_code=400, code="INVALID_FILENAME")

    suffix = Path(filename).suffix.lower()
    if suffix not in settings.upload_suffixes:
        raise AppError(
            f"不支持的文件类型: {suffix or 'unknown'}",
            status_code=400,
            code="UNSUPPORTED_FILE_TYPE",
        )

    upload_dir = settings.tenant_upload_path(tenant_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_upload_path(upload_dir, filename)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0

    try:
        with target.open("wb") as handle:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise AppError(
                        f"文件超过 {settings.max_upload_mb}MB 限制",
                        status_code=413,
                        code="FILE_TOO_LARGE",
                    )
                handle.write(chunk)
        if size == 0:
            raise AppError("上传文件为空", status_code=400, code="EMPTY_FILE")
        scanned = await asyncio.to_thread(
            build_scanned_file,
            target,
            settings.tenant_data_path(tenant_id),
        )
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    return await _queue(repo, worker, tenant_id, "upload", [scanned])


@router.get("/ingest/tasks", response_model=IngestionTaskListResponse)
async def list_tasks(
    repo: Annotated[IngestionRepository, Depends(get_ingest_repo)],
    worker: Annotated[IngestionWorker, Depends(get_ingest_worker)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
    status: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> IngestionTaskListResponse:
    tenant_id = target_tenant(ctx, allow_all=True)
    total, documents = await repo.list_tasks(
        tenant_id, status=status, limit=limit, offset=offset
    )
    return IngestionTaskListResponse(
        total=total,
        tasks=[
            _task_summary(document, stale=_is_stale(document, worker))
            for document in documents
        ],
    )


@router.get("/ingest/tasks/{task_id}", response_model=IngestionTaskSummary)
async def get_task(
    task_id: str,
    repo: Annotated[IngestionRepository, Depends(get_ingest_repo)],
    worker: Annotated[IngestionWorker, Depends(get_ingest_worker)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> IngestionTaskSummary:
    tenant_id = target_tenant(ctx, allow_all=True)
    document = await repo.get_task(tenant_id, task_id)
    if document is None:
        raise NotFoundError("task not found")
    return _task_summary(document, stale=_is_stale(document, worker))


@router.post(
    "/ingest/tasks/{task_id}/retry",
    response_model=IngestStartResponse,
)
async def retry_task(
    task_id: str,
    repo: Annotated[IngestionRepository, Depends(get_ingest_repo)],
    worker: Annotated[IngestionWorker, Depends(get_ingest_worker)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    ctx: Annotated[TenantContext, Depends(require_role("editor"))],
) -> IngestStartResponse:
    scope = target_tenant(ctx, allow_all=True)
    document = await repo.get_task(scope, task_id)
    if document is None:
        raise NotFoundError("task not found")

    task_tenant = str(document.get("tenant_id") or DEFAULT_TENANT_ID)
    if not ctx.is_global_admin and task_tenant != ctx.tenant_id:
        raise NotFoundError("task not found")

    interrupted = False
    if document.get("status") in {"queued", "running"}:
        if worker.is_active(task_id):
            raise AppError("任务仍在执行中", status_code=409, code="TASK_BUSY")
        interrupted = True
        await repo.update_task(
            task_tenant,
            task_id,
            status="failed",
            current_file="",
            finished_at=utc_now(),
            errors=[{"file": "", "error": "任务中断：服务重启后未恢复"}],
        )

    failed_files = sorted(
        {
            error["file"]
            for error in document.get("errors", [])
            if isinstance(error.get("file"), str) and error["file"]
        }
    )

    root = settings.tenant_data_path(task_tenant).resolve()
    files = []
    for relative_path in failed_files:
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        files.append(await asyncio.to_thread(build_scanned_file, path, root))

    if not files and interrupted and (document.get("task_type") or "scan") == "scan":
        try:
            files = await asyncio.to_thread(
                scan_tenant_directory, settings, task_tenant
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise AppError(
                str(exc), status_code=400, code="DATA_DIR_UNAVAILABLE"
            ) from exc
        except OSError as exc:
            raise AppError(
                f"无法扫描数据目录: {exc}",
                status_code=400,
                code="DATA_DIR_UNAVAILABLE",
            ) from exc

    if not files and failed_files:
        raise AppError(
            "失败文件已不存在",
            status_code=404,
            code="SOURCE_FILES_MISSING",
        )
    if not files:
        raise AppError("任务没有失败文件", status_code=400, code="NO_FAILED_FILES")
    return await _queue(
        repo, worker, task_tenant, "retry", files, force=bool(failed_files)
    )


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


def _is_stale(document: dict, worker: IngestionWorker) -> bool:
    if document.get("status") not in {"queued", "running"}:
        return False
    return not worker.is_active(str(document.get("task_id") or document["_id"]))


def _task_summary(document: dict, *, stale: bool = False) -> IngestionTaskSummary:
    return IngestionTaskSummary(
        task_id=str(document.get("task_id") or document["_id"]),
        tenant_id=document.get("tenant_id") or DEFAULT_TENANT_ID,
        task_type=document.get("task_type") or "scan",
        status=document.get("status") or "running",
        total=document.get("total") or 0,
        succeeded=document.get("succeeded") or 0,
        failed=document.get("failed") or 0,
        skipped=document.get("skipped") or 0,
        pruned=document.get("pruned") or 0,
        prune_failed=document.get("prune_failed") or 0,
        current_file=document.get("current_file") or "",
        stale=stale,
        errors=document.get("errors") or [],
        created_at=document.get("created_at"),
        started_at=document.get("started_at"),
        finished_at=document.get("finished_at"),
        duration_seconds=document.get("duration_seconds"),
    )


def _unique_upload_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise AppError("同名文件过多", status_code=409, code="FILENAME_CONFLICT")
