import asyncio
import re
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pymongo import DESCENDING

from app.api.deps import (
    get_milvus,
    get_mongo,
    get_settings_dep,
    get_tenant_registry,
    require_admin,
)
from app.core.config import Settings
from app.core.exceptions import AppError, NotFoundError
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext, TenantRegistry
from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS, MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.schemas.admin import (
    AdminChunk,
    AdminDocumentDetail,
    AdminDocumentListResponse,
    AdminDocumentSummary,
    AdminFeatures,
    AdminOverview,
    AdminTenant,
    AdminTenantListResponse,
)

router = APIRouter(tags=["admin"])

MAX_CHUNK_PREVIEW = 200
MAX_DOC_ID_LENGTH = 128


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
    except Exception as exc:
        chunk_error = f"{type(exc).__name__}: {exc}"

    return AdminOverview(
        app_name=settings.app_name,
        app_env=settings.app_env,
        documents_by_status=documents_by_status,
        total_documents=total_documents,
        tasks_by_status=tasks_by_status,
        total_tasks=total_tasks,
        cities=cities,
        content_types=content_types,
        chunks=chunks,
        chunk_error=chunk_error,
        llm_configured=bool(settings.llm_api_key),
        features=AdminFeatures(
            reranker=settings.reranker_enabled,
            hybrid=settings.hybrid_enabled,
            web_search=settings.web_search_enabled,
        ),
    )


@router.get("/admin/documents", response_model=AdminDocumentListResponse)
async def list_documents(
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
    ctx: Annotated[TenantContext, Depends(require_admin)],
    registry: Annotated[TenantRegistry, Depends(get_tenant_registry)],
    tenant_id: str | None = Query(default=None, max_length=64),
    status: str | None = Query(default=None, max_length=32),
    city: str | None = Query(default=None, max_length=64),
    content_type: str | None = Query(default=None, max_length=64),
    keyword: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AdminDocumentListResponse:
    tenant_scope = _resolve_admin_tenant(ctx, tenant_id, registry)
    collection = mongo.db[DOCUMENTS]
    query = _document_query(
        tenant_id=tenant_scope,
        status=status,
        city=city,
        content_type=content_type,
        keyword=keyword,
    )
    total = await collection.count_documents(query)
    cursor = (
        collection.find(query, {"_id": 0})
        .sort("updated_at", DESCENDING)
        .skip(offset)
        .limit(limit)
    )
    documents = await cursor.to_list(length=limit)
    return AdminDocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[AdminDocumentSummary.model_validate(document) for document in documents],
    )


@router.get("/admin/documents/{doc_id}", response_model=AdminDocumentDetail)
async def get_document(
    doc_id: str,
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
    milvus: Annotated[MilvusRepository, Depends(get_milvus)],
    ctx: Annotated[TenantContext, Depends(require_admin)],
    registry: Annotated[TenantRegistry, Depends(get_tenant_registry)],
    tenant_id: str | None = Query(default=None, max_length=64),
) -> AdminDocumentDetail:
    if not doc_id or len(doc_id) > MAX_DOC_ID_LENGTH:
        raise AppError("文档 ID 不合法", status_code=400, code="INVALID_DOC_ID")

    tenant_scope = _resolve_admin_tenant(ctx, tenant_id, registry)
    query: dict = {"doc_id": doc_id}
    if tenant_scope is not None:
        query["tenant_id"] = tenant_scope
    document = await mongo.db[DOCUMENTS].find_one(query, {"_id": 0})
    if document is None:
        raise NotFoundError("document not found")

    document_tenant = document.get("tenant_id") or DEFAULT_TENANT_ID
    chunks: list[AdminChunk] = []
    chunk_error = ""
    try:
        rows = await asyncio.to_thread(
            milvus.list_document_chunks,
            document_tenant,
            doc_id,
            MAX_CHUNK_PREVIEW,
        )
        chunks = sorted(
            (AdminChunk.model_validate(row) for row in rows),
            key=lambda chunk: chunk.chunk_index,
        )
    except Exception as exc:
        chunk_error = f"{type(exc).__name__}: {exc}"

    summary = AdminDocumentSummary.model_validate(document)
    return AdminDocumentDetail(
        **summary.model_dump(),
        chunks=chunks,
        chunk_error=chunk_error,
    )


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


async def _status_counts(collection, tenant_id: str | None) -> tuple[dict[str, int], int]:
    pipeline: list[dict] = []
    if tenant_id is not None:
        pipeline.append({"$match": {"tenant_id": tenant_id}})
    pipeline.append({"$group": {"_id": "$status", "count": {"$sum": 1}}})
    cursor = await collection.aggregate(pipeline)
    rows = await cursor.to_list(length=None)
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("_id") or "unknown"
        counts[str(key)] = int(row.get("count") or 0)
    return counts, sum(counts.values())


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
    rows = await cursor.to_list(length=None)
    return {
        str(row["_id"]): int(row.get("count") or 0)
        for row in rows
        if row.get("_id")
    }


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
    if status:
        query["status"] = status
    if city:
        query["city"] = city
    if content_type:
        query["content_type"] = content_type

    text = (keyword or "").strip()
    if text:
        pattern = re.escape(text)
        query["$or"] = [
            {"title": {"$regex": pattern, "$options": "i"}},
            {"source_file": {"$regex": pattern, "$options": "i"}},
            {"source_path": {"$regex": pattern, "$options": "i"}},
            {"city": {"$regex": pattern, "$options": "i"}},
        ]
    return query
