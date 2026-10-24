import asyncio
import re
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pymongo import DESCENDING

from app.api.deps import (
    get_milvus,
    get_mongo,
    get_settings_dep,
    require_admin,
)
from app.core.config import Settings
from app.core.exceptions import AppError, NotFoundError
from app.repositories.mongo import DOCUMENTS, INGESTION_TASKS, MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.schemas.admin import (
    AdminChunk,
    AdminDocumentDetail,
    AdminDocumentListResponse,
    AdminDocumentSummary,
    AdminFeatures,
    AdminOverview,
)

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])

MAX_CHUNK_PREVIEW = 200
MAX_DOC_ID_LENGTH = 128


@router.get("/admin/overview", response_model=AdminOverview)
async def overview(
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
    milvus: Annotated[MilvusRepository, Depends(get_milvus)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> AdminOverview:
    db = mongo.db
    documents_by_status, total_documents = await _status_counts(db[DOCUMENTS])
    tasks_by_status, total_tasks = await _status_counts(db[INGESTION_TASKS])
    cities, content_types = await asyncio.gather(
        _field_counts(db[DOCUMENTS], "city"),
        _field_counts(db[DOCUMENTS], "content_type"),
    )

    chunks: int | None = None
    chunk_error = ""
    try:
        chunks = await asyncio.to_thread(milvus.count_chunks)
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
    status: str | None = Query(default=None, max_length=32),
    city: str | None = Query(default=None, max_length=64),
    content_type: str | None = Query(default=None, max_length=64),
    keyword: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> AdminDocumentListResponse:
    collection = mongo.db[DOCUMENTS]
    query = _document_query(
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
) -> AdminDocumentDetail:
    if not doc_id or len(doc_id) > MAX_DOC_ID_LENGTH:
        raise AppError("文档 ID 不合法", status_code=400, code="INVALID_DOC_ID")

    document = await mongo.db[DOCUMENTS].find_one({"doc_id": doc_id}, {"_id": 0})
    if document is None:
        raise NotFoundError("document not found")

    chunks: list[AdminChunk] = []
    chunk_error = ""
    try:
        rows = await asyncio.to_thread(
            milvus.list_document_chunks,
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


async def _status_counts(collection) -> tuple[dict[str, int], int]:
    cursor = await collection.aggregate(
        [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    )
    rows = await cursor.to_list(length=None)
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("_id") or "unknown"
        counts[str(key)] = int(row.get("count") or 0)
    return counts, sum(counts.values())


async def _field_counts(collection, field: str) -> dict[str, int]:
    cursor = await collection.aggregate(
        [
            {"$match": {"status": "imported"}},
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
    status: str | None,
    city: str | None,
    content_type: str | None,
    keyword: str | None,
) -> dict:
    query: dict = {}
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
