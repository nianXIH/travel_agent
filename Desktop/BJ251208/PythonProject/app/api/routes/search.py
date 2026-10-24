import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_mongo, get_retrieval, get_settings_dep
from app.core.config import Settings
from app.core.exceptions import AppError
from app.repositories.mongo import DOCUMENTS, MongoRepository
from app.retrieval.service import RetrievalService
from app.schemas.document import CatalogResponse, Citation
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    retrieval: Annotated[RetrievalService, Depends(get_retrieval)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> SearchResponse:
    started_at = time.perf_counter()
    try:
        hits = await asyncio.to_thread(
            retrieval.search,
            payload.query,
            mode=payload.mode,
            top_k=payload.top_k,
            rerank=payload.rerank,
            city=payload.city,
            content_type=payload.content_type,
            scenic_name=payload.scenic_name,
            route_name=payload.route_name,
            hotel_name=payload.hotel_name,
        )
    except RuntimeError as exc:
        raise AppError(str(exc), status_code=503, code="RETRIEVAL_UNAVAILABLE") from exc
    except Exception as exc:
        raise AppError(
            "检索服务暂时不可用",
            status_code=503,
            code="RETRIEVAL_UNAVAILABLE",
        ) from exc

    reranked = settings.reranker_enabled if payload.rerank is None else payload.rerank
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        reranked=reranked,
        elapsed_ms=int((time.perf_counter() - started_at) * 1000),
        citations=[Citation.model_validate(hit) for hit in hits],
    )


@router.get("/catalog", response_model=CatalogResponse)
async def catalog(
    mongo: Annotated[MongoRepository, Depends(get_mongo)],
) -> CatalogResponse:
    pipeline = [
        {"$match": {"status": "imported"}},
        {
            "$group": {
                "_id": {"city": "$city", "content_type": "$content_type"},
                "count": {"$sum": 1},
            }
        },
    ]
    cursor = await mongo.db[DOCUMENTS].aggregate(pipeline)
    rows = await cursor.to_list(length=None)

    city_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for row in rows:
        key = row["_id"]
        count = int(row["count"])
        city = key.get("city") or "未知"
        content_type = key.get("content_type") or "其他"
        city_counts[city] = city_counts.get(city, 0) + count
        type_counts[content_type] = type_counts.get(content_type, 0) + count

    return CatalogResponse(
        cities=sorted(city_counts),
        content_types=sorted(type_counts),
        document_count=sum(city_counts.values()),
        city_counts=city_counts,
        content_type_counts=type_counts,
    )
