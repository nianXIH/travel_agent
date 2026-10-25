import asyncio
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import ValidationError

from app.chat.llm import ChatLLM, message_text, usage_dict
from app.chat.routing import CITY_ALIASES, KNOWN_CITIES
from app.core.config import Settings
from app.core.exceptions import AppError, LLMNotConfiguredError, UpstreamError
from app.core.prompts import (
    ITINERARY_SYSTEM_PROMPT,
    RECOMMENDATION_SYSTEM_PROMPT,
    build_context,
    build_itinerary_prompt,
    build_recommendation_prompt,
)
from app.retrieval.service import RetrievalService
from app.schemas.document import Citation
from app.schemas.planning import (
    ItineraryPlan,
    ItineraryRequest,
    ItineraryResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationSet,
)

ITINERARY_CONTENT_TYPES = ("景点介绍", "线路推荐", "酒店信息", "美食介绍", "交通指南")

QUERY_HINTS = {
    "景点介绍": "景点 玩法",
    "线路推荐": "行程 线路",
    "酒店信息": "住宿 酒店",
    "美食介绍": "美食 餐厅",
    "交通指南": "交通 怎么去",
}

CATEGORY_CONTENT_TYPES = {
    "景点": ("景点介绍",),
    "美食": ("美食介绍",),
    "酒店": ("酒店信息",),
    "线路": ("线路推荐",),
    "综合": ("景点介绍", "美食介绍", "酒店信息"),
}

INVALID_OUTPUT_MESSAGE = "生成结果格式不正确，请稍后重试"


class PlannerService:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        llm: ChatLLM,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._llm = llm

    async def plan(self, request: ItineraryRequest, tenant_id: str) -> ItineraryResponse:
        self._ensure_configured()
        started_at = time.perf_counter()
        city = resolve_city(request.destination)
        citations, retrieval_info = await self._collect_context(
            request.destination,
            city,
            ITINERARY_CONTENT_TYPES,
            tenant_id=tenant_id,
            per_type=self._settings.planner_context_per_type,
        )
        if not citations:
            raise AppError(
                "知识库中没有找到该目的地的资料，请尝试其他目的地",
                status_code=404,
                code="NO_CONTEXT",
            )

        response = await self._invoke(
            system=ITINERARY_SYSTEM_PROMPT,
            prompt=build_itinerary_prompt(request, build_context(citations)),
            error_message="生成行程失败，请稍后重试",
        )
        data = _parse_json_object(message_text(response))
        if data is None:
            raise UpstreamError(INVALID_OUTPUT_MESSAGE, code="INVALID_LLM_OUTPUT")

        try:
            plan = ItineraryPlan.model_validate(data)
        except ValidationError as exc:
            logger.warning("itinerary JSON validation failed: {}", exc)
            raise UpstreamError(INVALID_OUTPUT_MESSAGE, code="INVALID_LLM_OUTPUT") from exc

        if not plan.days:
            raise UpstreamError(INVALID_OUTPUT_MESSAGE, code="INVALID_LLM_OUTPUT")
        if len(plan.days) != request.days:
            logger.warning(
                "itinerary returned {} days, expected {}",
                len(plan.days),
                request.days,
            )

        total = len(citations)
        for day in plan.days:
            for stop in day.stops:
                stop.sources = _clean_sources(stop.sources, total)

        return ItineraryResponse(
            destination=request.destination,
            resolved_city=city,
            days=request.days,
            plan=plan,
            citations=[Citation.model_validate(item) for item in citations],
            retrieval=retrieval_info,
            usage=usage_dict(response),
            elapsed_ms=_elapsed_ms(started_at),
        )

    async def recommend(
        self, request: RecommendationRequest, tenant_id: str
    ) -> RecommendationResponse:
        self._ensure_configured()
        started_at = time.perf_counter()
        city = resolve_city(request.destination)
        content_types = CATEGORY_CONTENT_TYPES[request.category]
        citations, retrieval_info = await self._collect_context(
            request.destination,
            city,
            content_types,
            tenant_id=tenant_id,
            per_type=request.top_k,
        )
        if not citations:
            raise AppError(
                "知识库中没有找到该目的地的资料，请尝试其他目的地",
                status_code=404,
                code="NO_CONTEXT",
            )

        response = await self._invoke(
            system=RECOMMENDATION_SYSTEM_PROMPT,
            prompt=build_recommendation_prompt(request, build_context(citations)),
            error_message="生成推荐失败，请稍后重试",
        )
        data = _parse_json_object(message_text(response))
        if data is None:
            raise UpstreamError(INVALID_OUTPUT_MESSAGE, code="INVALID_LLM_OUTPUT")

        try:
            recommendations = RecommendationSet.model_validate(data)
        except ValidationError as exc:
            logger.warning("recommendation JSON validation failed: {}", exc)
            raise UpstreamError(INVALID_OUTPUT_MESSAGE, code="INVALID_LLM_OUTPUT") from exc

        if not recommendations.items:
            raise UpstreamError(INVALID_OUTPUT_MESSAGE, code="INVALID_LLM_OUTPUT")
        recommendations.items = recommendations.items[: request.top_k]

        total = len(citations)
        for item in recommendations.items:
            item.sources = _clean_sources(item.sources, total)

        return RecommendationResponse(
            destination=request.destination,
            resolved_city=city,
            category=request.category,
            recommendations=recommendations,
            citations=[Citation.model_validate(item) for item in citations],
            retrieval=retrieval_info,
            usage=usage_dict(response),
            elapsed_ms=_elapsed_ms(started_at),
        )

    async def _collect_context(
        self,
        destination: str,
        city: str | None,
        content_types: tuple[str, ...],
        *,
        tenant_id: str,
        per_type: int,
    ) -> tuple[list[dict], dict]:
        results = await asyncio.gather(
            *(
                self._search_content_type(
                    destination, city, content_type, tenant_id=tenant_id, per_type=per_type
                )
                for content_type in content_types
            )
        )

        citations: list[dict] = []
        seen: set[str] = set()
        categories: dict[str, dict] = {}
        degraded: list[str] = []
        cap = self._settings.planner_max_citations

        for content_type, hits, info in results:
            categories[content_type] = info
            if info["status"] == "error":
                degraded.append(content_type)
            if len(citations) >= cap:
                continue
            for hit in hits:
                chunk_id = str(hit.get("chunk_id") or "").strip()
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                item = dict(hit)
                item["chunk_id"] = chunk_id
                item["source_type"] = "kb"
                citations.append(item)
                if len(citations) >= cap:
                    break

        info = {
            "city": city,
            "categories": categories,
            "degraded": degraded,
            "total_citations": len(citations),
        }
        return citations, info

    async def _search_content_type(
        self,
        destination: str,
        city: str | None,
        content_type: str,
        *,
        tenant_id: str,
        per_type: int,
    ) -> tuple[str, list[dict], dict]:
        started_at = time.perf_counter()
        query = f"{destination} {QUERY_HINTS[content_type]}"
        error: str | None = None
        try:
            hits = await asyncio.to_thread(
                self._retrieval.search,
                query,
                tenant_id=tenant_id,
                mode="hybrid",
                top_k=per_type,
                city=city,
                content_type=content_type,
            )
            status = "ok" if hits else "empty"
        except asyncio.CancelledError:
            raise
        except ValueError:
            # Tenant/filter validation errors signal a programming fault; degrading
            # would turn them into a misleading empty-context 404, so fail closed.
            raise
        except Exception as exc:
            logger.warning(
                "planner retrieval failed for {}: {}",
                content_type,
                type(exc).__name__,
            )
            hits = []
            status = "error"
            error = type(exc).__name__

        info: dict[str, Any] = {
            "status": status,
            "hits": len(hits),
            "elapsed_ms": _elapsed_ms(started_at),
        }
        if error is not None:
            info["error"] = error
        return content_type, hits, info

    async def _invoke(self, *, system: str, prompt: str, error_message: str):
        try:
            return await self._llm.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)],
                temperature=self._settings.planner_temperature,
                max_tokens=self._settings.planner_max_tokens,
            )
        except asyncio.CancelledError:
            raise
        except AppError:
            raise
        except Exception as exc:
            logger.exception("planner LLM call failed")
            raise UpstreamError(error_message) from exc

    def _ensure_configured(self) -> None:
        if not self._llm.configured:
            raise LLMNotConfiguredError()


def resolve_city(destination: str) -> str | None:
    for city in KNOWN_CITIES:
        if city in destination:
            return city
    for alias, city in CITY_ALIASES.items():
        if alias in destination:
            return city
    return None


def _clean_sources(sources: list[int], total: int) -> list[int]:
    cleaned: list[int] = []
    for value in sources:
        index = int(value)
        if 1 <= index <= total and index not in cleaned:
            cleaned.append(index)
    return cleaned


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
