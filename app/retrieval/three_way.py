import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from loguru import logger

from app.core.config import Settings
from app.retrieval.fusion import rrf_fuse
from app.retrieval.hyde import HyDEGenerator
from app.retrieval.service import RetrievalService
from app.retrieval.web_search import WebSearchClient, WebSearchResult


@dataclass
class RetrievalOutcome:
    citations: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)


StageCallback = Callable[[str], Awaitable[None]]


class ThreeWayRetriever:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        hyde: HyDEGenerator,
        web: WebSearchClient,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._hyde = hyde
        self._web = web

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        city: str | None = None,
        content_type: str | None = None,
        on_stage: StageCallback | None = None,
    ) -> RetrievalOutcome:
        started_at = time.perf_counter()
        candidate_k = max(top_k, self._settings.retrieval_candidate_top_k)

        await _stage(on_stage, "retrieving")
        web_task = asyncio.create_task(self._search_web(query))
        await _stage(on_stage, "web_searching")
        hybrid_task = asyncio.create_task(
            self._search_local(query, "hybrid", candidate_k, city, content_type)
        )

        await _stage(on_stage, "hyde_generating")
        hyde_started = time.perf_counter()
        hyde_text, hyde_status = await self._generate_hyde(query, city, content_type)
        hybrid_hits, hybrid_info = await hybrid_task

        hyde_hits: list[dict] = []
        if hyde_text:
            hyde_hits, hyde_search_info = await self._search_local(
                hyde_text, "dense", candidate_k, city, content_type
            )
            hyde_status = hyde_search_info["status"]
        hyde_info = {
            "status": hyde_status,
            "hits": len(hyde_hits),
            "elapsed_ms": _elapsed_ms(hyde_started),
        }

        local_union = _fuse_local(
            hybrid_hits,
            hyde_hits,
            settings=self._settings,
            limit=candidate_k,
        )

        local_ranked: list[dict] = []
        rerank_failed = False
        if local_union:
            await _stage(on_stage, "reranking")
            local_ranked, rerank_failed = await self._rerank(query, local_union, top_k)

        web_result = await web_task
        web_info = {
            "status": web_result.status,
            "hits": len(web_result.hits),
            "elapsed_ms": web_result.elapsed_ms,
        }
        if web_result.error:
            web_info["error"] = web_result.error

        await _stage(on_stage, "fusing")
        final = _fuse_final(
            local_ranked,
            web_result.hits,
            settings=self._settings,
            top_k=top_k,
        )

        degraded: list[str] = []
        if hybrid_info["status"] == "error":
            degraded.append("hybrid")
        if hyde_status == "error":
            degraded.append("hyde")
        if web_result.status == "unavailable":
            degraded.append("web")
        if rerank_failed:
            degraded.append("reranker")

        info = {
            "hybrid": hybrid_info,
            "hyde": hyde_info,
            "web": web_info,
            "fusion": {
                "strategy": "rrf",
                "k": self._settings.rrf_k,
                "local_candidates": len(local_union),
                "local_ranked": len(local_ranked),
                "final": len(final),
                "web_in_final": sum(1 for item in final if item.get("source_type") == "web"),
                "web_min": self._settings.web_search_min_citations,
                "weights": {
                    "hybrid": 1.0,
                    "hyde": self._settings.hyde_weight,
                    "local": 1.0,
                    "web": self._settings.web_search_weight,
                },
            },
            "degraded": degraded,
            "elapsed_ms": _elapsed_ms(started_at),
        }
        return RetrievalOutcome(citations=final, info=info)

    async def aclose(self) -> None:
        await self._web.aclose()

    async def _search_local(
        self,
        text: str,
        mode: str,
        top_k: int,
        city: str | None,
        content_type: str | None,
    ) -> tuple[list[dict], dict]:
        started_at = time.perf_counter()
        try:
            hits = await asyncio.to_thread(
                self._retrieval.search,
                text,
                mode=mode,
                top_k=top_k,
                rerank=False,
                city=city,
                content_type=content_type,
            )
            status = "ok" if hits else "empty"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("{} retrieval failed: {}", mode, type(exc).__name__)
            hits = []
            status = "error"
        return hits, {
            "status": status,
            "hits": len(hits),
            "elapsed_ms": _elapsed_ms(started_at),
        }

    async def _generate_hyde(
        self,
        query: str,
        city: str | None,
        content_type: str | None,
    ) -> tuple[str, str]:
        if not self._settings.hyde_enabled:
            return "", "disabled"
        try:
            text = await asyncio.wait_for(
                self._hyde.generate(query, city=city, content_type=content_type),
                timeout=self._settings.llm_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("HyDE generation failed: {}", type(exc).__name__)
            return "", "error"
        return text, "ok" if text else "empty"

    async def _rerank(
        self,
        query: str,
        hits: list[dict],
        top_n: int,
    ) -> tuple[list[dict], bool]:
        try:
            ranked = await asyncio.to_thread(
                self._retrieval.rerank_hits,
                query,
                hits,
                top_n=top_n,
            )
            return ranked, False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("rerank failed: {}", type(exc).__name__)
            return hits[:top_n], True

    async def _search_web(self, query: str) -> WebSearchResult:
        started_at = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._web.search(query, limit=self._settings.web_search_top_k),
                timeout=self._settings.web_search_timeout_seconds + 1,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning("web search timed out")
            return WebSearchResult(
                status="unavailable",
                error="TimeoutError",
                elapsed_ms=_elapsed_ms(started_at),
            )
        except Exception as exc:
            logger.warning("web search failed: {}", type(exc).__name__)
            return WebSearchResult(
                status="unavailable",
                error=type(exc).__name__,
                elapsed_ms=_elapsed_ms(started_at),
            )
        if result.elapsed_ms == 0:
            result.elapsed_ms = _elapsed_ms(started_at)
        return result


def _fuse_local(
    hybrid_hits: list[dict],
    hyde_hits: list[dict],
    *,
    settings: Settings,
    limit: int,
) -> list[dict]:
    lists: list[list[dict]] = []
    weights: list[float] = []
    names: list[str] = []
    if hybrid_hits:
        lists.append(hybrid_hits)
        weights.append(1.0)
        names.append("hybrid")
    if hyde_hits:
        lists.append(hyde_hits)
        weights.append(settings.hyde_weight)
        names.append("hyde")
    if not lists:
        return []
    return rrf_fuse(
        lists,
        weights=weights,
        k=settings.rrf_k,
        limit=limit,
        source_names=names,
    )


def _fuse_final(
    local_ranked: list[dict],
    web_hits: list[dict],
    *,
    settings: Settings,
    top_k: int,
) -> list[dict]:
    lists: list[list[dict]] = []
    weights: list[float] = []
    if local_ranked:
        lists.append(local_ranked)
        weights.append(1.0)
    if web_hits:
        lists.append(web_hits)
        weights.append(settings.web_search_weight)
    if not lists:
        return []
    web_cap = settings.web_search_max_citations if local_ranked else top_k
    selected = rrf_fuse(
        lists,
        weights=weights,
        k=settings.rrf_k,
        limit=top_k,
        max_from={"web": web_cap},
    )
    if local_ranked and web_hits:
        minimum = min(settings.web_search_min_citations, max(0, top_k - 1))
        selected = _ensure_web_minimum(
            selected,
            web_hits,
            settings=settings,
            minimum=minimum,
            top_k=top_k,
        )
    return selected


def _ensure_web_minimum(
    selected: list[dict],
    web_hits: list[dict],
    *,
    settings: Settings,
    minimum: int,
    top_k: int,
) -> list[dict]:
    if minimum <= 0:
        return selected
    web_count = sum(1 for item in selected if item.get("source_type") == "web")
    if web_count >= minimum:
        return selected

    working = list(selected)
    selected_ids = {item.get("chunk_id") for item in working}
    available = [hit for hit in web_hits if hit.get("chunk_id") not in selected_ids]
    to_add = min(minimum - web_count, len(available))

    for _ in range(to_add):
        for index in range(len(working) - 1, -1, -1):
            if working[index].get("source_type") != "web":
                working.pop(index)
                break

    web_rank = {hit.get("chunk_id"): rank for rank, hit in enumerate(web_hits, start=1)}
    for hit in available[:to_add]:
        item = dict(hit)
        rank = web_rank.get(hit.get("chunk_id"), 1)
        item["fusion_score"] = settings.web_search_weight / (settings.rrf_k + rank)
        working.append(item)

    working.sort(key=lambda item: item.get("fusion_score") or 0.0, reverse=True)
    return working[:top_k]


async def _stage(callback: StageCallback | None, name: str) -> None:
    if callback is not None:
        await callback(name)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
