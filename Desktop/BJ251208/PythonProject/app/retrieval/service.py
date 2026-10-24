from app.core.config import Settings
from app.retrieval.embedder import BGEM3Embedder
from app.retrieval.filters import build_filter
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.reranker import BGEReranker

SEARCH_MODES = ("dense", "sparse", "hybrid")


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: BGEM3Embedder | None = None,
        milvus: MilvusRepository | None = None,
        reranker: BGEReranker | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder or BGEM3Embedder(settings)
        self._milvus = milvus or MilvusRepository(settings)
        self._reranker = reranker or (BGEReranker(settings) if settings.reranker_enabled else None)

    @property
    def embedder(self) -> BGEM3Embedder:
        return self._embedder

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = 5,
        rerank: bool | None = None,
        city: str | None = None,
        content_type: str | None = None,
        scenic_name: str | None = None,
        route_name: str | None = None,
        hotel_name: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[dict]:
        if mode not in SEARCH_MODES:
            raise ValueError(f"unknown search mode: {mode}")

        dense, sparse = self._embedder.encode_query(query)
        filter_expr = build_filter(
            city=city,
            content_type=content_type,
            scenic_name=scenic_name,
            route_name=route_name,
            hotel_name=hotel_name,
            doc_ids=doc_ids,
        )
        use_rerank = self._resolve_rerank(rerank)
        limit = self._candidate_limit(top_k) if use_rerank else top_k
        hits = self._retrieve(
            dense,
            sparse,
            mode=mode,
            limit=limit,
            filter_expr=filter_expr,
        )
        if use_rerank:
            hits = self._reranker.rerank(query, hits, top_n=top_k)
        return hits

    def search_all(
        self,
        query: str,
        *,
        top_k: int = 5,
        rerank: bool = False,
        **filters,
    ) -> dict[str, list[dict]]:
        dense, sparse = self._embedder.encode_query(query)
        filter_expr = build_filter(**filters)
        use_rerank = self._resolve_rerank(rerank)
        limit = self._candidate_limit(top_k) if use_rerank else top_k

        results: dict[str, list[dict]] = {}
        for mode in SEARCH_MODES:
            hits = self._retrieve(
                dense,
                sparse,
                mode=mode,
                limit=limit,
                filter_expr=filter_expr,
            )
            if use_rerank:
                hits = self._reranker.rerank(query, hits, top_n=top_k)
            results[mode] = hits
        return results

    def _retrieve(
        self,
        dense: list[float],
        sparse: dict[int, float],
        *,
        mode: str,
        limit: int,
        filter_expr: str,
    ) -> list[dict]:
        if mode == "dense" or (mode != "dense" and not sparse):
            hits = self._milvus.search_dense(dense, limit=limit, filter_expr=filter_expr)
        elif mode == "sparse":
            hits = self._milvus.search_sparse(sparse, limit=limit, filter_expr=filter_expr)
        else:
            hits = self._milvus.search_hybrid(
                dense,
                sparse,
                limit=limit,
                dense_limit=max(self._settings.retrieval_dense_top_k, limit),
                sparse_limit=max(self._settings.retrieval_sparse_top_k, limit),
                rrf_k=self._settings.rrf_k,
                filter_expr=filter_expr,
            )
        return [_normalize_hit(hit) for hit in hits]

    def rerank_hits(self, query: str, hits: list[dict], *, top_n: int) -> list[dict]:
        if not hits:
            return []
        if not self._resolve_rerank(None):
            return hits[:top_n]
        return self._reranker.rerank(query, hits, top_n=top_n)

    def _resolve_rerank(self, rerank: bool | None) -> bool:
        use_rerank = self._settings.reranker_enabled if rerank is None else rerank
        if use_rerank and self._reranker is None:
            raise RuntimeError("reranker is disabled by configuration")
        return use_rerank

    def _candidate_limit(self, top_k: int) -> int:
        return max(
            top_k,
            self._settings.retrieval_dense_top_k,
            self._settings.retrieval_sparse_top_k,
        )

    def close(self) -> None:
        self._milvus.close()


def _normalize_hit(hit: dict) -> dict:
    entity = dict(hit.get("entity") or {})
    return {
        "chunk_id": hit.get("chunk_id") or entity.get("chunk_id"),
        "score": float(hit.get("distance", 0.0)),
        **entity,
    }
