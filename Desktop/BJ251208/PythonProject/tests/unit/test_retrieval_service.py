"""Service-level tenant checks for RetrievalService.search_all.

Fakes replace the embedder and Milvus so the test records the exact filter
expressions without loading a model or connecting to a running Milvus.
"""

from app.core.config import Settings
from app.retrieval.service import RetrievalService


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        self.queries.append(query)
        return [0.1, 0.2], {1: 0.5}


class FakeMilvus:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def _record(self, mode: str, filter_expr: str) -> list[dict]:
        self.calls.append({"mode": mode, "filter": filter_expr})
        return []

    def search_dense(self, dense, *, limit: int, filter_expr: str) -> list[dict]:
        return self._record("dense", filter_expr)

    def search_sparse(self, sparse, *, limit: int, filter_expr: str) -> list[dict]:
        return self._record("sparse", filter_expr)

    def search_hybrid(
        self,
        dense,
        sparse,
        *,
        limit: int,
        dense_limit: int,
        sparse_limit: int,
        rrf_k: int,
        filter_expr: str,
    ) -> list[dict]:
        return self._record("hybrid", filter_expr)


def _service() -> tuple[RetrievalService, FakeEmbedder, FakeMilvus]:
    embedder = FakeEmbedder()
    milvus = FakeMilvus()
    service = RetrievalService(
        Settings(reranker_enabled=False, _env_file=None),
        embedder=embedder,
        milvus=milvus,
    )
    return service, embedder, milvus


def test_search_all_scopes_every_mode_to_the_same_tenant():
    service, embedder, milvus = _service()

    results = service.search_all("西湖", tenant_id="sanya", top_k=5)

    assert set(results) == {"dense", "sparse", "hybrid"}
    assert [call["mode"] for call in milvus.calls] == ["dense", "sparse", "hybrid"]
    assert {call["filter"] for call in milvus.calls} == {'tenant_id == "sanya"'}
    assert embedder.queries == ["西湖"]
