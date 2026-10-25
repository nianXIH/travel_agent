import pytest

from app.core.config import Settings
from app.retrieval.filters import build_filter
from app.retrieval.three_way import ThreeWayRetriever
from app.retrieval.web_search import WebSearchResult


def _kb(chunk_id: str, **extra) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": chunk_id,
        "source_type": "kb",
        **extra,
    }


def _web(index: int) -> dict:
    return {
        "chunk_id": f"web:{index}",
        "content": f"web {index}",
        "source_type": "web",
        "url": f"https://example.com/{index}",
        "retrieval_sources": ["web"],
    }


class FakeRetrieval:
    def __init__(
        self,
        *,
        hybrid: list[dict] | None = None,
        hyde: list[dict] | None = None,
        hybrid_error: Exception | None = None,
        rerank_error: Exception | None = None,
    ) -> None:
        self.hybrid = hybrid or []
        self.hyde = hyde or []
        self.hybrid_error = hybrid_error
        self.rerank_error = rerank_error
        self.search_calls: list[dict] = []
        self.rerank_calls: list[dict] = []

    def search(
        self,
        text: str,
        *,
        mode: str,
        top_k: int,
        rerank: bool,
        city,
        content_type,
        tenant_id: str,
    ):
        # Mirror RetrievalService: a blank tenant fails before any Milvus call.
        build_filter(tenant_id=tenant_id)
        self.search_calls.append(
            {"text": text, "mode": mode, "top_k": top_k, "tenant_id": tenant_id}
        )
        if mode == "hybrid":
            if self.hybrid_error is not None:
                raise self.hybrid_error
            return list(self.hybrid)
        return list(self.hyde)

    def rerank_hits(self, query: str, hits: list[dict], *, top_n: int):
        self.rerank_calls.append({"query": query, "hits": hits, "top_n": top_n})
        if self.rerank_error is not None:
            raise self.rerank_error
        return list(reversed(hits))[:top_n]


class FakeHyDE:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    async def generate(self, query: str, *, city=None, content_type=None):
        self.calls.append({"query": query, "city": city, "content_type": content_type})
        if self.error is not None:
            raise self.error
        return self.text


class FakeWeb:
    def __init__(
        self,
        hits: list[dict] | None = None,
        *,
        error: Exception | None = None,
        status: str = "ok",
    ) -> None:
        self.hits = hits or []
        self.error = error
        self.status = status
        self.calls: list[str] = []
        self.closed = False

    async def search(self, query: str, *, limit: int | None = None) -> WebSearchResult:
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return WebSearchResult(hits=self.hits, status=self.status if self.hits else "empty")

    async def aclose(self) -> None:
        self.closed = True


def _settings(**overrides) -> Settings:
    values = {
        "reranker_enabled": True,
        "hyde_enabled": True,
        "web_search_enabled": True,
        "web_search_top_k": 5,
        "web_search_max_citations": 2,
        "retrieval_candidate_top_k": 20,
    }
    values.update(overrides)
    return Settings(**values)


async def test_combines_local_and_web_results():
    retrieval = FakeRetrieval(hybrid=[_kb("c1")], hyde=[_kb("c2")])
    hyde = FakeHyDE("假设性资料")
    web = FakeWeb([_web(1)])
    retriever = ThreeWayRetriever(_settings(), retrieval, hyde, web)

    outcome = await retriever.search(
        "西湖怎么玩", top_k=5, city="杭州", tenant_id="default"
    )

    assert [item["chunk_id"] for item in outcome.citations] == ["c2", "c1", "web:1"]
    assert retrieval.search_calls[0]["tenant_id"] == "default"
    assert {call["tenant_id"] for call in retrieval.search_calls} == {"default"}
    assert outcome.info["hybrid"]["status"] == "ok"
    assert outcome.info["hyde"]["status"] == "ok"
    assert outcome.info["web"]["status"] == "ok"
    assert retrieval.rerank_calls[0]["query"] == "西湖怎么玩"
    assert hyde.calls[0]["city"] == "杭州"


async def test_web_failure_degrades_to_local():
    retrieval = FakeRetrieval(hybrid=[_kb("c1")], hyde=[_kb("c2")])
    web = FakeWeb(error=RuntimeError("mcp down"))
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE("资料"), web)

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert all(item["source_type"] == "kb" for item in outcome.citations)
    assert outcome.info["web"]["status"] == "unavailable"
    assert "web" in outcome.info["degraded"]


async def test_hyde_failure_degrades_to_hybrid():
    retrieval = FakeRetrieval(hybrid=[_kb("c1")], hyde=[])
    web = FakeWeb([_web(1)])
    retriever = ThreeWayRetriever(
        _settings(), retrieval, FakeHyDE(error=RuntimeError("llm down")), web
    )

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert {"c1", "web:1"} <= {item["chunk_id"] for item in outcome.citations}
    assert outcome.info["hyde"]["status"] == "error"
    assert "hyde" in outcome.info["degraded"]


async def test_hybrid_failure_still_uses_hyde_and_web():
    retrieval = FakeRetrieval(hybrid=[], hyde=[_kb("c2")], hybrid_error=RuntimeError("milvus down"))
    web = FakeWeb([_web(1)])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE("资料"), web)

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert {"c2", "web:1"} <= {item["chunk_id"] for item in outcome.citations}
    assert outcome.info["hybrid"]["status"] == "error"
    assert "hybrid" in outcome.info["degraded"]


async def test_all_paths_empty_returns_empty():
    retrieval = FakeRetrieval(hybrid=[], hyde=[])
    web = FakeWeb([])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE(""), web)

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert outcome.citations == []
    assert retrieval.rerank_calls == []


async def test_web_citations_are_capped_when_local_results_exist():
    retrieval = FakeRetrieval(
        hybrid=[_kb(f"c{index}") for index in range(1, 6)],
        hyde=[],
    )
    web = FakeWeb([_web(index) for index in range(1, 6)])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE(""), web)

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert len(outcome.citations) == 5
    web_in_final = [item for item in outcome.citations if item.get("source_type") == "web"]
    assert len(web_in_final) <= 2


async def test_web_citation_is_reserved_when_local_results_exist():
    retrieval = FakeRetrieval(
        hybrid=[_kb(f"c{index}") for index in range(1, 6)],
        hyde=[],
    )
    web = FakeWeb([_web(1)])
    retriever = ThreeWayRetriever(
        _settings(web_search_min_citations=1), retrieval, FakeHyDE(""), web
    )

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    web_in_final = [item for item in outcome.citations if item.get("source_type") == "web"]
    assert len(outcome.citations) == 5
    assert len(web_in_final) == 1


async def test_web_cap_is_relaxed_when_local_is_empty():
    retrieval = FakeRetrieval(hybrid=[], hyde=[])
    web = FakeWeb([_web(index) for index in range(1, 6)])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE(""), web)

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert len(outcome.citations) == 5
    assert all(item["source_type"] == "web" for item in outcome.citations)


async def test_rerank_failure_falls_back_to_local_order():
    retrieval = FakeRetrieval(
        hybrid=[_kb("c1"), _kb("c2")],
        hyde=[],
        rerank_error=RuntimeError("reranker down"),
    )
    web = FakeWeb([])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE(""), web)

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert [item["chunk_id"] for item in outcome.citations] == ["c1", "c2"]
    assert "reranker" in outcome.info["degraded"]


async def test_emits_expected_stage_sequence():
    retrieval = FakeRetrieval(hybrid=[_kb("c1")], hyde=[_kb("c2")])
    web = FakeWeb([_web(1)])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE("资料"), web)
    stages: list[str] = []

    async def on_stage(name: str) -> None:
        stages.append(name)

    await retriever.search(
        "西湖", top_k=5, tenant_id="default", on_stage=on_stage
    )

    assert stages == [
        "retrieving",
        "web_searching",
        "hyde_generating",
        "reranking",
        "fusing",
    ]


async def test_hyde_disabled_skips_generation():
    retrieval = FakeRetrieval(hybrid=[_kb("c1")], hyde=[])
    hyde = FakeHyDE("资料")
    web = FakeWeb([])
    retriever = ThreeWayRetriever(_settings(hyde_enabled=False), retrieval, hyde, web)

    outcome = await retriever.search("西湖", top_k=5, tenant_id="default")

    assert hyde.calls == []
    assert outcome.info["hyde"]["status"] == "disabled"


async def test_non_default_tenant_reaches_hybrid_and_hyde_searches():
    retrieval = FakeRetrieval(hybrid=[_kb("c1")], hyde=[_kb("c2")])
    web = FakeWeb([_web(1)])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE("资料"), web)

    await retriever.search("西湖", top_k=5, tenant_id="sanya")

    calls_by_mode = {call["mode"]: call for call in retrieval.search_calls}
    assert set(calls_by_mode) == {"hybrid", "dense"}
    assert {call["tenant_id"] for call in calls_by_mode.values()} == {"sanya"}


@pytest.mark.parametrize("tenant_id", ["", None])
async def test_blank_tenant_raises_instead_of_degrading(tenant_id):
    retrieval = FakeRetrieval(hybrid=[_kb("c1")], hyde=[_kb("c2")])
    web = FakeWeb([_web(1)])
    retriever = ThreeWayRetriever(_settings(), retrieval, FakeHyDE("资料"), web)

    with pytest.raises(ValueError, match="tenant_id is required"):
        await retriever.search("西湖", top_k=5, tenant_id=tenant_id)

    assert retrieval.search_calls == []
