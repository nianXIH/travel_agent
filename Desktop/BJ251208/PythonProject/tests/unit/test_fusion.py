from app.retrieval.fusion import rrf_fuse


def _hit(chunk_id: str, **extra) -> dict:
    return {"chunk_id": chunk_id, "content": chunk_id, "score": 0.0, **extra}


def test_rrf_merges_duplicate_chunks_and_sources():
    hybrid = [_hit("c1"), _hit("c2")]
    hyde = [_hit("c2"), _hit("c3")]

    fused = rrf_fuse(
        [hybrid, hyde],
        source_names=["hybrid", "hyde"],
        k=60,
    )

    assert [item["chunk_id"] for item in fused] == ["c2", "c1", "c3"]
    assert fused[0]["retrieval_sources"] == ["hybrid", "hyde"]
    assert fused[0]["fusion_score"] > fused[1]["fusion_score"]


def test_weights_promote_hyde_hit():
    hybrid = [_hit("c1")]
    hyde = [_hit("c2")]

    fused = rrf_fuse(
        [hybrid, hyde],
        weights=[1.0, 2.0],
        source_names=["hybrid", "hyde"],
    )

    assert [item["chunk_id"] for item in fused] == ["c2", "c1"]


def test_limit_and_web_cap():
    local = [_hit(f"l{index}") for index in range(1, 5)]
    web = [
        {"chunk_id": "web:1", "url": "https://a", "content": "a", "retrieval_sources": ["web"]},
        {"chunk_id": "web:2", "url": "https://b", "content": "b", "retrieval_sources": ["web"]},
        {"chunk_id": "web:3", "url": "https://c", "content": "c", "retrieval_sources": ["web"]},
    ]

    fused = rrf_fuse(
        [local, web],
        source_names=["local"],
        limit=3,
        max_from={"web": 1},
    )

    assert len(fused) == 3
    assert sum("web" in item["retrieval_sources"] for item in fused) == 1


def test_empty_lists_return_empty():
    assert rrf_fuse([[], []]) == []


def test_does_not_mutate_input_hits():
    original = _hit("c1")
    rrf_fuse([[original]], source_names=["hybrid"])

    assert original == {"chunk_id": "c1", "content": "c1", "score": 0.0}
