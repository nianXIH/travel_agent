import importlib.util
from pathlib import Path

import pytest

_EVAL_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_retrieval.py"


@pytest.fixture(scope="module")
def ev() -> object:
    spec = importlib.util.spec_from_file_location("eval_retrieval", _EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_of_returns_first_matching_position(ev) -> None:
    hits = [
        {"source_file": "other.md"},
        {"source_file": "target.md"},
        {"source_file": "target.md"},
    ]
    assert ev.rank_of(hits, ["target.md"]) == 2
    assert ev.rank_of(hits, ["missing.md"]) is None


def test_aggregate_reports_hit_rates_and_mrr(ev) -> None:
    metrics = ev.aggregate([1, 3, None, 5], top_k=5)

    assert metrics["hit@1"] == 0.25
    assert metrics["hit@3"] == 0.5
    assert metrics["hit@5"] == 0.75
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 5) / 4)


def test_aggregate_handles_all_misses(ev) -> None:
    assert ev.aggregate([None, None], top_k=5) == {
        "hit@1": 0.0,
        "hit@3": 0.0,
        "hit@5": 0.0,
        "mrr": 0.0,
    }


def test_summarize_three_way_counts_status_and_web(ev) -> None:
    infos = [
        {
            "hybrid": {"status": "ok"},
            "hyde": {"status": "ok"},
            "web": {"status": "ok"},
            "degraded": [],
            "fusion": {"web_in_final": 1},
            "elapsed_ms": 100,
        },
        {
            "hybrid": {"status": "ok"},
            "hyde": {"status": "disabled"},
            "web": {"status": "unavailable"},
            "degraded": ["web"],
            "fusion": {"web_in_final": 0},
            "elapsed_ms": 300,
        },
    ]

    summary = ev.summarize_three_way(infos)

    assert summary["queries"] == 2
    assert summary["path_status"]["hyde"] == {"ok": 1, "disabled": 1}
    assert summary["path_status"]["web"] == {"ok": 1, "unavailable": 1}
    assert summary["degraded"] == {"web": 1}
    assert summary["web_in_final"] == 1
    assert summary["avg_elapsed_ms"] == 200
