import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

_EVAL_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_qa.py"


@pytest.fixture(scope="module")
def ev() -> object:
    spec = importlib.util.spec_from_file_location("eval_qa", _EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(**overrides) -> dict:
    case = {
        "id": "case-1",
        "question": "杭州萧山机场到西湖怎么走？",
        "expected_sources": ["杭州交通指南.md"],
        "expected_facts": ["地铁", "机场大巴"],
    }
    case.update(overrides)
    return case


def test_load_cases_reads_json(tmp_path, ev) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([_case()], ensure_ascii=False), encoding="utf-8")

    cases = ev.load_cases(path)

    assert cases[0]["id"] == "case-1"


def test_load_cases_rejects_missing_fields(tmp_path, ev) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "bad"}], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        ev.load_cases(path)


def test_score_case_counts_citations_and_facts(ev) -> None:
    citations = [{"source_file": "杭州交通指南.md"}, {"source_file": "杭州景点推荐.md"}]
    row = ev.score_case(
        _case(),
        answer="可以坐地铁，也可以选机场大巴。",
        citations=citations,
    )

    assert row["citation_hit"] == 1.0
    assert row["citation_recall"] == 1.0
    assert row["fact_coverage"] == 1.0
    assert row["covered_facts"] == ["地铁", "机场大巴"]
    assert row["citation_count"] == 2


def test_score_case_reports_partial_coverage(ev) -> None:
    row = ev.score_case(
        _case(expected_sources=["a.md", "b.md"], expected_facts=["地铁", "机场大巴"]),
        answer="建议坐地铁。",
        citations=[{"source_file": "a.md"}],
    )

    assert row["citation_recall"] == 0.5
    assert row["fact_coverage"] == 0.5
    assert row["covered_facts"] == ["地铁"]


def test_aggregate_computes_rates(ev) -> None:
    rows = [
        {"citation_hit": 1.0, "citation_recall": 1.0, "fact_coverage": 1.0, "latency_ms": 100},
        {"citation_hit": 0.0, "citation_recall": 0.0, "fact_coverage": 0.5, "latency_ms": 300},
    ]

    summary = ev.aggregate(rows)

    assert summary["cases"] == 2
    assert summary["citation_hit_rate"] == 0.5
    assert summary["citation_recall"] == 0.5
    assert summary["fact_coverage"] == 0.75
    assert summary["avg_latency_ms"] == 200


def test_parse_judge_payload_accepts_fenced_json(ev) -> None:
    payload = ev.parse_judge_payload('```json\n{"score": 4, "reason": "覆盖主要信息"}\n```')

    assert payload == {"score": 4.0, "reason": "覆盖主要信息"}
    assert ev.parse_judge_payload("not json") is None


@pytest.mark.asyncio
async def test_evaluate_case_uses_retriever_and_llm(ev) -> None:
    class FakeRetriever:
        async def search(self, query, *, top_k, city, content_type):
            return SimpleNamespace(
                citations=[{"source_file": "杭州交通指南.md"}],
                info={"degraded": []},
            )

    class FakeLLM:
        async def ainvoke(self, messages, **kwargs):
            return AIMessage(content="可以坐地铁或机场大巴前往。")

    row = await ev.evaluate_case(
        _case(),
        retriever=FakeRetriever(),
        llm=FakeLLM(),
        routing="off",
        top_k=5,
    )

    assert row["answer"] == "可以坐地铁或机场大巴前往。"
    assert row["fact_coverage"] == 1.0
    assert row["citation_hit"] == 1.0
