"""Evaluate RAG answer quality against a golden QA set.

Metrics:
- citation_hit: at least one expected source appears in the citations
- citation_recall: share of expected sources that appear in the citations
- fact_coverage: share of expected facts present in the answer
- optional LLM-as-judge score (1-5) with reason
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chat.llm import ChatLLM, message_text
from app.chat.routing import heuristic_route
from app.core.config import get_settings
from app.core.prompts import QA_SYSTEM_PROMPT, build_context
from app.core.tenancy import DEFAULT_TENANT_ID
from app.retrieval.hyde import HyDEGenerator
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService
from app.retrieval.three_way import ThreeWayRetriever
from app.retrieval.web_search import WebSearchClient

DEFAULT_CASES = Path(__file__).with_name("qa_cases.json")

REQUIRED_FIELDS = ("id", "question", "expected_sources", "expected_facts")

JUDGE_SYSTEM_PROMPT = (
    "你是严格的 RAG 回答质量评测员。根据问题、参考答案要点和候选回答，"
    "从相关性、事实一致性、要点覆盖三个方面打分（1-5 分，5 分最好）。"
    '只输出 JSON，例如 {"score": 4, "reason": "覆盖主要信息，无事实冲突"}。'
)

_WHITESPACE_RE = re.compile(r"\s+")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub("", text).lower()


def load_cases(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("qa cases must be a non-empty list")

    for index, case in enumerate(data):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        for field in REQUIRED_FIELDS:
            if field not in case:
                raise ValueError(f"case {index} missing field: {field}")
        if not case["question"]:
            raise ValueError(f"case {index} missing field: question")
        for field in ("expected_sources", "expected_facts"):
            if not isinstance(case[field], list):
                raise ValueError(f"case {index} field {field} must be a list")
    return data


def score_case(case: dict, *, answer: str, citations: list[dict]) -> dict:
    cited_files = {
        str(hit.get("source_file")).strip()
        for hit in citations
        if hit.get("source_file")
    }
    expected_sources = [str(item) for item in case.get("expected_sources", [])]
    hit_sources = [source for source in expected_sources if source in cited_files]

    expected_facts = [str(item) for item in case.get("expected_facts", [])]
    normalized_answer = _normalize(answer)
    covered_facts = [fact for fact in expected_facts if _normalize(fact) in normalized_answer]
    missing_facts = [fact for fact in expected_facts if fact not in covered_facts]

    return {
        "citation_hit": 1.0 if hit_sources else 0.0,
        "citation_recall": len(hit_sources) / len(expected_sources) if expected_sources else 0.0,
        "fact_coverage": len(covered_facts) / len(expected_facts) if expected_facts else 0.0,
        "covered_facts": covered_facts,
        "missing_facts": missing_facts,
        "hit_sources": hit_sources,
        "citation_count": len(citations),
        "answer_chars": len(answer),
        "answer": answer,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {
            "cases": 0,
            "citation_hit_rate": 0.0,
            "citation_recall": 0.0,
            "fact_coverage": 0.0,
            "avg_latency_ms": 0,
            "avg_answer_chars": 0,
        }

    total = len(rows)
    summary = {
        "cases": total,
        "citation_hit_rate": sum(row["citation_hit"] for row in rows) / total,
        "citation_recall": sum(row["citation_recall"] for row in rows) / total,
        "fact_coverage": sum(row["fact_coverage"] for row in rows) / total,
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / total),
        "avg_answer_chars": round(sum(row.get("answer_chars", 0) for row in rows) / total),
    }
    judge_scores = [
        row["judge"]["score"] for row in rows if isinstance(row.get("judge"), dict)
    ]
    if judge_scores:
        summary["avg_judge_score"] = sum(judge_scores) / len(judge_scores)
    return summary


def parse_judge_payload(text: str) -> dict | None:
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    score = data.get("score")
    if not isinstance(score, (int, float)):
        return None
    return {
        "score": max(1.0, min(5.0, float(score))),
        "reason": str(data.get("reason") or ""),
    }


async def judge_answer(llm: ChatLLM, case: dict, answer: str) -> dict | None:
    prompt = (
        f"用户问题：{case['question']}\n"
        f"参考答案要点：{'、'.join(str(item) for item in case['expected_facts'])}\n"
        f"候选回答：{answer}"
    )
    response = await llm.ainvoke(
        [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=prompt)],
        temperature=0.0,
    )
    return parse_judge_payload(message_text(response))


async def evaluate_case(
    case: dict,
    *,
    retriever: ThreeWayRetriever,
    llm: ChatLLM,
    routing: str = "heuristic",
    top_k: int = 5,
    judge: bool = False,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> dict:
    question = case["question"]
    city = case.get("city")
    content_type = case.get("content_type")
    if routing == "heuristic" and (city is None or content_type is None):
        routed_city, routed_type, _ = heuristic_route(question)
        city = city or routed_city
        content_type = content_type or routed_type

    started_at = time.perf_counter()
    outcome = await retriever.search(
        question,
        tenant_id=tenant_id,
        top_k=top_k,
        city=city,
        content_type=content_type,
    )
    citations = outcome.citations
    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=f"参考资料：\n{build_context(citations)}\n\n用户问题：{question}"),
    ]
    response = await llm.ainvoke(messages)
    answer = message_text(response).strip()
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    row = {
        "id": case["id"],
        "question": question,
        "route": {"city": city, "content_type": content_type},
        "latency_ms": latency_ms,
        "citations": [_citation_brief(hit) for hit in citations],
        "retrieval": outcome.info,
    }
    row.update(score_case(case, answer=answer, citations=citations))
    if judge:
        row["judge"] = await judge_answer(llm, case, answer)
    return row


async def run_cases(
    cases: list[dict],
    *,
    retriever: ThreeWayRetriever,
    llm: ChatLLM,
    routing: str,
    top_k: int,
    judge: bool,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> list[dict]:
    rows: list[dict] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:>2}/{len(cases)}] {case['id']} ...", flush=True)
        try:
            row = await evaluate_case(
                case,
                retriever=retriever,
                llm=llm,
                routing=routing,
                top_k=top_k,
                judge=judge,
                tenant_id=tenant_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            row = _failed_row(case, exc)
        rows.append(row)
        _print_case(row, case)
    return rows


def _failed_row(case: dict, exc: Exception) -> dict:
    return {
        "id": case["id"],
        "question": case["question"],
        "error": f"{type(exc).__name__}: {exc}",
        "citation_hit": 0.0,
        "citation_recall": 0.0,
        "fact_coverage": 0.0,
        "covered_facts": [],
        "missing_facts": list(case.get("expected_facts", [])),
        "hit_sources": [],
        "citation_count": 0,
        "answer_chars": 0,
        "answer": "",
        "latency_ms": 0,
    }


def _citation_brief(hit: dict) -> dict:
    keys = (
        "chunk_id",
        "source_type",
        "source_file",
        "title",
        "url",
        "content_type",
        "city",
        "score",
        "fusion_score",
    )
    return {key: hit.get(key) for key in keys}


def _print_case(row: dict, case: dict) -> None:
    if row.get("error"):
        print(f"    ERROR {row['error']}")
        return
    route = row.get("route") or {}
    route_label = f"{route.get('city') or '-'}/{route.get('content_type') or '-'}"
    print(
        f"    citation={row['citation_hit']:.2f} recall={row['citation_recall']:.2f} "
        f"facts={row['fact_coverage']:.2f} ({len(row['covered_facts'])}/"
        f"{len(case['expected_facts'])}) route={route_label} {row['latency_ms']}ms"
    )
    if row["missing_facts"]:
        print(f"    missing: {'、'.join(row['missing_facts'])}")
    if isinstance(row.get("judge"), dict):
        print(f"    judge={row['judge']['score']:.1f} {row['judge']['reason']}")


def _print_summary(summary: dict, *, routing: str, judge: bool) -> None:
    print()
    print(
        f"evaluation: {summary['cases']} cases, routing={routing}, "
        f"judge={'on' if judge else 'off'}"
    )
    print(
        f"citation_hit={summary['citation_hit_rate']:.3f} "
        f"citation_recall={summary['citation_recall']:.3f} "
        f"fact_coverage={summary['fact_coverage']:.3f} "
        f"avg_latency_ms={summary['avg_latency_ms']} "
        f"avg_answer_chars={summary['avg_answer_chars']}"
    )
    if "avg_judge_score" in summary:
        print(f"avg_judge_score={summary['avg_judge_score']:.2f}")


async def _run(
    cases: list[dict],
    *,
    retriever: ThreeWayRetriever,
    llm: ChatLLM,
    routing: str,
    top_k: int,
    judge: bool,
    tenant_id: str = DEFAULT_TENANT_ID,
) -> list[dict]:
    try:
        return await run_cases(
            cases,
            retriever=retriever,
            llm=llm,
            routing=routing,
            top_k=top_k,
            judge=judge,
            tenant_id=tenant_id,
        )
    finally:
        await retriever.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N cases")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--routing", choices=("off", "heuristic"), default="heuristic")
    parser.add_argument("--judge", action="store_true", help="enable LLM-as-judge scoring")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--min-citation-hit", type=float, default=0.9)
    parser.add_argument("--min-fact-coverage", type=float, default=0.8)
    parser.add_argument(
        "--tenant",
        default=DEFAULT_TENANT_ID,
        help="tenant id (default: default)",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    settings = get_settings()
    retrieval = RetrievalService(settings, milvus=MilvusRepository(settings))
    llm = ChatLLM(settings)
    retriever = ThreeWayRetriever(
        settings,
        retrieval,
        HyDEGenerator(settings, llm),
        WebSearchClient(settings),
    )

    try:
        rows = asyncio.run(
            _run(
                cases,
                retriever=retriever,
                llm=llm,
                routing=args.routing,
                top_k=args.top_k,
                judge=args.judge,
                tenant_id=args.tenant,
            )
        )
    finally:
        retrieval.close()

    summary = aggregate(rows)
    _print_summary(summary, routing=args.routing, judge=args.judge)

    if args.json_out:
        report = {
            "config": {
                "top_k": args.top_k,
                "routing": args.routing,
                "judge": args.judge,
                "tenant": args.tenant,
            },
            "summary": summary,
            "cases": rows,
        }
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"report written to {args.json_out}")

    return (
        0
        if summary["citation_hit_rate"] >= args.min_citation_hit
        and summary["fact_coverage"] >= args.min_fact_coverage
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
