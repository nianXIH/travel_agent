"""Evaluate local retrieval modes and the three-way retrieval pipeline."""

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chat.llm import ChatLLM
from app.chat.routing import heuristic_route
from app.core.config import get_settings
from app.retrieval.hyde import HyDEGenerator
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import SEARCH_MODES, RetrievalService
from app.retrieval.three_way import ThreeWayRetriever
from app.retrieval.web_search import WebSearchClient

EVAL_MODES = (*SEARCH_MODES, "hybrid_rerank")

EVAL_CASES = [
    ("杭州萧山机场到西湖怎么走，地铁和打车怎么选", ["杭州交通指南.md"]),
    ("杭州住西湖湖滨还是灵隐附近，带家人怎么选", ["杭州住宿推荐.md"]),
    ("杭州西湖灵隐龙井这些景点怎么取舍", ["杭州景点推荐.md"]),
    ("杭州加乌镇三日游线路怎么安排", ["杭州线路推荐.md"]),
    ("杭州杭帮菜和特色小吃有哪些", ["杭州美食推荐.md"]),
    ("成都双流机场到市区怎么走", ["成都交通指南.md"]),
    ("成都住春熙路还是宽窄巷子方便", ["成都住宿推荐.md"]),
    ("成都必去景点有哪些", ["成都景点推荐.md"]),
    ("成都周边三日游线路怎么排", ["成都线路推荐.md"]),
    ("成都火锅和串串去哪里吃", ["成都美食推荐.md"]),
    ("三亚凤凰机场到亚龙湾交通", ["三亚交通指南.md"]),
    ("三亚住亚龙湾还是海棠湾", ["三亚住宿推荐.md"]),
    ("三亚海岛和景点怎么选", ["三亚景点推荐.md"]),
    ("三亚五天行程线路怎么安排", ["三亚线路推荐.md"]),
    ("三亚海鲜和椰子鸡去哪吃", ["三亚美食推荐.md"]),
    ("厦门高崎机场到鼓浪屿怎么走", ["厦门交通指南.md"]),
    ("厦门住曾厝垵还是中山路", ["厦门住宿推荐.md"]),
    ("厦门鼓浪屿和环岛路怎么玩", ["厦门景点推荐.md"]),
    ("厦门三四天线路怎么安排", ["厦门线路推荐.md"]),
    ("厦门沙茶面和海鲜小吃", ["厦门美食推荐.md"]),
    ("张家界机场到武陵源景区怎么走", ["张家界交通指南.md"]),
    ("张家界住市区还是武陵源", ["张家界住宿推荐.md"]),
    ("张家界国家森林公园怎么玩", ["张家界景点推荐.md"]),
    ("张家界四天线路怎么安排", ["张家界线路推荐.md"]),
    ("张家界特色菜吃什么", ["张家界美食推荐.md"]),
    ("云南昆明大理丽江交通怎么串联", ["云南交通指南.md"]),
    ("云南大理丽江住哪里", ["云南住宿推荐.md"]),
    ("云南玉龙雪山和洱海怎么玩", ["云南景点推荐.md"]),
    ("云南七日游线路怎么安排", ["云南线路推荐.md"]),
    ("云南过桥米线和菌子火锅", ["云南美食推荐.md"]),
]


def rank_of(hits: list[dict], expected: list[str]) -> int | None:
    for position, hit in enumerate(hits, start=1):
        if hit.get("source_file") in expected:
            return position
    return None


def hit_labels(hits: list[dict], *, limit: int = 3) -> list[str]:
    labels = []
    for position, hit in enumerate(hits[:limit], start=1):
        source = hit.get("source_type", "kb")
        name = (
            hit.get("source_file")
            or hit.get("title")
            or hit.get("url")
            or hit.get("chunk_id")
            or "?"
        )
        labels.append(f"{position}.{source}:{name}")
    return labels


def aggregate(ranks: list[int | None], *, top_k: int) -> dict[str, float]:
    total = len(ranks)
    if total == 0:
        return {"hit@1": 0.0, "hit@3": 0.0, f"hit@{top_k}": 0.0, "mrr": 0.0}
    return {
        "hit@1": sum(1 for rank in ranks if rank == 1) / total,
        "hit@3": sum(1 for rank in ranks if rank and rank <= 3) / total,
        f"hit@{top_k}": sum(1 for rank in ranks if rank and rank <= top_k) / total,
        "mrr": sum(1.0 / rank for rank in ranks if rank) / total,
    }


def summarize_three_way(infos: list[dict]) -> dict:
    path_status: dict[str, dict[str, int]] = {path: {} for path in ("hybrid", "hyde", "web")}
    degraded: Counter[str] = Counter()
    for info in infos:
        for path in path_status:
            status = info.get(path, {}).get("status", "unknown")
            path_status[path][status] = path_status[path].get(status, 0) + 1
        degraded.update(info.get("degraded", []))

    elapsed = [info.get("elapsed_ms", 0) for info in infos]
    return {
        "queries": len(infos),
        "path_status": path_status,
        "degraded": dict(degraded),
        "web_in_final": sum(info.get("fusion", {}).get("web_in_final", 0) for info in infos),
        "avg_elapsed_ms": round(sum(elapsed) / len(elapsed)) if elapsed else 0,
    }


def _print_local_metrics(ranks: dict[str, list[int | None]], *, top_k: int) -> None:
    for mode in EVAL_MODES:
        metrics = aggregate(ranks[mode], top_k=top_k)
        print(
            f"{mode:14s} Hit@1={metrics['hit@1']:.3f} Hit@3={metrics['hit@3']:.3f} "
            f"Hit@{top_k}={metrics[f'hit@{top_k}']:.3f} MRR={metrics['mrr']:.3f}"
        )


def _run_local(cases: list[tuple[str, list[str]]], top_k: int) -> int:
    service = RetrievalService(get_settings())
    ranks: dict[str, list[int | None]] = {mode: [] for mode in EVAL_MODES}

    try:
        for query, expected in cases:
            results = service.search_all(query, top_k=top_k)
            results["hybrid_rerank"] = service.search(query, top_k=top_k)
            line = [f"{query[:30]:<30}"]
            for mode in EVAL_MODES:
                rank = rank_of(results[mode], expected)
                ranks[mode].append(rank)
                label = "HR" if mode == "hybrid_rerank" else mode[:1].upper()
                line.append(f"{label}={rank or '-'}")
            print("  ".join(line))
    finally:
        service.close()

    print()
    print(f"evaluation: {len(cases)} queries, top_k={top_k}, pipeline=local")
    _print_local_metrics(ranks, top_k=top_k)

    reranked_ranks = ranks["hybrid_rerank"]
    hits = sum(1 for rank in reranked_ranks if rank and rank <= top_k)
    return 0 if hits / len(cases) >= 0.9 else 1


async def _evaluate_three_way(
    cases: list[tuple[str, list[str]]],
    top_k: int,
    show_hits: int = 0,
    routing: str = "off",
) -> tuple[list[int | None], list[dict]]:
    settings = get_settings()
    retrieval = RetrievalService(settings, milvus=MilvusRepository(settings))
    llm = ChatLLM(settings)
    retriever = ThreeWayRetriever(
        settings,
        retrieval,
        HyDEGenerator(settings, llm),
        WebSearchClient(settings),
    )

    ranks: list[int | None] = []
    infos: list[dict] = []
    try:
        for query, expected in cases:
            city = content_type = None
            if routing == "heuristic":
                city, content_type, _ = heuristic_route(query)
            outcome = await retriever.search(
                query,
                top_k=top_k,
                city=city,
                content_type=content_type,
            )
            rank = rank_of(outcome.citations, expected)
            ranks.append(rank)
            infos.append(outcome.info)
            info = outcome.info
            route = f" route={city or '-'}/{content_type or '-'}" if routing != "off" else ""
            print(
                f"{query[:30]:<30} final={str(rank or '-'):<3} "
                f"hybrid={info['hybrid']['status']:<11} hyde={info['hyde']['status']:<9} "
                f"web={info['web']['status']:<11} "
                f"web_final={info['fusion']['web_in_final']} {info['elapsed_ms']}ms{route}"
            )
            if show_hits:
                print("    " + "  ".join(hit_labels(outcome.citations, limit=show_hits)))
    finally:
        await retriever.aclose()
        retrieval.close()
    return ranks, infos


def _run_three_way(
    cases: list[tuple[str, list[str]]],
    top_k: int,
    show_hits: int = 0,
    routing: str = "off",
) -> int:
    ranks, infos = asyncio.run(_evaluate_three_way(cases, top_k, show_hits, routing))
    metrics = aggregate(ranks, top_k=top_k)
    summary = summarize_three_way(infos)

    print()
    print(f"evaluation: {summary['queries']} queries, top_k={top_k}, pipeline=three-way")
    print(
        f"{'final':14s} Hit@1={metrics['hit@1']:.3f} Hit@3={metrics['hit@3']:.3f} "
        f"Hit@{top_k}={metrics[f'hit@{top_k}']:.3f} MRR={metrics['mrr']:.3f}"
    )
    print(f"avg_elapsed_ms={summary['avg_elapsed_ms']}  web_in_final={summary['web_in_final']}")
    for path, counts in summary["path_status"].items():
        detail = "  ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        print(f"{path:14s} {detail}")
    if summary["degraded"]:
        detail = "  ".join(f"{name}={count}" for name, count in sorted(summary["degraded"].items()))
        print(f"{'degraded':14s} {detail}")

    misses = [expected[0] for (_, expected), rank in zip(cases, ranks, strict=True) if rank is None]
    if misses:
        print("three-way misses:")
        for miss in misses:
            print(f"  - {miss}")
    return 0 if metrics[f"hit@{top_k}"] >= 0.9 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pipeline", choices=("local", "three-way"), default="local")
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N cases")
    parser.add_argument(
        "--show-hits",
        type=int,
        default=0,
        metavar="N",
        help="print the top N final hits for each three-way query",
    )
    parser.add_argument(
        "--routing",
        choices=("off", "heuristic"),
        default="off",
        help="apply heuristic city/content-type filters before retrieval",
    )
    args = parser.parse_args()

    cases = EVAL_CASES[: args.limit] if args.limit else EVAL_CASES
    if args.pipeline == "three-way":
        return _run_three_way(cases, args.top_k, args.show_hits, args.routing)
    return _run_local(cases, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
