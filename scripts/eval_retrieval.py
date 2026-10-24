"""Compare dense, sparse and hybrid retrieval against a hand-built query set."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.retrieval.service import SEARCH_MODES, RetrievalService

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    service = RetrievalService(get_settings())
    ranks: dict[str, list[int | None]] = {mode: [] for mode in EVAL_MODES}

    try:
        for query, expected in EVAL_CASES:
            results = service.search_all(query, top_k=args.top_k)
            results["hybrid_rerank"] = service.search(query, top_k=args.top_k)
            line = [f"{query[:30]:<30}"]
            for mode in EVAL_MODES:
                rank = rank_of(results[mode], expected)
                ranks[mode].append(rank)
                label = "HR" if mode == "hybrid_rerank" else mode[:1].upper()
                line.append(f"{label}={rank or '-'}")
            print("  ".join(line))
    finally:
        service.close()

    total = len(EVAL_CASES)
    print()
    print(f"evaluation: {total} queries, top_k={args.top_k}")
    for mode in EVAL_MODES:
        values = ranks[mode]
        hit1 = sum(1 for rank in values if rank == 1) / total
        hit3 = sum(1 for rank in values if rank and rank <= 3) / total
        hit5 = sum(1 for rank in values if rank and rank <= args.top_k) / total
        mrr = sum(1.0 / rank for rank in values if rank) / total
        print(
            f"{mode:14s} Hit@1={hit1:.3f} Hit@3={hit3:.3f} "
            f"Hit@{args.top_k}={hit5:.3f} MRR={mrr:.3f}"
        )

    reranked_ranks = ranks["hybrid_rerank"]
    misses = [
        case[1][0]
        for case, rank in zip(EVAL_CASES, reranked_ranks, strict=True)
        if rank is None
    ]
    if misses:
        print("hybrid+rerank misses:")
        for miss in misses:
            print(f"  - {miss}")

    reranked_hit = sum(1 for rank in reranked_ranks if rank and rank <= args.top_k) / total
    return 0 if reranked_hit >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
