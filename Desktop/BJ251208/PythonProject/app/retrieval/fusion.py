from collections.abc import Sequence


def rrf_fuse(
    ranked_lists: Sequence[Sequence[dict]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
    limit: int | None = None,
    id_key: str = "chunk_id",
    source_names: Sequence[str] | None = None,
    max_from: dict[str, int] | None = None,
) -> list[dict]:
    if k <= 0:
        raise ValueError("rrf k must be positive")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists length")

    merged: dict[str, dict] = {}
    first_seen: dict[str, int] = {}

    for list_index, hits in enumerate(ranked_lists):
        weight = float(weights[list_index])
        source = (
            source_names[list_index]
            if source_names is not None and list_index < len(source_names)
            else None
        )
        for rank, hit in enumerate(hits, start=1):
            key = _item_id(hit, id_key)
            contribution = weight / (k + rank)
            if key in merged:
                item = merged[key]
                item["fusion_score"] += contribution
                item["retrieval_sources"] = _merge_sources(
                    item.get("retrieval_sources"),
                    [source] if source else [],
                    hit.get("retrieval_sources"),
                )
                continue

            item = dict(hit)
            item["fusion_score"] = contribution
            item["retrieval_sources"] = _merge_sources(
                hit.get("retrieval_sources"),
                [source] if source else [],
            )
            merged[key] = item
            first_seen[key] = len(first_seen)

    ordered = sorted(
        merged.items(),
        key=lambda entry: (-entry[1]["fusion_score"], first_seen[entry[0]]),
    )

    selected: list[dict] = []
    counts: dict[str, int] = {}
    for _, item in ordered:
        sources = item.get("retrieval_sources") or []
        if _blocked(sources, counts, max_from):
            continue
        selected.append(item)
        for source in set(sources):
            counts[source] = counts.get(source, 0) + 1
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _item_id(item: dict, id_key: str) -> str:
    value = item.get(id_key)
    if value:
        return str(value)
    fallback = "|".join(str(item.get(field) or "") for field in ("url", "title", "content"))
    return fallback or f"anonymous:{id(item)}"


def _merge_sources(*groups) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for source in group or []:
            if source and source not in merged:
                merged.append(source)
    return merged


def _blocked(
    sources: Sequence[str],
    counts: dict[str, int],
    max_from: dict[str, int] | None,
) -> bool:
    if not max_from:
        return False
    capped = [source for source in sources if source in max_from]
    if not capped:
        return False
    if any(source not in max_from for source in sources):
        return False
    return any(counts.get(source, 0) >= max_from[source] for source in capped)
