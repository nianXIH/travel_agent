import uuid

import pytest

from app.core.tenancy import DEFAULT_TENANT_ID
from app.retrieval.milvus_repo import DENSE_FIELD, SPARSE_FIELD

pytestmark = pytest.mark.integration


def _seed(milvus, filter_expr: str = 'chunk_id != ""') -> dict:
    try:
        rows = milvus.client.query(
            milvus.collection_name,
            filter=filter_expr,
            output_fields=[
                "tenant_id",
                "chunk_id",
                "doc_id",
                "city",
                "content",
                DENSE_FIELD,
                SPARSE_FIELD,
            ],
            limit=1,
        )
    except Exception as exc:
        pytest.skip(f"Milvus tenant schema unavailable: {exc}")
    if not rows:
        pytest.skip("Milvus collection is empty; ingest the knowledge base first")
    return rows[0]


def _sparse_vector(row: dict) -> dict[int, float]:
    return {int(token): float(weight) for token, weight in (row[SPARSE_FIELD] or {}).items()}


def _hit_ids(hits: list[dict]) -> list[str]:
    return [hit.get("chunk_id") for hit in hits]


def test_schema_and_indexes_are_valid(milvus):
    assert milvus.validate() == []
    assert milvus.count_chunks() >= 0


def test_dense_search_returns_seed_chunk(milvus):
    seed = _seed(milvus)

    hits = milvus.search_dense(seed[DENSE_FIELD], limit=3)

    assert hits
    assert seed["chunk_id"] in _hit_ids(hits)


def test_sparse_search_returns_seed_chunk(milvus):
    seed = _seed(milvus)
    sparse = _sparse_vector(seed)
    if not sparse:
        pytest.skip("seed chunk has no sparse tokens")

    hits = milvus.search_sparse(sparse, limit=3)

    assert hits
    assert seed["chunk_id"] in _hit_ids(hits)


def test_hybrid_search_returns_seed_chunk(milvus):
    seed = _seed(milvus)
    sparse = _sparse_vector(seed)
    if not sparse:
        pytest.skip("seed chunk has no sparse tokens")

    hits = milvus.search_hybrid(
        seed[DENSE_FIELD],
        sparse,
        limit=5,
        dense_limit=10,
        sparse_limit=10,
    )

    assert hits
    assert seed["chunk_id"] in _hit_ids(hits)


def test_city_filter_is_applied(milvus):
    seed = _seed(milvus, 'city != ""')
    city = seed["city"]

    hits = milvus.search_dense(
        seed[DENSE_FIELD],
        limit=10,
        filter_expr=f'city == "{city}"',
    )

    assert hits
    assert all((hit.get("entity") or {}).get("city") == city for hit in hits)


def test_doc_id_filter_is_applied(milvus):
    seed = _seed(milvus)
    doc_id = seed["doc_id"]

    hits = milvus.search_dense(
        seed[DENSE_FIELD],
        limit=10,
        filter_expr=f'doc_id == "{doc_id}"',
    )

    assert hits
    assert all((hit.get("entity") or {}).get("doc_id") == doc_id for hit in hits)


def test_tenant_filter_is_applied(milvus):
    seed = _seed(milvus, 'tenant_id != ""')
    tenant_id = seed["tenant_id"]
    assert tenant_id == DEFAULT_TENANT_ID

    hits = milvus.search_dense(
        seed[DENSE_FIELD],
        limit=10,
        filter_expr=f'tenant_id == "{tenant_id}"',
    )

    assert hits
    assert all((hit.get("entity") or {}).get("tenant_id") == tenant_id for hit in hits)

    other = f"it{uuid.uuid4().hex[:8]}"
    leaks = milvus.search_dense(
        seed[DENSE_FIELD],
        limit=10,
        filter_expr=f'tenant_id == "{other}"',
    )
    assert leaks == []
