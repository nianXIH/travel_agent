"""Repository-level checks for the tenant filters emitted to Milvus.

The fake client never opens a connection; it only records the exact `filter`
expressions passed by `MilvusRepository`.
"""

import pytest

from app.retrieval.milvus_repo import MilvusRepository

COLLECTION = "travel_chunks"

PREVIEW_FIELDS = [
    "chunk_id",
    "chunk_index",
    "section_path",
    "content",
    "content_type",
    "city",
    "updated_at",
]


class FakeMilvusClient:
    def __init__(self, count: int = 0) -> None:
        self.deletes: list[tuple[str, str]] = []
        self.queries: list[dict] = []
        self._count = count

    def delete(self, collection_name: str, filter: str) -> None:  # noqa: A002
        self.deletes.append((collection_name, filter))

    def query(
        self,
        collection_name: str,
        filter: str,  # noqa: A002
        output_fields: list[str],
        limit: int | None = None,
    ) -> list[dict]:
        self.queries.append(
            {
                "collection_name": collection_name,
                "filter": filter,
                "output_fields": output_fields,
                "limit": limit,
            }
        )
        if output_fields == ["count(*)"]:
            return [{"count(*)": self._count}]
        return []


def repo_with(client: FakeMilvusClient) -> MilvusRepository:
    """Build a repository around the fake client without touching Milvus/`.env`."""
    repo = object.__new__(MilvusRepository)
    repo._collection = COLLECTION
    repo._client = client
    return repo


def test_delete_stale_chunks_without_kept_ids_filters_by_tenant_and_doc():
    client = FakeMilvusClient()

    repo_with(client).delete_stale_chunks("sanya", "doc_1", [])

    assert client.deletes == [
        (COLLECTION, 'tenant_id == "sanya" and doc_id == "doc_1"'),
    ]


def test_delete_stale_chunks_excludes_kept_ids_after_tenant_clause():
    client = FakeMilvusClient()

    repo_with(client).delete_stale_chunks("sanya", "doc_1", ["doc_1:0", 'doc_1:"1'])

    assert client.deletes == [
        (
            COLLECTION,
            'tenant_id == "sanya" and doc_id == "doc_1"'
            ' and chunk_id not in ["doc_1:0", "doc_1:\\"1"]',
        ),
    ]


def test_delete_stale_chunks_uses_quoted_tenant_and_doc():
    client = FakeMilvusClient()

    repo_with(client).delete_stale_chunks('a"b', "d\\e", [])

    assert client.deletes == [
        (COLLECTION, 'tenant_id == "a\\"b" and doc_id == "d\\\\e"'),
    ]


def test_delete_document_chunks_always_filters_by_tenant_and_doc():
    client = FakeMilvusClient()

    repo_with(client).delete_document_chunks("default", "doc_2")

    assert client.deletes == [
        (COLLECTION, 'tenant_id == "default" and doc_id == "doc_2"'),
    ]


def test_count_chunks_without_tenant_is_the_global_aggregate():
    client = FakeMilvusClient(count=42)

    total = repo_with(client).count_chunks()

    assert total == 42
    assert client.queries == [
        {
            "collection_name": COLLECTION,
            "filter": 'chunk_id != ""',
            "output_fields": ["count(*)"],
            "limit": None,
        },
    ]


def test_count_chunks_with_tenant_is_scoped():
    client = FakeMilvusClient(count=3)

    total = repo_with(client).count_chunks("default")

    assert total == 3
    assert client.queries == [
        {
            "collection_name": COLLECTION,
            "filter": 'tenant_id == "default" and chunk_id != ""',
            "output_fields": ["count(*)"],
            "limit": None,
        },
    ]


def test_list_document_chunks_emits_tenant_clause_and_preview_fields():
    client = FakeMilvusClient()

    rows = repo_with(client).list_document_chunks("sanya", "doc_1")

    assert rows == []
    assert client.queries == [
        {
            "collection_name": COLLECTION,
            "filter": 'tenant_id == "sanya" and doc_id == "doc_1"',
            "output_fields": PREVIEW_FIELDS,
            "limit": 200,
        },
    ]


def test_list_document_chunks_honours_limit_and_quotes_values():
    client = FakeMilvusClient()

    repo_with(client).list_document_chunks('a"b', "d\\e", limit=5)

    assert client.queries == [
        {
            "collection_name": COLLECTION,
            "filter": 'tenant_id == "a\\"b" and doc_id == "d\\\\e"',
            "output_fields": PREVIEW_FIELDS,
            "limit": 5,
        },
    ]


@pytest.mark.parametrize("tenant_id", ["", None])
def test_list_document_chunks_rejects_blank_tenant(tenant_id):
    client = FakeMilvusClient()

    with pytest.raises(ValueError, match="tenant_id is required"):
        repo_with(client).list_document_chunks(tenant_id, "doc_1")

    assert client.queries == []
