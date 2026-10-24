from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    MilvusClient,
    RRFRanker,
)

from app.core.config import Settings
from app.retrieval.filters import quote

DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
DENSE_DIM = 1024
MAX_CONTENT_LENGTH = 16384

OUTPUT_FIELDS = [
    "doc_id",
    "content",
    "content_type",
    "city",
    "scenic_name",
    "route_name",
    "hotel_name",
    "restaurant_name",
    "source_file",
    "source_path",
    "section_path",
    "chunk_index",
    "extra",
    "updated_at",
]

SCALAR_INDEXES = (
    "doc_id",
    "content_type",
    "city",
    "scenic_name",
    "route_name",
    "hotel_name",
    "restaurant_name",
)


def build_schema() -> CollectionSchema:
    schema = CollectionSchema(
        fields=[
            _varchar("chunk_id", 256, primary=True),
            _varchar("doc_id", 128),
            _varchar("content", MAX_CONTENT_LENGTH),
            _varchar("content_type", 64),
            _varchar("city", 64),
            _varchar("scenic_name", 256),
            _varchar("route_name", 256),
            _varchar("hotel_name", 256),
            _varchar("restaurant_name", 256),
            _varchar("source_file", 512),
            _varchar("source_path", 1024),
            _varchar("section_path", 2048),
            _int64("chunk_index"),
            _json("extra"),
            _int64("updated_at"),
            _vector(DENSE_FIELD, DataType.FLOAT_VECTOR, DENSE_DIM),
            _vector(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR),
        ],
        description="Travel knowledge base chunks",
        enable_dynamic_field=False,
    )
    return schema


def build_index_params():
    params = MilvusClient.prepare_index_params()
    params.add_index(
        field_name=DENSE_FIELD,
        index_name="idx_dense_hnsw",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    params.add_index(
        field_name=SPARSE_FIELD,
        index_name="idx_sparse_inverted",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    for field in SCALAR_INDEXES:
        params.add_index(
            field_name=field,
            index_name=f"idx_{field}",
            index_type="INVERTED",
        )
    return params


class MilvusRepository:
    def __init__(self, settings: Settings) -> None:
        self._collection = settings.milvus_collection
        self._client = MilvusClient(
            uri=settings.milvus_uri,
            token=settings.milvus_token or None,
            db_name=settings.milvus_db_name,
        )

    @property
    def client(self) -> MilvusClient:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._collection

    def has_collection(self) -> bool:
        return self._client.has_collection(self._collection)

    def create_collection(self, recreate: bool = False, consistency_level: str = "Strong") -> bool:
        """Create the collection if needed. Returns True when it was created."""
        if self.has_collection():
            if not recreate:
                return False
            self._client.drop_collection(self._collection)

        self._client.create_collection(
            collection_name=self._collection,
            schema=build_schema(),
            index_params=build_index_params(),
            consistency_level=consistency_level,
        )
        return True

    def ensure_loaded(self) -> dict:
        self._client.load_collection(self._collection)
        return self._client.get_load_state(self._collection)

    def describe(self) -> dict:
        return self._client.describe_collection(self._collection)

    def list_indexes(self) -> list[str]:
        return list(self._client.list_indexes(self._collection))

    def describe_index(self, index_name: str) -> dict:
        return self._client.describe_index(self._collection, index_name)

    def search_dense(
        self,
        vector: list[float],
        limit: int = 10,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        return self._client.search(
            self._collection,
            data=[vector],
            anns_field=DENSE_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 2)}},
            limit=limit,
            filter=filter_expr,
            output_fields=output_fields or OUTPUT_FIELDS,
        )[0]

    def search_sparse(
        self,
        vector: dict[int, float],
        limit: int = 10,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        return self._client.search(
            self._collection,
            data=[vector],
            anns_field=SPARSE_FIELD,
            search_params={"metric_type": "IP"},
            limit=limit,
            filter=filter_expr,
            output_fields=output_fields or OUTPUT_FIELDS,
        )[0]

    def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        limit: int = 10,
        dense_limit: int = 20,
        sparse_limit: int = 20,
        rrf_k: int = 60,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        requests = [
            AnnSearchRequest(
                data=[dense_vector],
                anns_field=DENSE_FIELD,
                param={"metric_type": "COSINE", "params": {"ef": max(64, dense_limit * 2)}},
                limit=dense_limit,
                expr=filter_expr or None,
            ),
            AnnSearchRequest(
                data=[sparse_vector],
                anns_field=SPARSE_FIELD,
                param={"metric_type": "IP"},
                limit=sparse_limit,
                expr=filter_expr or None,
            ),
        ]
        return self._client.hybrid_search(
            self._collection,
            requests,
            ranker=RRFRanker(rrf_k),
            limit=limit,
            output_fields=output_fields or OUTPUT_FIELDS,
        )[0]

    def upsert_chunks(self, rows: list[dict], batch_size: int = 64) -> int:
        written = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            self._client.upsert(self._collection, batch)
            written += len(batch)
        return written

    def delete_stale_chunks(self, doc_id: str, keep_chunk_ids: list[str]) -> None:
        expression = f"doc_id == {quote(doc_id)}"
        if keep_chunk_ids:
            joined = ", ".join(quote(chunk_id) for chunk_id in keep_chunk_ids)
            expression += f" and chunk_id not in [{joined}]"
        self._client.delete(self._collection, filter=expression)

    def delete_document_chunks(self, doc_id: str) -> None:
        self._client.delete(self._collection, filter=f"doc_id == {quote(doc_id)}")

    def flush(self) -> None:
        self._client.flush(self._collection)

    def stats(self) -> dict:
        return self._client.get_collection_stats(self._collection)

    def count_chunks(self) -> int:
        result = self._client.query(
            self._collection,
            filter='chunk_id != ""',
            output_fields=["count(*)"],
        )
        return int(result[0]["count(*)"])

    def validate(self) -> list[str]:
        """Return human-readable schema problems; empty list means healthy."""
        info = self.describe()
        problems: list[str] = []

        actual_fields = {field["name"]: field for field in info.get("fields", [])}
        expected = {
            "chunk_id": True,
            "doc_id": False,
            "content": False,
            "content_type": False,
            "city": False,
            "scenic_name": False,
            "route_name": False,
            "hotel_name": False,
            "restaurant_name": False,
            "source_file": False,
            "source_path": False,
            "section_path": False,
            "chunk_index": False,
            "extra": False,
            "updated_at": False,
            DENSE_FIELD: False,
            SPARSE_FIELD: False,
        }
        for name, is_primary in expected.items():
            field = actual_fields.get(name)
            if field is None:
                problems.append(f"missing field: {name}")
            elif bool(field.get("is_primary")) != is_primary:
                problems.append(f"unexpected primary flag on {name}")

        dense = actual_fields.get(DENSE_FIELD) or {}
        if dense.get("type") != DataType.FLOAT_VECTOR:
            problems.append(f"{DENSE_FIELD} is not FLOAT_VECTOR")
        params = dense.get("params") or {}
        if int(params.get("dim", 0)) != DENSE_DIM:
            problems.append(f"{DENSE_FIELD} dimension is {params.get('dim')}, expected {DENSE_DIM}")

        indexes = {
            name: self.describe_index(name) for name in self.list_indexes()
        }
        indexed_fields = {detail["field_name"]: detail for detail in indexes.values()}
        for field in (DENSE_FIELD, SPARSE_FIELD, *SCALAR_INDEXES):
            if field not in indexed_fields:
                problems.append(f"missing index on {field}")

        dense_index = indexed_fields.get(DENSE_FIELD) or {}
        if dense_index and dense_index.get("index_type") != "HNSW":
            problems.append(f"{DENSE_FIELD} index is not HNSW")
        if dense_index and dense_index.get("metric_type") != "COSINE":
            problems.append(f"{DENSE_FIELD} metric is not COSINE")

        sparse_index = indexed_fields.get(SPARSE_FIELD) or {}
        if sparse_index and sparse_index.get("metric_type") != "IP":
            problems.append(f"{SPARSE_FIELD} metric is not IP")

        return problems

    def close(self) -> None:
        self._client.close()


def _varchar(name: str, max_length: int, primary: bool = False):
    kwargs = {"max_length": max_length, "is_primary": primary, "auto_id": False}
    if primary:
        kwargs["description"] = "Format: {doc_id}:{chunk_index}"
    return _make_field(name, DataType.VARCHAR, **kwargs)


def _int64(name: str):
    return _make_field(name, DataType.INT64)


def _json(name: str):
    return _make_field(name, DataType.JSON, nullable=True)


def _vector(name: str, dtype, dim: int | None = None):
    kwargs = {"dim": dim} if dim is not None else {}
    return _make_field(name, dtype, **kwargs)


def _make_field(name: str, dtype, **kwargs):
    from pymilvus import FieldSchema

    return FieldSchema(name=name, dtype=dtype, **kwargs)
