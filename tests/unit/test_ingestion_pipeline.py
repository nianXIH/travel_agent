from pathlib import Path

from app.core.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import ScannedFile, file_sha256
from app.retrieval.embedder import EmbeddingBatch


class FakeRepo:
    def __init__(self) -> None:
        self.tasks: list[tuple[str, dict]] = []
        self.documents: dict[str, dict] = {}
        self.failures: list[dict] = []

    async def get_document(self, source_path: str) -> dict | None:
        return self.documents.get(source_path)

    async def upsert_document(self, document: dict) -> dict:
        self.documents[document["source_path"]] = document
        return document

    async def mark_document_failed(self, **fields) -> None:
        self.failures.append(fields)

    async def update_task(self, task_id: str, **fields) -> None:
        self.tasks.append((task_id, fields))


class FakeEmbedder:
    def encode(self, texts: list[str], batch_size: int | None = None) -> EmbeddingBatch:
        return EmbeddingBatch(
            dense=[[0.1, 0.2, 0.3, 0.4] for _ in texts],
            sparse=[{1: 1.0} for _ in texts],
        )


class FakeMilvus:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.deleted: list[tuple[str, list[str]]] = []
        self.flushed = False

    def upsert_chunks(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def delete_stale_chunks(self, doc_id: str, chunk_ids: list[str]) -> None:
        self.deleted.append((doc_id, chunk_ids))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        pass


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, embedding_device="cpu", reranker_enabled=False)


def _scanned(path: Path, root: Path) -> ScannedFile:
    stat = path.stat()
    return ScannedFile(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        size=stat.st_size,
        mtime=stat.st_mtime,
        sha256=file_sha256(path),
    )


def _write_document(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


async def test_run_task_imports_documents_and_marks_success(tmp_path):
    root = tmp_path / "数据"
    path = _write_document(
        root,
        "景点攻略/杭州景点推荐.md",
        "# 杭州景点推荐\n\n## 元数据\n- 城市：杭州\n- 内容类型：景点介绍\n\n"
        "## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    repo = FakeRepo()
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        _settings(tmp_path),
        embedder=FakeEmbedder(),
        repo=repo,
        milvus=milvus,
    )

    summary = await pipeline.run_task("task-1", [_scanned(path, root)])

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.chunks_written == 1
    assert repo.documents["景点攻略/杭州景点推荐.md"]["status"] == "imported"
    assert repo.tasks[-1][1]["status"] == "success"
    assert repo.tasks[-1][1]["succeeded"] == 1
    assert milvus.flushed is True
    assert milvus.rows[0]["city"] == "杭州"


async def test_run_task_isolates_failed_files(tmp_path):
    root = tmp_path / "数据"
    good = _write_document(
        root,
        "景点攻略/杭州景点推荐.md",
        "# 杭州景点推荐\n\n## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    bad = _write_document(root, "景点攻略/空文件.md", "# 空文件\n")
    repo = FakeRepo()
    pipeline = IngestionPipeline(
        _settings(tmp_path),
        embedder=FakeEmbedder(),
        repo=repo,
        milvus=FakeMilvus(),
    )

    summary = await pipeline.run_task(
        "task-2",
        [_scanned(good, root), _scanned(bad, root)],
    )

    assert summary.succeeded == 1
    assert summary.failed == 1
    assert summary.errors[0]["file"] == "景点攻略/空文件.md"
    assert repo.failures[0]["source_path"] == "景点攻略/空文件.md"
    assert repo.tasks[-1][1]["status"] == "partial"
