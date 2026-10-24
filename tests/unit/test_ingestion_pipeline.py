from pathlib import Path

import pytest

from app.core.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import ScannedFile, file_sha256
from app.retrieval.embedder import EmbeddingBatch


class FakeRepo:
    def __init__(self) -> None:
        self.tasks: list[tuple[str, dict]] = []
        self.documents: dict[str, dict] = {}
        self.failures: list[dict] = []
        self.deleted: list[dict] = []

    async def get_document(self, source_path: str) -> dict | None:
        return self.documents.get(source_path)

    async def upsert_document(self, document: dict) -> dict:
        self.documents[document["source_path"]] = document
        return document

    async def mark_document_failed(self, **fields) -> None:
        self.failures.append(fields)

    async def list_documents(self) -> list[dict]:
        return list(self.documents.values())

    async def mark_document_deleted(self, *, source_path: str, doc_id: str) -> None:
        self.deleted.append({"source_path": source_path, "doc_id": doc_id})
        if source_path in self.documents:
            self.documents[source_path]["status"] = "deleted"

    async def update_task(self, task_id: str, **fields) -> None:
        self.tasks.append((task_id, fields))


class FakeMongo:
    async def close(self) -> None:
        pass


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
        self.deleted_docs: list[str] = []
        self.flushed = False

    def upsert_chunks(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def delete_stale_chunks(self, doc_id: str, chunk_ids: list[str]) -> None:
        self.deleted.append((doc_id, chunk_ids))

    def delete_document_chunks(self, doc_id: str) -> None:
        self.deleted_docs.append(doc_id)
        self.rows = [row for row in self.rows if row.get("doc_id") != doc_id]

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


async def test_prune_marks_missing_documents_deleted(tmp_path):
    root = tmp_path / "数据"
    kept = _write_document(
        root,
        "景点攻略/保留.md",
        "# 保留\n\n## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    repo = FakeRepo()
    repo.documents["景点攻略/删除.md"] = {
        "source_path": "景点攻略/删除.md",
        "doc_id": "doc_gone",
        "status": "imported",
    }
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=repo,
        milvus=milvus,
    )

    summary = await pipeline.run_task("task-prune", [_scanned(kept, root)], prune=True)

    assert summary.pruned == 1
    assert summary.prune_failed == 0
    assert summary.stale_files == ["景点攻略/删除.md"]
    assert repo.deleted == [{"source_path": "景点攻略/删除.md", "doc_id": "doc_gone"}]
    assert repo.documents["景点攻略/删除.md"]["status"] == "deleted"
    assert milvus.deleted_docs == ["doc_gone"]
    assert repo.tasks[-1][1]["pruned"] == 1
    assert repo.tasks[-1][1]["status"] == "success"


async def test_rename_prunes_old_document_and_imports_new_one(tmp_path):
    root = tmp_path / "数据"
    new_path = _write_document(
        root,
        "景点攻略/杭州景点推荐.md",
        "# 杭州景点推荐\n\n## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    repo = FakeRepo()
    repo.documents["景点攻略/杭州攻略.md"] = {
        "source_path": "景点攻略/杭州攻略.md",
        "doc_id": "doc_old",
        "status": "imported",
    }
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=repo,
        milvus=milvus,
    )

    summary = await pipeline.run_task(
        "task-rename",
        [_scanned(new_path, root)],
        prune=True,
    )

    assert summary.succeeded == 1
    assert summary.pruned == 1
    assert "景点攻略/杭州景点推荐.md" in repo.documents
    assert repo.documents["景点攻略/杭州攻略.md"]["status"] == "deleted"
    assert milvus.deleted_docs == ["doc_old"]


async def test_repeated_import_is_idempotent_and_force_replaces(tmp_path):
    root = tmp_path / "数据"
    path = _write_document(
        root,
        "景点攻略/杭州景点推荐.md",
        "# 杭州景点推荐\n\n## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    repo = FakeRepo()
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=repo,
        milvus=milvus,
    )
    scanned = _scanned(path, root)

    first = await pipeline.run_task("task-first", [scanned])
    second = await pipeline.run_task("task-second", [scanned])

    assert first.succeeded == 1
    assert second.skipped == 1
    assert second.chunks_written == 0
    assert len(milvus.rows) == first.chunks_written

    forced = await pipeline.run_task("task-forced", [scanned], force=True)

    assert forced.succeeded == 1
    assert forced.chunks_written == first.chunks_written
    doc_id = repo.documents["景点攻略/杭州景点推荐.md"]["doc_id"]
    assert milvus.deleted[-1][0] == doc_id


async def test_dry_run_reports_stale_documents_without_writing(tmp_path):
    root = tmp_path / "数据"
    _write_document(root, "保留.md", "# 保留\n\n## 内容\n西湖适合清晨慢游，建议避开节假日。\n")
    repo = FakeRepo()
    repo.documents["删除.md"] = {
        "source_path": "删除.md",
        "doc_id": "doc_gone",
        "status": "imported",
    }
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=repo,
        milvus=milvus,
    )

    summary = await pipeline.run(prune=True, dry_run=True)

    assert summary.dry_run is True
    assert summary.stale_files == ["删除.md"]
    assert summary.pruned == 0
    assert repo.tasks == []
    assert repo.deleted == []
    assert milvus.deleted_docs == []


async def test_prune_rejects_limit(tmp_path):
    root = tmp_path / "数据"
    _write_document(root, "a.md", "# A\n\n## Body\ncontent\n")
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=FakeRepo(),
        milvus=FakeMilvus(),
    )

    with pytest.raises(ValueError, match="limit"):
        await pipeline.run(prune=True, limit=1)


async def test_prune_rejects_empty_scan(tmp_path):
    root = tmp_path / "数据"
    root.mkdir()
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=FakeRepo(),
        milvus=FakeMilvus(),
    )

    with pytest.raises(ValueError, match="empty"):
        await pipeline.run(prune=True)


async def test_run_task_prune_rejects_partial_scan(tmp_path):
    root = tmp_path / "数据"
    first = _write_document(root, "a.md", "# A\n\n## Body\ncontent for chunking\n")
    _write_document(root, "b.md", "# B\n\n## Body\ncontent for chunking\n")
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=FakeRepo(),
        milvus=FakeMilvus(),
    )

    with pytest.raises(ValueError, match="complete DATA_DIR"):
        await pipeline.run_task("task-partial", [_scanned(first, root)], prune=True)


async def test_run_task_prune_rejects_empty_files(tmp_path):
    root = tmp_path / "数据"
    root.mkdir()
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=FakeRepo(),
        milvus=FakeMilvus(),
    )

    with pytest.raises(ValueError, match="empty scan"):
        await pipeline.run_task("task-empty", [], prune=True)


async def test_prune_rejects_non_data_directory(tmp_path):
    root = tmp_path / "数据"
    subdir = root / "景点攻略"
    _write_document(subdir, "杭州景点推荐.md", "# 杭州\n\n## 西湖\ncontent\n")
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=FakeRepo(),
        milvus=FakeMilvus(),
    )

    with pytest.raises(ValueError, match="DATA_DIR"):
        await pipeline.run(directory=subdir, prune=True)
