from pathlib import Path

import pytest

from app.core.config import Settings
from app.ingestion import pipeline as pipeline_module
from app.ingestion.metadata import make_doc_id
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.scanner import (
    ScannedFile,
    file_sha256,
    scan_tenant_directory,
)
from app.retrieval.embedder import EmbeddingBatch


class FakeRepo:
    def __init__(self) -> None:
        self.tasks: list[tuple[str, str, dict]] = []
        self.documents: dict[tuple[str, str], dict] = {}
        self.failures: list[dict] = []
        self.deleted: list[dict] = []

    async def create_task(
        self, tenant_id: str, task_type: str, total: int, *, status: str = "running"
    ) -> str:
        task_id = f"task-{len(self.tasks) + 1}"
        self.tasks.append(
            (tenant_id, task_id, {"task_type": task_type, "total": total, "status": status})
        )
        return task_id

    async def get_document(self, tenant_id: str, source_path: str) -> dict | None:
        return self.documents.get((tenant_id, source_path))

    async def upsert_document(self, tenant_id: str, document: dict) -> dict:
        self.documents[(tenant_id, document["source_path"])] = {
            **document,
            "tenant_id": tenant_id,
        }
        return document

    async def mark_document_failed(self, tenant_id: str, **fields) -> None:
        self.failures.append({"tenant_id": tenant_id, **fields})

    async def list_documents(self, tenant_id: str) -> list[dict]:
        return [document for (owner, _), document in self.documents.items() if owner == tenant_id]

    async def mark_document_deleted(self, tenant_id: str, *, source_path: str, doc_id: str) -> None:
        self.deleted.append({"tenant_id": tenant_id, "source_path": source_path, "doc_id": doc_id})
        if (tenant_id, source_path) in self.documents:
            self.documents[(tenant_id, source_path)]["status"] = "deleted"

    async def update_task(self, tenant_id: str, task_id: str, **fields) -> None:
        self.tasks.append((tenant_id, task_id, fields))


class FakeMongo:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class FakeEmbedder:
    def encode(self, texts: list[str], batch_size: int | None = None) -> EmbeddingBatch:
        return EmbeddingBatch(
            dense=[[0.1, 0.2, 0.3, 0.4] for _ in texts],
            sparse=[{1: 1.0} for _ in texts],
        )


class FakeMilvus:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.deleted: list[tuple[str, str, list[str]]] = []
        self.deleted_docs: list[tuple[str, str]] = []
        self.flushed = False
        self.close_calls = 0

    def upsert_chunks(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def delete_stale_chunks(self, tenant_id: str, doc_id: str, chunk_ids: list[str]) -> None:
        self.deleted.append((tenant_id, doc_id, chunk_ids))

    def delete_document_chunks(self, tenant_id: str, doc_id: str) -> None:
        self.deleted_docs.append((tenant_id, doc_id))
        self.rows = [
            row
            for row in self.rows
            if not (row.get("tenant_id") == tenant_id and row.get("doc_id") == doc_id)
        ]

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.close_calls += 1


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

    summary = await pipeline.run_task("task-1", [_scanned(path, root)], tenant_id="default")

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.chunks_written == 1
    document = repo.documents[("default", "景点攻略/杭州景点推荐.md")]
    assert document["status"] == "imported"
    assert document["tenant_id"] == "default"
    assert document["doc_id"] == make_doc_id("default", "景点攻略/杭州景点推荐.md")
    assert repo.tasks[-1][0] == "default"
    assert repo.tasks[-1][1] == "task-1"
    assert repo.tasks[-1][2]["status"] == "success"
    assert repo.tasks[-1][2]["succeeded"] == 1
    assert milvus.flushed is True
    assert milvus.rows[0]["city"] == "杭州"
    assert milvus.rows[0]["tenant_id"] == "default"


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
        tenant_id="default",
    )

    assert summary.succeeded == 1
    assert summary.failed == 1
    assert summary.errors[0]["file"] == "景点攻略/空文件.md"
    assert repo.failures[0]["tenant_id"] == "default"
    assert repo.failures[0]["source_path"] == "景点攻略/空文件.md"
    assert repo.failures[0]["doc_id"] == make_doc_id("default", "景点攻略/空文件.md")
    assert repo.tasks[-1][2]["status"] == "partial"


async def test_prune_marks_missing_documents_deleted(tmp_path):
    root = tmp_path / "数据"
    kept = _write_document(
        root,
        "景点攻略/保留.md",
        "# 保留\n\n## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    repo = FakeRepo()
    repo.documents[("default", "景点攻略/删除.md")] = {
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

    summary = await pipeline.run_task(
        "task-prune",
        [_scanned(kept, root)],
        tenant_id="default",
        prune=True,
    )

    assert summary.pruned == 1
    assert summary.prune_failed == 0
    assert summary.stale_files == ["景点攻略/删除.md"]
    assert repo.deleted == [
        {"tenant_id": "default", "source_path": "景点攻略/删除.md", "doc_id": "doc_gone"}
    ]
    assert repo.documents[("default", "景点攻略/删除.md")]["status"] == "deleted"
    assert milvus.deleted_docs == [("default", "doc_gone")]
    assert repo.tasks[-1][2]["pruned"] == 1
    assert repo.tasks[-1][2]["status"] == "success"


async def test_rename_prunes_old_document_and_imports_new_one(tmp_path):
    root = tmp_path / "数据"
    new_path = _write_document(
        root,
        "景点攻略/杭州景点推荐.md",
        "# 杭州景点推荐\n\n## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    repo = FakeRepo()
    repo.documents[("default", "景点攻略/杭州攻略.md")] = {
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
        tenant_id="default",
        prune=True,
    )

    assert summary.succeeded == 1
    assert summary.pruned == 1
    assert ("default", "景点攻略/杭州景点推荐.md") in repo.documents
    assert repo.documents[("default", "景点攻略/杭州攻略.md")]["status"] == "deleted"
    assert milvus.deleted_docs == [("default", "doc_old")]


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

    first = await pipeline.run_task("task-first", [scanned], tenant_id="default")
    second = await pipeline.run_task("task-second", [scanned], tenant_id="default")

    assert first.succeeded == 1
    assert second.skipped == 1
    assert second.chunks_written == 0
    assert len(milvus.rows) == first.chunks_written

    forced = await pipeline.run_task("task-forced", [scanned], tenant_id="default", force=True)

    assert forced.succeeded == 1
    assert forced.chunks_written == first.chunks_written
    doc_id = repo.documents[("default", "景点攻略/杭州景点推荐.md")]["doc_id"]
    assert milvus.deleted[-1][0] == "default"
    assert milvus.deleted[-1][1] == doc_id


async def test_run_task_replaces_legacy_doc_id_with_tenant_scoped_id(tmp_path):
    root = tmp_path / "数据"
    path = _write_document(
        root,
        "景点攻略/杭州景点推荐.md",
        "# 杭州景点推荐\n\n## 西湖\n西湖适合清晨慢游，建议避开节假日。\n",
    )
    source_path = "景点攻略/杭州景点推荐.md"
    repo = FakeRepo()
    repo.documents[("default", source_path)] = {
        "tenant_id": "default",
        "source_path": source_path,
        "doc_id": "doc_legacy",
        "sha256": "legacy-digest",
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

    summary = await pipeline.run_task("task-legacy", [_scanned(path, root)], tenant_id="default")

    expected = make_doc_id("default", source_path)
    assert summary.succeeded == 1
    document = repo.documents[("default", source_path)]
    assert document["doc_id"] == expected
    assert document["doc_id"] != "doc_legacy"
    assert milvus.rows[0]["doc_id"] == expected
    assert milvus.rows[0]["chunk_id"] == f"{expected}:0"
    assert milvus.deleted[-1] == ("default", expected, [f"{expected}:0"])


async def test_tenant_imports_are_isolated(tmp_path):
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

    await pipeline.run_task("task-a", [scanned], tenant_id="default")
    await pipeline.run_task("task-b", [scanned], tenant_id="sanya")

    first = repo.documents[("default", "景点攻略/杭州景点推荐.md")]
    second = repo.documents[("sanya", "景点攻略/杭州景点推荐.md")]
    assert first["doc_id"] != second["doc_id"]
    assert first["tenant_id"] == "default"
    assert second["tenant_id"] == "sanya"
    assert [row["tenant_id"] for row in milvus.rows] == ["default", "sanya"]


def test_scan_tenant_directory_excludes_other_tenants(tmp_path):
    root = tmp_path / "数据"
    _write_document(root, "景点攻略/杭州景点推荐.md", "# 杭州\n\n## 西湖\n内容\n")
    _write_document(root, "tenants/sanya/景点攻略/三亚.md", "# 三亚\n\n## 亚龙湾\n内容\n")
    settings = _settings(root)

    default_files = scan_tenant_directory(settings, "default")
    sanya_files = scan_tenant_directory(settings, "sanya")

    assert [file.relative_path for file in default_files] == ["景点攻略/杭州景点推荐.md"]
    assert [file.relative_path for file in sanya_files] == ["景点攻略/三亚.md"]


async def test_run_scans_tenant_directory(tmp_path):
    root = tmp_path / "数据"
    settings = _settings(root)
    sanya_root = settings.tenant_data_path("sanya")
    _write_document(
        sanya_root,
        "景点攻略/三亚.md",
        "# 三亚\n\n## 亚龙湾\n亚龙湾适合清晨浮潜，建议避开节假日。\n",
    )
    repo = FakeRepo()
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        settings,
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=repo,
        milvus=milvus,
    )

    summary = await pipeline.run(tenant_id="sanya", progress=lambda *_args: None)

    assert summary.succeeded == 1
    assert milvus.rows[0]["tenant_id"] == "sanya"
    assert repo.documents[("sanya", "景点攻略/三亚.md")]["tenant_id"] == "sanya"
    assert all(entry[0] == "sanya" for entry in repo.tasks)


async def test_dry_run_reports_stale_documents_without_writing(tmp_path):
    root = tmp_path / "数据"
    _write_document(root, "保留.md", "# 保留\n\n## 内容\n西湖适合清晨慢游，建议避开节假日。\n")
    repo = FakeRepo()
    repo.documents[("default", "删除.md")] = {
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

    with pytest.raises(ValueError, match="complete data directory"):
        await pipeline.run_task(
            "task-partial", [_scanned(first, root)], tenant_id="default", prune=True
        )


async def test_prune_full_scan_ignores_other_tenant_directories(tmp_path):
    root = tmp_path / "数据"
    kept = _write_document(
        root, "保留.md", "# 保留\n\n## 内容\n西湖适合清晨慢游，建议避开节假日。\n"
    )
    _write_document(root, "tenants/sanya/三亚.md", "# 三亚\n\n## 亚龙湾\n浮潜。\n")
    repo = FakeRepo()
    repo.documents[("default", "删除.md")] = {
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

    summary = await pipeline.run_task(
        "task-prune-tenant", [_scanned(kept, root)], tenant_id="default", prune=True
    )

    assert summary.pruned == 1
    assert summary.stale_files == ["删除.md"]
    assert milvus.deleted_docs == [("default", "doc_gone")]


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
        await pipeline.run_task("task-empty", [], tenant_id="default", prune=True)


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

    with pytest.raises(ValueError, match="tenant data directory"):
        await pipeline.run(directory=subdir, prune=True)


async def test_prune_rejects_other_tenant_directory(tmp_path):
    settings = _settings(tmp_path / "数据")
    sanya_root = settings.tenant_data_path("sanya")
    _write_document(sanya_root, "景点攻略/三亚.md", "# 三亚\n\n## 亚龙湾\n浮潜。\n")
    pipeline = IngestionPipeline(
        settings,
        embedder=FakeEmbedder(),
        mongo=FakeMongo(),
        repo=FakeRepo(),
        milvus=FakeMilvus(),
    )

    with pytest.raises(ValueError, match="tenant data directory"):
        await pipeline.run(directory=sanya_root, tenant_id="default", prune=True)


async def test_run_does_not_close_injected_clients(tmp_path):
    root = tmp_path / "数据"
    root.mkdir()
    mongo = FakeMongo()
    milvus = FakeMilvus()
    pipeline = IngestionPipeline(
        _settings(root),
        embedder=FakeEmbedder(),
        mongo=mongo,
        milvus=milvus,
        repo=FakeRepo(),
    )

    summary = await pipeline.run(prune=True, dry_run=True)

    assert summary.dry_run is True
    assert pipeline._owns_mongo is False
    assert pipeline._owns_milvus is False
    assert mongo.close_calls == 0
    assert milvus.close_calls == 0


async def test_run_closes_clients_it_constructs(tmp_path, monkeypatch):
    root = tmp_path / "数据"
    root.mkdir()
    closed: list[str] = []

    class OwnedMongo:
        def __init__(self, settings) -> None:
            pass

        async def close(self) -> None:
            closed.append("mongo")

    class OwnedMilvus:
        def __init__(self, settings) -> None:
            pass

        def close(self) -> None:
            closed.append("milvus")

    monkeypatch.setattr(pipeline_module, "MongoRepository", OwnedMongo)
    monkeypatch.setattr(pipeline_module, "MilvusRepository", OwnedMilvus)

    pipeline = IngestionPipeline(_settings(root), embedder=FakeEmbedder(), repo=FakeRepo())
    assert pipeline._owns_mongo is True
    assert pipeline._owns_milvus is True

    with pytest.raises(ValueError, match="empty scan"):
        await pipeline.run(prune=True)

    assert closed == ["mongo", "milvus"]
