"""Ingest Markdown documents into Milvus and MongoDB."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, TENANT_ID_RE
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.milvus_repo import MilvusRepository


def tenant_id(value: str) -> str:
    if TENANT_ID_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(f"invalid tenant id: {value}")
    return value


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="re-import unchanged files")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N files")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete documents whose source file no longer exists (DATA_DIR scans only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report stale documents without writing; requires --prune",
    )
    parser.add_argument(
        "--tenant",
        type=tenant_id,
        default=DEFAULT_TENANT_ID,
        help="tenant id (default: default)",
    )
    args = parser.parse_args()

    settings = get_settings()
    pipeline = IngestionPipeline(settings)
    try:
        summary = await pipeline.run(
            directory=args.directory,
            tenant_id=args.tenant,
            force=args.force,
            limit=args.limit,
            prune=args.prune,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    if summary.dry_run:
        print(f"dry-run: files={summary.total} stale={len(summary.stale_files)}")
        for path in summary.stale_files:
            print(f"  stale: {path}")
        return 0

    print()
    print(
        f"done: total={summary.total} imported={summary.succeeded} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"pruned={summary.pruned} prune_failed={summary.prune_failed} "
        f"chunks={summary.chunks_written} in {summary.duration_seconds}s"
    )
    print(f"task_id={summary.task_id}")
    for path in summary.stale_files:
        print(f"  pruned: {path}")
    if summary.errors:
        for error in summary.errors:
            print(f"  error: {error['file']}: {error['error']}")

    milvus = MilvusRepository(settings)
    try:
        print(f"milvus live chunks: {milvus.count_chunks(args.tenant)}")
    finally:
        milvus.close()

    return 0 if summary.failed == 0 and summary.prune_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
