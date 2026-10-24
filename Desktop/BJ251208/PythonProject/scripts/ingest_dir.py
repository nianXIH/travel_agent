"""Ingest Markdown documents into Milvus and MongoDB."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.milvus_repo import MilvusRepository


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="re-import unchanged files")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N files")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = IngestionPipeline(settings)
    summary = await pipeline.run(
        directory=args.directory,
        force=args.force,
        limit=args.limit,
    )

    print()
    print(
        f"done: total={summary.total} imported={summary.succeeded} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"chunks={summary.chunks_written} in {summary.duration_seconds}s"
    )
    print(f"task_id={summary.task_id}")
    if summary.errors:
        for error in summary.errors:
            print(f"  error: {error['file']}: {error['error']}")

    milvus = MilvusRepository(settings)
    try:
        print(f"milvus live chunks: {milvus.count_chunks()}")
    finally:
        milvus.close()

    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
