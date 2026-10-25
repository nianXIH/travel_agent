"""Migrate existing collections and vectors to the multi-tenant schema.

Dry-run by default. Use --apply to write MongoDB changes; back up MongoDB first.

--recreate-milvus re-creates the Milvus collection so it gains the tenant field
and index. It is destructive (all vectors are removed) and requires --apply; it
is skipped when the current schema is already valid. --force-recreate rebuilds
the collection even when the schema is valid and requires --apply
--recreate-milvus.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.tenancy import DEFAULT_TENANT_ID
from app.ingestion.metadata import make_doc_id
from app.repositories.mongo import (
    CHAT_MESSAGES,
    CHAT_SESSIONS,
    DOCUMENTS,
    INGESTION_TASKS,
    MongoRepository,
)
from app.retrieval.milvus_repo import MilvusRepository

TENANT_COLLECTIONS = (DOCUMENTS, INGESTION_TASKS, CHAT_SESSIONS, CHAT_MESSAGES)

LEGACY_INDEXES = {
    DOCUMENTS: [
        "uq_source_path",
        "ix_sha256",
        "ix_city_type",
        "ix_status_imported",
    ],
    INGESTION_TASKS: ["ix_status_created", "ix_created"],
    CHAT_SESSIONS: ["ix_user_updated", "ix_updated"],
    CHAT_MESSAGES: ["ix_session_created"],
}

# MongoDB's {field: None} matches both null and missing documents.
MISSING_TENANT = {"$or": [{"tenant_id": None}, {"tenant_id": ""}]}


def should_recreate(*, healthy: bool, recreate_requested: bool, force: bool) -> bool:
    return recreate_requested and (not healthy or force)


async def backfill_tenant_ids(mongo: MongoRepository, *, apply: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in TENANT_COLLECTIONS:
        counts[name] = await mongo.db[name].count_documents(MISSING_TENANT)
        if apply and counts[name]:
            await mongo.db[name].update_many(
                MISSING_TENANT, {"$set": {"tenant_id": DEFAULT_TENANT_ID}}
            )
    return counts


async def recompute_doc_ids(mongo: MongoRepository, *, apply: bool) -> tuple[int, list[str]]:
    changed = 0
    skipped: list[str] = []
    cursor = mongo.db[DOCUMENTS].find(
        {},
        {"_id": 1, "tenant_id": 1, "source_path": 1, "doc_id": 1},
    )
    async for document in cursor:
        source_path = document.get("source_path")
        if not source_path:
            skipped.append(str(document["_id"]))
            continue
        tenant_id = document.get("tenant_id") or DEFAULT_TENANT_ID
        doc_id = make_doc_id(tenant_id, source_path)
        if doc_id == document.get("doc_id"):
            continue
        changed += 1
        if apply:
            await mongo.db[DOCUMENTS].update_one(
                {"_id": document["_id"]},
                {"$set": {"tenant_id": tenant_id, "doc_id": doc_id}},
            )
    return changed, skipped


async def drop_legacy_indexes(mongo: MongoRepository, *, apply: bool) -> dict[str, list[str]]:
    present: dict[str, list[str]] = {}
    for collection, names in LEGACY_INDEXES.items():
        existing = await mongo.db[collection].index_information()
        found = [name for name in names if name in existing]
        present[collection] = found
        if apply:
            for name in found:
                await mongo.db[collection].drop_index(name)
    return present


def milvus_health(milvus: MilvusRepository) -> tuple[bool, list[str]]:
    try:
        if not milvus.has_collection():
            return False, ["collection missing"]
        problems = milvus.validate()
        return not problems, problems
    except Exception as exc:
        return False, [f"{type(exc).__name__}: {exc}"]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes")
    parser.add_argument(
        "--recreate-milvus",
        action="store_true",
        help="drop and recreate the Milvus collection (destructive)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="recreate Milvus even when the schema is valid (destructive)",
    )
    args = parser.parse_args()
    if args.recreate_milvus and not args.apply:
        parser.error("--recreate-milvus requires --apply")
    if args.force_recreate and not (args.apply and args.recreate_milvus):
        parser.error("--force-recreate requires --apply --recreate-milvus")

    settings = get_settings()
    mongo = MongoRepository(settings)
    milvus = MilvusRepository(settings)
    try:
        await mongo.ping()

        print(f"mode: {'apply' if args.apply else 'dry-run'}")
        if args.apply:
            print(
                "backup reminder: back up MongoDB before applying; --apply rewrites doc_id values"
            )

        healthy, problems = await asyncio.to_thread(milvus_health, milvus)
        recreate = args.apply and should_recreate(
            healthy=healthy,
            recreate_requested=args.recreate_milvus,
            force=args.force_recreate,
        )

        if args.apply and args.recreate_milvus and healthy and not args.force_recreate:
            print(
                "milvus recreation skipped: the tenant schema is already valid; "
                "pass --force-recreate to rebuild anyway"
            )

        if args.apply and not healthy and not recreate:
            print("milvus tenant schema is not ready:")
            for problem in problems:
                print(f"  problem: {problem}")
            print(
                "re-run with --apply --recreate-milvus to rebuild the collection; "
                "MongoDB was not modified"
            )
            return 2

        recreated = False
        if recreate:
            milvus.create_collection(
                recreate=True,
                consistency_level=settings.milvus_consistency_level,
            )
            milvus.ensure_loaded()
            healthy, problems = await asyncio.to_thread(milvus_health, milvus)
            recreated = True
            if not healthy:
                print("milvus recreate did not produce a valid tenant schema:")
                for problem in problems:
                    print(f"  problem: {problem}")
                print("MongoDB was not modified")
                return 2

        backfill = await backfill_tenant_ids(mongo, apply=args.apply)
        changed, skipped = await recompute_doc_ids(mongo, apply=args.apply)
        if args.apply:
            await mongo.ensure_schema()
        legacy = await drop_legacy_indexes(mongo, apply=args.apply)

        for name, count in backfill.items():
            print(f"backfill {name}: {count} document(s) without a usable tenant_id")
        print(f"doc ids to rewrite: {changed}")
        if skipped:
            print(f"documents without source_path skipped: {len(skipped)}")
            for object_id in skipped:
                print(f"  skipped: {object_id}")
        for collection, names in legacy.items():
            print(f"legacy indexes {collection}: {', '.join(names) or 'none'}")
        print(f"milvus tenant schema: {'ok' if healthy else 'not ready'}")
        for problem in problems:
            print(f"  problem: {problem}")
        if recreated:
            print("milvus collection recreated: all vectors were removed")

        if not args.apply:
            print("dry-run: no changes written; re-run with --apply")
            return 0

        print("next: .venv/Scripts/python scripts/ingest_dir.py --tenant default --force")
        if not healthy:
            print("milvus is missing the tenant schema; re-run with --apply --recreate-milvus")
            return 2
        return 0
    finally:
        milvus.close()
        await mongo.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
