"""Create the Milvus collection and MongoDB indexes (idempotent)."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings, get_settings
from app.repositories.mongo import COLLECTIONS, MongoRepository
from app.retrieval.milvus_repo import MilvusRepository


async def init_mongo(settings: Settings) -> bool:
    repo = MongoRepository(settings)
    try:
        print(f"[mongo] uri={settings.mongo_uri} db={settings.mongo_db}")
        if not await repo.ping():
            print("[mongo] ping failed")
            return False
        print("[mongo] ping ok")

        await repo.ensure_schema()
        for name in COLLECTIONS:
            cursor = await repo.db[name].list_indexes()
            indexes = await cursor.to_list(length=None)
            print(f"[mongo] {name}: indexes={len(indexes)}")
            for index in indexes:
                key_items = dict(index["key"]).items()
                keys = ", ".join(f"{key}:{direction}" for key, direction in key_items)
                flags = " unique" if index.get("unique") else ""
                print(f"        - {index['name']} ({keys}){flags}")
        return True
    finally:
        await repo.close()


def init_milvus(settings: Settings, recreate: bool) -> bool:
    repo = MilvusRepository(settings)
    try:
        print(
            f"[milvus] uri={settings.milvus_uri} db={settings.milvus_db_name} "
            f"collection={settings.milvus_collection}"
        )
        created = repo.create_collection(
            recreate=recreate,
            consistency_level=settings.milvus_consistency_level,
        )
        print(f"[milvus] collection {'created' if created else 'already exists, kept as is'}")

        problems = repo.validate()
        if problems:
            for problem in problems:
                print(f"[milvus] schema problem: {problem}")
            return False

        state = repo.ensure_loaded()
        print(f"[milvus] load state: {state.get('state')}")

        info = repo.describe()
        print(
            f"[milvus] fields={len(info.get('fields', []))} "
            f"consistency={info.get('consistency_level_name')}"
        )
        for index_name in repo.list_indexes():
            detail = repo.describe_index(index_name)
            print(
                f"        - {index_name}: field={detail.get('field_name')} "
                f"type={detail.get('index_type')} metric={detail.get('metric_type')} "
                f"state={detail.get('state')}"
            )
        return True
    finally:
        repo.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate-milvus",
        action="store_true",
        help="drop and recreate the Milvus collection (destroys all vectors)",
    )
    parser.add_argument("--skip-milvus", action="store_true")
    parser.add_argument("--skip-mongo", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ok = True

    if not args.skip_mongo:
        ok = await init_mongo(settings) and ok
    if not args.skip_milvus:
        ok = init_milvus(settings, recreate=args.recreate_milvus) and ok

    print("[result] OK" if ok else "[result] FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
