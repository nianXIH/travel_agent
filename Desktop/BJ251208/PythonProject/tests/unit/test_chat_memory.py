from datetime import UTC, datetime, timedelta

import pytest
from bson import ObjectId
from pymongo import DESCENDING, ReturnDocument

from app.chat.memory import ChatMemory
from app.core.exceptions import NotFoundError
from app.repositories.mongo import CHAT_MESSAGES, CHAT_SESSIONS


def _matches(document: dict, query: dict) -> bool:
    for key, expected in query.items():
        actual = document.get(key)
        if isinstance(expected, dict):
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            continue
        if actual != expected:
            return False
    return True


class FakeCursor:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = [dict(document) for document in documents]

    def sort(self, field: str, direction: int) -> "FakeCursor":
        self.documents.sort(
            key=lambda document: (document.get(field) is None, document.get(field) or 0),
            reverse=direction == DESCENDING,
        )
        return self

    def skip(self, offset: int) -> "FakeCursor":
        self.documents = self.documents[offset:]
        return self

    def limit(self, limit: int) -> "FakeCursor":
        self.documents = self.documents[:limit]
        return self

    async def to_list(self, length: int | None = None) -> list[dict]:
        return [dict(document) for document in self.documents]


class InsertOneResult:
    def __init__(self, inserted_id: ObjectId) -> None:
        self.inserted_id = inserted_id


class FakeCollection:
    def __init__(self) -> None:
        self.documents: list[dict] = []

    async def find_one(self, query: dict, projection: dict | None = None) -> dict | None:
        for document in self.documents:
            if _matches(document, query):
                return dict(document)
        return None

    async def insert_one(self, document: dict) -> InsertOneResult:
        stored = dict(document)
        stored.setdefault("_id", ObjectId())
        self.documents.append(stored)
        return InsertOneResult(stored["_id"])

    async def update_one(self, query: dict, update: dict) -> None:
        for document in self.documents:
            if _matches(document, query):
                for key, value in update.get("$inc", {}).items():
                    document[key] = document.get(key, 0) + value
                document.update(update.get("$set", {}))
                return

    async def find_one_and_delete(
        self, query: dict, return_document: ReturnDocument = ReturnDocument.BEFORE
    ) -> dict | None:
        for position, document in enumerate(self.documents):
            if _matches(document, query):
                del self.documents[position]
                return dict(document)
        return None

    async def delete_many(self, query: dict) -> None:
        self.documents = [document for document in self.documents if not _matches(document, query)]

    async def count_documents(self, query: dict) -> int:
        return len([document for document in self.documents if _matches(document, query)])

    def find(self, query: dict, projection: dict | None = None) -> FakeCursor:
        return FakeCursor([document for document in self.documents if _matches(document, query)])


class FakeDb:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {
            CHAT_SESSIONS: FakeCollection(),
            CHAT_MESSAGES: FakeCollection(),
        }

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongo:
    def __init__(self) -> None:
        self.db = FakeDb()


NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
DEFAULT_SESSION_ID = ObjectId()
DEFAULT_MESSAGE_ID = ObjectId()
SANYA_SESSION_ID = ObjectId()
SANYA_MESSAGE_ID = ObjectId()


def _seed() -> FakeMongo:
    mongo = FakeMongo()
    mongo.db[CHAT_SESSIONS].documents = [
        {
            "_id": DEFAULT_SESSION_ID,
            "tenant_id": "default",
            "user_id": "u-default",
            "title": "默认会话",
            "summary": "",
            "message_count": 1,
            "created_at": NOW,
            "updated_at": NOW,
        },
        {
            "_id": SANYA_SESSION_ID,
            "tenant_id": "sanya",
            "user_id": "u-sanya",
            "title": "三亚会话",
            "summary": "",
            "message_count": 1,
            "created_at": NOW + timedelta(minutes=1),
            "updated_at": NOW + timedelta(minutes=1),
        },
    ]
    mongo.db[CHAT_MESSAGES].documents = [
        {
            "_id": DEFAULT_MESSAGE_ID,
            "tenant_id": "default",
            "session_id": str(DEFAULT_SESSION_ID),
            "role": "assistant",
            "content": "默认租户消息",
            "citations": [],
            "intent": "",
            "token_usage": None,
            "status": "completed",
            "created_at": NOW,
            "updated_at": NOW,
        },
        {
            "_id": SANYA_MESSAGE_ID,
            "tenant_id": "sanya",
            "session_id": str(SANYA_SESSION_ID),
            "role": "assistant",
            "content": "三亚租户消息",
            "citations": [],
            "intent": "",
            "token_usage": None,
            "status": "completed",
            "created_at": NOW + timedelta(minutes=1),
            "updated_at": NOW + timedelta(minutes=1),
        },
    ]
    return mongo


def _session_message_count(mongo: FakeMongo, session_id: ObjectId) -> int:
    session = mongo.db[CHAT_SESSIONS].documents[
        [document["_id"] for document in mongo.db[CHAT_SESSIONS].documents].index(session_id)
    ]
    return session["message_count"]


async def test_get_session_rejects_other_tenant():
    mongo = _seed()
    memory = ChatMemory(mongo)

    with pytest.raises(NotFoundError):
        await memory.get_session("sanya", str(DEFAULT_SESSION_ID))

    session = await memory.get_session("sanya", str(SANYA_SESSION_ID))
    assert session["title"] == "三亚会话"


async def test_delete_session_is_tenant_scoped():
    mongo = _seed()
    memory = ChatMemory(mongo)

    assert await memory.delete_session("sanya", str(DEFAULT_SESSION_ID)) is False
    session = await memory.get_session("default", str(DEFAULT_SESSION_ID))
    assert session["session_id"] == str(DEFAULT_SESSION_ID)
    assert len(mongo.db[CHAT_MESSAGES].documents) == 2

    assert await memory.delete_session("sanya", str(SANYA_SESSION_ID)) is True
    remaining = [str(document["_id"]) for document in mongo.db[CHAT_MESSAGES].documents]
    assert remaining == [str(DEFAULT_MESSAGE_ID)]


async def test_list_sessions_only_returns_own_tenant():
    memory = ChatMemory(_seed())

    total, sessions = await memory.list_sessions(
        tenant_id="sanya", user_id=None, limit=10, offset=0
    )

    assert total == 1
    assert [session["session_id"] for session in sessions] == [str(SANYA_SESSION_ID)]


async def test_list_messages_only_returns_own_tenant():
    memory = ChatMemory(_seed())

    messages = await memory.list_messages("sanya", str(SANYA_SESSION_ID))

    assert [message["message_id"] for message in messages] == [str(SANYA_MESSAGE_ID)]
    assert [message["content"] for message in messages] == ["三亚租户消息"]


async def test_add_message_stores_tenant_and_increments_only_own_session():
    mongo = _seed()
    memory = ChatMemory(mongo)

    message_id = await memory.add_message(
        tenant_id="sanya",
        session_id=str(SANYA_SESSION_ID),
        role="user",
        content="三亚问题",
    )

    stored = mongo.db[CHAT_MESSAGES].documents[-1]
    assert str(stored["_id"]) == message_id
    assert stored["tenant_id"] == "sanya"
    assert stored["session_id"] == str(SANYA_SESSION_ID)
    assert _session_message_count(mongo, SANYA_SESSION_ID) == 2
    assert _session_message_count(mongo, DEFAULT_SESSION_ID) == 1


async def test_update_message_does_not_cross_tenants():
    mongo = _seed()
    memory = ChatMemory(mongo)

    await memory.update_message("sanya", str(DEFAULT_MESSAGE_ID), content="被篡改")

    default_message = mongo.db[CHAT_MESSAGES].documents[0]
    assert default_message["content"] == "默认租户消息"
