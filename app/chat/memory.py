from datetime import UTC, datetime

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, ReturnDocument

from app.core.exceptions import NotFoundError
from app.repositories.mongo import CHAT_MESSAGES, CHAT_SESSIONS, MongoRepository


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_object_id(value: str, *, resource: str = "session") -> ObjectId:
    if not ObjectId.is_valid(value):
        raise NotFoundError(f"{resource} not found")
    return ObjectId(value)


class ChatMemory:
    def __init__(self, mongo: MongoRepository) -> None:
        self._db = mongo.db

    async def get_or_create_session(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        title: str,
    ) -> dict:
        if session_id:
            document = await self._db[CHAT_SESSIONS].find_one(
                {"_id": parse_object_id(session_id)}
            )
            if document is None:
                raise NotFoundError("session not found")
            return _session_out(document)

        now = utc_now()
        document = {
            "user_id": user_id,
            "title": title[:60] or "新会话",
            "summary": "",
            "message_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        result = await self._db[CHAT_SESSIONS].insert_one(document)
        document["_id"] = result.inserted_id
        return _session_out(document)

    async def get_session(self, session_id: str) -> dict:
        document = await self._db[CHAT_SESSIONS].find_one(
            {"_id": parse_object_id(session_id)}
        )
        if document is None:
            raise NotFoundError("session not found")
        return _session_out(document)

    async def load_history(
        self,
        session_id: str,
        limit: int,
        *,
        exclude_message_id: str | None = None,
    ) -> list[dict]:
        query: dict = {"session_id": session_id}
        if exclude_message_id and ObjectId.is_valid(exclude_message_id):
            query["_id"] = {"$ne": ObjectId(exclude_message_id)}

        cursor = (
            self._db[CHAT_MESSAGES]
            .find(query)
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        documents = await cursor.to_list(length=limit)
        documents.reverse()
        return [_message_out(document) for document in documents]

    async def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        intent: str = "",
        token_usage: dict | None = None,
        status: str = "completed",
    ) -> str:
        now = utc_now()
        document = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "intent": intent,
            "token_usage": token_usage,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        result = await self._db[CHAT_MESSAGES].insert_one(document)

        await self._db[CHAT_SESSIONS].update_one(
            {"_id": parse_object_id(session_id)},
            {
                "$inc": {"message_count": 1},
                "$set": {"updated_at": now},
            },
        )
        return str(result.inserted_id)

    async def update_message(self, message_id: str, **fields) -> None:
        allowed = {"content", "citations", "intent", "token_usage", "status"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = utc_now()
        await self._db[CHAT_MESSAGES].update_one(
            {"_id": parse_object_id(message_id, resource="message")},
            {"$set": updates},
        )

    async def message_status(self, message_id: str) -> str | None:
        document = await self._db[CHAT_MESSAGES].find_one(
            {"_id": parse_object_id(message_id, resource="message")},
            {"status": 1},
        )
        return document.get("status") if document else None

    async def list_sessions(
        self,
        *,
        user_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[int, list[dict]]:
        query = {"user_id": user_id} if user_id is not None else {}
        total = await self._db[CHAT_SESSIONS].count_documents(query)
        cursor = (
            self._db[CHAT_SESSIONS]
            .find(query)
            .sort("updated_at", DESCENDING)
            .skip(offset)
            .limit(limit)
        )
        sessions = [_session_out(document) for document in await cursor.to_list(length=limit)]
        return total, sessions

    async def list_messages(self, session_id: str, limit: int = 200) -> list[dict]:
        cursor = (
            self._db[CHAT_MESSAGES]
            .find({"session_id": session_id})
            .sort("created_at", ASCENDING)
            .limit(limit)
        )
        return [_message_out(document) for document in await cursor.to_list(length=limit)]

    async def delete_session(self, session_id: str) -> bool:
        session_object_id = parse_object_id(session_id)
        session = await self._db[CHAT_SESSIONS].find_one_and_delete(
            {"_id": session_object_id},
            return_document=ReturnDocument.BEFORE,
        )
        await self._db[CHAT_MESSAGES].delete_many({"session_id": session_id})
        return session is not None

    async def refresh_summary(self, session_id: str, summary: str) -> None:
        await self._db[CHAT_SESSIONS].update_one(
            {"_id": parse_object_id(session_id)},
            {"$set": {"summary": summary, "updated_at": utc_now()}},
        )


def _session_out(document: dict) -> dict:
    data = dict(document)
    data["session_id"] = str(data.pop("_id"))
    return data


def _message_out(document: dict) -> dict:
    data = dict(document)
    data["message_id"] = str(data.pop("_id"))
    return data
