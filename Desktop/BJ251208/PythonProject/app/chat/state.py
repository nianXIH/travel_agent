from dataclasses import dataclass, field
from typing import TypedDict


class ChatState(TypedDict, total=False):
    query: str
    session_id: str
    user_id: str | None
    user_message_id: str
    top_k: int
    city: str | None
    content_type: str | None
    summary: str
    history: list[dict]
    rewritten_query: str
    filters: dict
    intent: str
    citations: list[dict]
    retrieval_info: dict
    assistant_message_id: str
    answer: str
    token_usage: dict | None


@dataclass
class ChatRuntime:
    """Mutable progress shared between graph nodes and the SSE consumer."""

    assistant_message_id: str | None = None
    citations: list[dict] = field(default_factory=list)
    intent: str = ""
    rewritten_query: str = ""
    answer: str = ""
    persisted: bool = False
