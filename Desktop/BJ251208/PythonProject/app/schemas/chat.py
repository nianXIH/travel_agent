from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.document import Citation


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None
    user_id: str | None = Field(default=None, max_length=128)
    top_k: int = Field(default=5, ge=1, le=10)
    city: str | None = None
    content_type: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    rewritten_query: str
    intent: str
    citations: list[Citation]
    retrieval: dict | None = None
    usage: dict | None = None
    elapsed_ms: int


class SessionSummary(BaseModel):
    session_id: str
    title: str
    summary: str = ""
    message_count: int = 0
    user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatMessage(BaseModel):
    message_id: str
    role: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    intent: str = ""
    token_usage: dict | None = None
    status: str = "completed"
    created_at: datetime | None = None


class SessionListResponse(BaseModel):
    total: int
    sessions: list[SessionSummary]


class SessionDetailResponse(BaseModel):
    session: SessionSummary
    messages: list[ChatMessage]


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool
