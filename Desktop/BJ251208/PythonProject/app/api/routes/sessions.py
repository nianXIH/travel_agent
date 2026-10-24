from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_memory
from app.chat.memory import ChatMemory
from app.core.exceptions import NotFoundError
from app.schemas.chat import (
    ChatMessage,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionSummary,
)

router = APIRouter(tags=["sessions"])


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    memory: Annotated[ChatMemory, Depends(get_memory)],
    user_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SessionListResponse:
    total, sessions = await memory.list_sessions(
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    return SessionListResponse(
        total=total,
        sessions=[SessionSummary.model_validate(session) for session in sessions],
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    memory: Annotated[ChatMemory, Depends(get_memory)],
) -> SessionDetailResponse:
    session = await memory.get_session(session_id)
    messages = await memory.list_messages(session_id)
    return SessionDetailResponse(
        session=SessionSummary.model_validate(session),
        messages=[ChatMessage.model_validate(message) for message in messages],
    )


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str,
    memory: Annotated[ChatMemory, Depends(get_memory)],
) -> SessionDeleteResponse:
    deleted = await memory.delete_session(session_id)
    if not deleted:
        raise NotFoundError("session not found")
    return SessionDeleteResponse(session_id=session_id, deleted=True)
