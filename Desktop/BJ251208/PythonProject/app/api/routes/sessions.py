from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_memory, require_role, target_tenant
from app.chat.memory import ChatMemory
from app.core.exceptions import NotFoundError
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
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
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
    user_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> SessionListResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    total, sessions = await memory.list_sessions(
        tenant_id=tenant_id, user_id=user_id, limit=limit, offset=offset
    )
    return SessionListResponse(
        total=total,
        sessions=[SessionSummary.model_validate(session) for session in sessions],
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    memory: Annotated[ChatMemory, Depends(get_memory)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> SessionDetailResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    session = await memory.get_session(tenant_id, session_id)
    messages = await memory.list_messages(tenant_id, session_id)
    return SessionDetailResponse(
        session=SessionSummary.model_validate(session),
        messages=[ChatMessage.model_validate(message) for message in messages],
    )


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str,
    memory: Annotated[ChatMemory, Depends(get_memory)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> SessionDeleteResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    deleted = await memory.delete_session(tenant_id, session_id)
    if not deleted:
        raise NotFoundError("session not found")
    return SessionDeleteResponse(session_id=session_id, deleted=True)
