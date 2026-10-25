from typing import Annotated

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_chat_service, require_role, target_tenant
from app.chat.service import ChatService
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> ChatResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    return await service.chat(payload, tenant_id)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> EventSourceResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    events = await service.stream(payload, tenant_id)
    return EventSourceResponse(events, headers={"X-Accel-Buffering": "no"})
