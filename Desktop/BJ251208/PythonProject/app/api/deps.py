from typing import Annotated

from fastapi import Depends, Header, Request

from app.chat.llm import ChatLLM
from app.chat.memory import ChatMemory
from app.chat.service import ChatService
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.tenancy import (
    DEFAULT_TENANT_ID,
    DEFAULT_TENANT_NAME,
    ROLE_RANK,
    TenantContext,
    TenantCredential,
    TenantRegistry,
)
from app.ingestion.task_repo import IngestionRepository
from app.planner.service import PlannerService
from app.repositories.mongo import MongoRepository
from app.retrieval.milvus_repo import MilvusRepository
from app.retrieval.service import RetrievalService
from app.workers.ingest_worker import IngestionWorker


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_mongo(request: Request) -> MongoRepository:
    return request.app.state.mongo


def get_milvus(request: Request) -> MilvusRepository:
    return request.app.state.milvus


def get_retrieval(request: Request) -> RetrievalService:
    return request.app.state.retrieval


def get_memory(request: Request) -> ChatMemory:
    return request.app.state.memory


def get_llm(request: Request) -> ChatLLM:
    return request.app.state.llm


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_planner(request: Request) -> PlannerService:
    return request.app.state.planner


def get_ingest_repo(request: Request) -> IngestionRepository:
    return request.app.state.ingest_repo


def get_ingest_worker(request: Request) -> IngestionWorker:
    return request.app.state.ingest_worker


def get_tenant_registry(request: Request) -> TenantRegistry:
    return request.app.state.tenants


def get_tenant(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> TenantContext:
    registry: TenantRegistry = request.app.state.tenants
    requested = (x_tenant_id or "").strip() or None

    if registry.open_mode:
        if requested and requested != DEFAULT_TENANT_ID:
            raise AppError("租户不存在", status_code=404, code="TENANT_NOT_FOUND")
        return TenantContext(
            tenant_id=DEFAULT_TENANT_ID,
            tenant_name=DEFAULT_TENANT_NAME,
            role="admin",
            requested_tenant_id=requested,
            open_mode=True,
            admin_verified=registry.verify_admin_token(x_admin_token),
        )

    if registry.verify_admin_token(x_admin_token):
        if requested and registry.get_tenant(requested) is None:
            raise AppError("租户不存在", status_code=404, code="TENANT_NOT_FOUND")
        return TenantContext(
            tenant_id=None,
            tenant_name="全局管理员",
            role="admin",
            is_global_admin=True,
            requested_tenant_id=requested,
            admin_verified=True,
        )

    credential: TenantCredential | None = registry.verify_api_key(x_api_key)
    if credential is None:
        raise AppError("需要有效的访问凭证", status_code=401, code="UNAUTHORIZED")
    if requested and requested != credential.tenant_id:
        raise AppError("无权访问其他租户", status_code=403, code="TENANT_MISMATCH")
    return TenantContext(
        tenant_id=credential.tenant_id,
        tenant_name=credential.tenant_name,
        role=credential.role,
        requested_tenant_id=requested,
    )


def require_role(minimum: str):
    def dependency(
        ctx: Annotated[TenantContext, Depends(get_tenant)],
    ) -> TenantContext:
        if ctx.is_global_admin or ctx.open_mode:
            return ctx
        if ROLE_RANK.get(ctx.role, 0) < ROLE_RANK[minimum]:
            raise AppError("权限不足", status_code=403, code="FORBIDDEN")
        return ctx

    return dependency


def require_admin(
    request: Request,
    ctx: Annotated[TenantContext, Depends(get_tenant)],
) -> TenantContext:
    if ctx.open_mode:
        expected = request.app.state.settings.admin_token
        if expected and not ctx.admin_verified:
            raise AppError("管理员令牌无效", status_code=401, code="UNAUTHORIZED")
        return ctx
    if not ctx.is_global_admin and ROLE_RANK.get(ctx.role, 0) < ROLE_RANK["admin"]:
        raise AppError("权限不足", status_code=403, code="FORBIDDEN")
    return ctx


def target_tenant(ctx: TenantContext, *, allow_all: bool = False) -> str | None:
    if not ctx.is_global_admin:
        return ctx.tenant_id
    if ctx.requested_tenant_id:
        return ctx.requested_tenant_id
    return None if allow_all else DEFAULT_TENANT_ID
