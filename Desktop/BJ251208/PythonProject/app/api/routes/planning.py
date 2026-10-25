from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_planner, require_role, target_tenant
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.planner.service import PlannerService
from app.schemas.planning import (
    ItineraryRequest,
    ItineraryResponse,
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter(tags=["planning"])


@router.post("/itineraries", response_model=ItineraryResponse)
async def plan_itinerary(
    payload: ItineraryRequest,
    service: Annotated[PlannerService, Depends(get_planner)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> ItineraryResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    return await service.plan(payload, tenant_id)


@router.post("/recommendations", response_model=RecommendationResponse)
async def recommend(
    payload: RecommendationRequest,
    service: Annotated[PlannerService, Depends(get_planner)],
    ctx: Annotated[TenantContext, Depends(require_role("viewer"))],
) -> RecommendationResponse:
    tenant_id = target_tenant(ctx) or DEFAULT_TENANT_ID
    return await service.recommend(payload, tenant_id)
