from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_planner
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
) -> ItineraryResponse:
    return await service.plan(payload)


@router.post("/recommendations", response_model=RecommendationResponse)
async def recommend(
    payload: RecommendationRequest,
    service: Annotated[PlannerService, Depends(get_planner)],
) -> RecommendationResponse:
    return await service.recommend(payload)
