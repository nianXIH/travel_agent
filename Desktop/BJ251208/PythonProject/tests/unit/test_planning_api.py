from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.api.routes import planning
from app.core.exceptions import AppError
from app.schemas.planning import (
    ItineraryPlan,
    ItineraryResponse,
    RecommendationResponse,
    RecommendationSet,
)


class FakePlanner:
    def __init__(self, error: AppError | None = None) -> None:
        self.error = error
        self.requests: list[dict] = []

    async def plan(self, request):
        self.requests.append({"type": "plan", "payload": request})
        if self.error is not None:
            raise self.error
        return ItineraryResponse(
            destination=request.destination,
            resolved_city="三亚",
            days=request.days,
            plan=ItineraryPlan(summary="行程概述"),
            elapsed_ms=5,
        )

    async def recommend(self, request):
        self.requests.append({"type": "recommend", "payload": request})
        if self.error is not None:
            raise self.error
        return RecommendationResponse(
            destination=request.destination,
            resolved_city="三亚",
            category=request.category,
            recommendations=RecommendationSet(summary="推荐概述"),
            elapsed_ms=5,
        )


def _app(planner: FakePlanner) -> FastAPI:
    app = FastAPI()
    app.state.planner = planner

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    app.include_router(planning.router, prefix="/api/v1")
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_itinerary_endpoint_returns_plan():
    planner = FakePlanner()
    app = _app(planner)

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/itineraries",
            json={"destination": "三亚", "days": 2, "interests": ["亲子"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["destination"] == "三亚"
    assert body["days"] == 2
    assert body["plan"]["summary"] == "行程概述"
    assert planner.requests[0]["payload"].days == 2
    assert planner.requests[0]["payload"].interests == ["亲子"]


async def test_itinerary_endpoint_validates_days():
    planner = FakePlanner()
    app = _app(planner)

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/itineraries",
            json={"destination": "三亚", "days": 0},
        )

    assert response.status_code == 422
    assert planner.requests == []


async def test_itinerary_endpoint_propagates_app_error():
    planner = FakePlanner(
        AppError("知识库中没有资料", status_code=404, code="NO_CONTEXT")
    )
    app = _app(planner)

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/itineraries",
            json={"destination": "火星"},
        )

    assert response.status_code == 404
    assert response.json()["code"] == "NO_CONTEXT"


async def test_recommendation_endpoint_returns_items():
    planner = FakePlanner()
    app = _app(planner)

    async with _client(app) as client:
        response = await client.post(
            "/api/v1/recommendations",
            json={"destination": "三亚", "category": "美食", "top_k": 2},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "美食"
    assert planner.requests[0]["payload"].top_k == 2
