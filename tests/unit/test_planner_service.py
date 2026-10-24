import json

import pytest
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.core.exceptions import AppError, LLMNotConfiguredError, UpstreamError
from app.planner.service import PlannerService, resolve_city
from app.schemas.planning import ItineraryRequest, RecommendationRequest


class FakeRetrieval:
    def __init__(
        self,
        hits_by_type: dict[str, list[dict]] | None = None,
        fail_types: tuple[str, ...] = (),
    ) -> None:
        self.hits_by_type = hits_by_type or {}
        self.fail_types = set(fail_types)
        self.calls: list[dict] = []

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = 5,
        rerank: bool | None = None,
        city: str | None = None,
        content_type: str | None = None,
        **kwargs,
    ) -> list[dict]:
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "city": city,
                "content_type": content_type,
            }
        )
        if content_type in self.fail_types:
            raise RuntimeError("milvus unavailable")
        return [dict(hit) for hit in self.hits_by_type.get(content_type or "", [])][:top_k]


class FakeLLM:
    def __init__(
        self,
        payload: dict | str | None = None,
        *,
        configured: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.configured = configured
        self.payload = payload
        self.error = error
        self.calls: list[dict] = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        content = (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )
        return AIMessage(content=content)


def _hits(content_type: str, count: int = 2) -> list[dict]:
    return [
        {
            "chunk_id": f"{content_type}-{index}",
            "content": f"{content_type}资料{index}",
            "score": 0.9 - index * 0.01,
            "city": "三亚",
            "content_type": content_type,
            "source_file": f"{content_type}.md",
            "section_path": f"第{index}节",
        }
        for index in range(1, count + 1)
    ]


def _all_hits(count: int = 2) -> dict[str, list[dict]]:
    return {
        content_type: _hits(content_type, count)
        for content_type in ("景点介绍", "线路推荐", "酒店信息", "美食介绍", "交通指南")
    }


def _plan_payload(days: int = 2) -> dict:
    return {
        "summary": "三亚亲子行程",
        "days": [
            {
                "day": day,
                "theme": "海岛休闲",
                "area": "亚龙湾",
                "stops": [
                    {
                        "time": "上午",
                        "title": "亚龙湾沙滩",
                        "activity": "海边活动",
                        "duration_hours": 2.5,
                        "tips": "注意防晒",
                        "sources": [1, 99],
                    },
                    {
                        "time": "afternoon",
                        "title": "海鲜午餐",
                        "activity": "品尝当地海鲜",
                        "sources": [2],
                    },
                ],
                "meals": ["海鲜"],
                "hotel": "海景酒店",
                "budget_estimate": 800,
            }
            for day in range(1, days + 1)
        ],
        "budget": {
            "transport": 100,
            "accommodation": 500,
            "food": 200,
            "tickets": 100,
            "other": 0,
            "total": 900,
        },
        "tips": ["提前预约热门景点"],
        "alternatives": ["雨天改去室内场馆"],
    }


def _service(retrieval: FakeRetrieval, llm: FakeLLM) -> PlannerService:
    return PlannerService(Settings(), retrieval, llm)


async def test_plan_returns_grounded_itinerary_and_filters_sources():
    retrieval = FakeRetrieval(_all_hits())
    llm = FakeLLM(_plan_payload(days=2))
    service = _service(retrieval, llm)

    response = await service.plan(ItineraryRequest(destination="三亚", days=2))

    assert response.destination == "三亚"
    assert response.resolved_city == "三亚"
    assert response.days == 2
    assert len(response.plan.days) == 2
    assert response.plan.days[0].stops[0].time == "morning"
    assert response.plan.days[0].stops[0].sources == [1]
    assert response.plan.budget is not None
    assert response.plan.budget.total == 900
    assert len(response.citations) == 10
    assert response.retrieval is not None
    assert response.retrieval["degraded"] == []
    assert response.retrieval["categories"]["酒店信息"]["status"] == "ok"

    assert {call["content_type"] for call in retrieval.calls} == {
        "景点介绍",
        "线路推荐",
        "酒店信息",
        "美食介绍",
        "交通指南",
    }
    assert all(call["city"] == "三亚" for call in retrieval.calls)
    assert len(llm.calls) == 1
    assert llm.calls[0]["kwargs"]["temperature"] == Settings().planner_temperature
    assert "三亚" in llm.calls[0]["messages"][1].content


async def test_plan_returns_no_context_error_without_citations():
    retrieval = FakeRetrieval()
    llm = FakeLLM(_plan_payload())
    service = _service(retrieval, llm)

    with pytest.raises(AppError) as exc_info:
        await service.plan(ItineraryRequest(destination="三亚", days=2))

    assert exc_info.value.code == "NO_CONTEXT"
    assert llm.calls == []


async def test_plan_degrades_when_one_retrieval_fails():
    retrieval = FakeRetrieval(_all_hits(), fail_types=("酒店信息",))
    llm = FakeLLM(_plan_payload(days=2))
    service = _service(retrieval, llm)

    response = await service.plan(ItineraryRequest(destination="三亚", days=2))

    assert response.retrieval is not None
    assert response.retrieval["categories"]["酒店信息"]["status"] == "error"
    assert response.retrieval["degraded"] == ["酒店信息"]
    assert len(response.citations) == 8


async def test_plan_maps_destination_alias_to_city():
    retrieval = FakeRetrieval(_all_hits())
    llm = FakeLLM(_plan_payload(days=2))
    service = _service(retrieval, llm)

    response = await service.plan(ItineraryRequest(destination="昆明", days=2))

    assert response.resolved_city == "云南"
    assert all(call["city"] == "云南" for call in retrieval.calls)


async def test_plan_rejects_invalid_json():
    retrieval = FakeRetrieval(_all_hits())
    llm = FakeLLM("这不是 JSON")
    service = _service(retrieval, llm)

    with pytest.raises(UpstreamError) as exc_info:
        await service.plan(ItineraryRequest(destination="三亚"))

    assert exc_info.value.code == "INVALID_LLM_OUTPUT"


async def test_plan_rejects_invalid_schema():
    retrieval = FakeRetrieval(_all_hits())
    llm = FakeLLM({"days": [{"day": 0, "stops": []}]})
    service = _service(retrieval, llm)

    with pytest.raises(UpstreamError) as exc_info:
        await service.plan(ItineraryRequest(destination="三亚"))

    assert exc_info.value.code == "INVALID_LLM_OUTPUT"


async def test_plan_wraps_llm_failure():
    retrieval = FakeRetrieval(_all_hits())
    llm = FakeLLM(error=RuntimeError("upstream down"))
    service = _service(retrieval, llm)

    with pytest.raises(UpstreamError) as exc_info:
        await service.plan(ItineraryRequest(destination="三亚"))

    assert exc_info.value.code == "UPSTREAM_ERROR"


async def test_plan_requires_configured_llm():
    retrieval = FakeRetrieval(_all_hits())
    llm = FakeLLM(_plan_payload(), configured=False)
    service = _service(retrieval, llm)

    with pytest.raises(LLMNotConfiguredError):
        await service.plan(ItineraryRequest(destination="三亚"))

    assert retrieval.calls == []


async def test_recommend_uses_selected_category_and_limits_items():
    retrieval = FakeRetrieval({"美食介绍": _hits("美食介绍", count=3)})
    payload = {
        "summary": "美食推荐",
        "items": [
            {
                "name": f"餐厅{index}",
                "category": "美食",
                "reason": "本地风味",
                "highlights": ["招牌菜"],
                "best_for": "家庭",
                "sources": [1, 99],
            }
            for index in range(3)
        ],
        "tips": ["错峰用餐"],
    }
    llm = FakeLLM(payload)
    service = _service(retrieval, llm)

    response = await service.recommend(
        RecommendationRequest(destination="三亚", category="美食", top_k=2)
    )

    assert response.category == "美食"
    assert len(response.recommendations.items) == 2
    assert response.recommendations.items[0].sources == [1]
    assert [call["content_type"] for call in retrieval.calls] == ["美食介绍"]
    assert len(response.citations) == 2


async def test_recommend_composite_queries_three_content_types():
    retrieval = FakeRetrieval(_all_hits())
    payload = {
        "summary": "综合推荐",
        "items": [
            {
                "name": "亚龙湾",
                "category": "景点",
                "reason": "沙滩优质",
                "sources": [1],
            }
        ],
        "tips": [],
    }
    service = _service(retrieval, FakeLLM(payload))

    response = await service.recommend(RecommendationRequest(destination="三亚"))

    assert {call["content_type"] for call in retrieval.calls} == {
        "景点介绍",
        "美食介绍",
        "酒店信息",
    }
    assert response.recommendations.items[0].name == "亚龙湾"


def test_resolve_city_handles_names_and_aliases():
    assert resolve_city("三亚") == "三亚"
    assert resolve_city("昆明") == "云南"
    assert resolve_city("鼓浪屿") == "厦门"
    assert resolve_city("火星") is None
