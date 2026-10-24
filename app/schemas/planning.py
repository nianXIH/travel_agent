from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.document import Citation

TravelerType = Literal["solo", "couple", "family", "elderly", "friends"]
Pace = Literal["relaxed", "balanced", "intensive"]
RecommendationCategory = Literal["景点", "美食", "酒店", "线路", "综合"]
StopTime = Literal["morning", "afternoon", "evening"]

_TIME_ALIASES = {
    "上午": "morning",
    "早晨": "morning",
    "早上": "morning",
    "中午": "afternoon",
    "下午": "afternoon",
    "傍晚": "evening",
    "晚上": "evening",
    "夜间": "evening",
}


class _PreferenceFields(BaseModel):
    destination: str = Field(min_length=1, max_length=50)
    interests: list[str] = Field(default_factory=list, max_length=5)
    traveler_type: TravelerType = "solo"
    notes: str | None = Field(default=None, max_length=500)

    @field_validator("destination")
    @classmethod
    def _strip_destination(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("destination must not be blank")
        return value

    @field_validator("interests")
    @classmethod
    def _clean_interests(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if value and value not in cleaned:
                cleaned.append(value)
        return cleaned


class ItineraryRequest(_PreferenceFields):
    days: int = Field(default=3, ge=1, le=10)
    travelers: int = Field(default=1, ge=1, le=20)
    budget: int | None = Field(default=None, ge=0)
    pace: Pace = "balanced"
    hotel_preference: str | None = Field(default=None, max_length=50)


class PlanStop(BaseModel):
    time: StopTime
    title: str
    activity: str = ""
    duration_hours: float | None = Field(default=None, ge=0, le=24)
    tips: str = ""
    sources: list[int] = Field(default_factory=list)

    @field_validator("time", mode="before")
    @classmethod
    def _normalize_time(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().lower()
            return _TIME_ALIASES.get(value, value)
        return value


class PlanDay(BaseModel):
    day: int = Field(ge=1)
    theme: str = ""
    area: str = ""
    stops: list[PlanStop] = Field(default_factory=list)
    meals: list[str] = Field(default_factory=list)
    hotel: str = ""
    budget_estimate: int | None = Field(default=None, ge=0)


class BudgetBreakdown(BaseModel):
    transport: int = Field(default=0, ge=0)
    accommodation: int = Field(default=0, ge=0)
    food: int = Field(default=0, ge=0)
    tickets: int = Field(default=0, ge=0)
    other: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)


class ItineraryPlan(BaseModel):
    summary: str = ""
    days: list[PlanDay] = Field(default_factory=list)
    budget: BudgetBreakdown | None = None
    tips: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)


class ItineraryResponse(BaseModel):
    destination: str
    resolved_city: str | None = None
    days: int
    plan: ItineraryPlan
    citations: list[Citation] = Field(default_factory=list)
    retrieval: dict | None = None
    usage: dict | None = None
    elapsed_ms: int


class RecommendationRequest(_PreferenceFields):
    category: RecommendationCategory = "综合"
    top_k: int = Field(default=5, ge=1, le=10)


class RecommendationItem(BaseModel):
    name: str
    category: str = ""
    reason: str = ""
    highlights: list[str] = Field(default_factory=list)
    best_for: str = ""
    sources: list[int] = Field(default_factory=list)


class RecommendationSet(BaseModel):
    summary: str = ""
    items: list[RecommendationItem] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    destination: str
    resolved_city: str | None = None
    category: RecommendationCategory
    recommendations: RecommendationSet
    citations: list[Citation] = Field(default_factory=list)
    retrieval: dict | None = None
    usage: dict | None = None
    elapsed_ms: int
