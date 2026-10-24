from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.document import Citation


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank: bool | None = None
    city: str | None = None
    content_type: str | None = None
    scenic_name: str | None = None
    route_name: str | None = None
    hotel_name: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    elapsed_ms: int
    citations: list[Citation]
