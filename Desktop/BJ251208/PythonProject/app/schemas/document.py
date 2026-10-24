from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    doc_id: str = ""
    content: str = ""
    score: float = 0.0
    rerank_score: float | None = None
    content_type: str = ""
    city: str = ""
    scenic_name: str = ""
    route_name: str = ""
    hotel_name: str = ""
    restaurant_name: str = ""
    source_file: str = ""
    source_path: str = ""
    section_path: str = ""
    source_type: str = "kb"
    title: str = ""
    url: str = ""
    site: str = ""
    published_at: str = ""
    retrieval_sources: list[str] = Field(default_factory=list)
    fusion_score: float | None = None


class DocumentSummary(BaseModel):
    doc_id: str
    title: str = ""
    source_file: str = ""
    source_path: str = ""
    city: str = ""
    content_type: str = ""
    chunk_count: int = 0


class CatalogResponse(BaseModel):
    cities: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    document_count: int = 0
    city_counts: dict[str, int] = Field(default_factory=dict)
    content_type_counts: dict[str, int] = Field(default_factory=dict)
