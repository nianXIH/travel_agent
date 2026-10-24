from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AdminFeatures(BaseModel):
    reranker: bool = False
    hybrid: bool = False
    web_search: bool = False


class AdminOverview(BaseModel):
    app_name: str
    app_env: str
    documents_by_status: dict[str, int] = Field(default_factory=dict)
    total_documents: int = 0
    tasks_by_status: dict[str, int] = Field(default_factory=dict)
    total_tasks: int = 0
    cities: dict[str, int] = Field(default_factory=dict)
    content_types: dict[str, int] = Field(default_factory=dict)
    chunks: int | None = None
    chunk_error: str = ""
    llm_configured: bool = False
    features: AdminFeatures = Field(default_factory=AdminFeatures)


class AdminDocumentSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_id: str = ""
    title: str = ""
    source_file: str = ""
    source_path: str = ""
    city: str = ""
    content_type: str = ""
    status: str = ""
    size: int = 0
    chunk_count: int = 0
    sha256: str = ""
    error: str = ""
    imported_at: datetime | None = None
    updated_at: datetime | None = None
    deleted_at: datetime | None = None


class AdminDocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[AdminDocumentSummary] = Field(default_factory=list)


class AdminChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = ""
    chunk_index: int = 0
    section_path: str = ""
    content: str = ""
    content_type: str = ""
    city: str = ""
    updated_at: int | None = None


class AdminDocumentDetail(AdminDocumentSummary):
    chunks: list[AdminChunk] = Field(default_factory=list)
    chunk_error: str = ""
