from datetime import datetime

from pydantic import BaseModel, Field


class IngestScanRequest(BaseModel):
    force: bool = False
    limit: int | None = Field(default=None, ge=1)
    prune: bool = False


class IngestStartResponse(BaseModel):
    task_id: str
    status: str = "queued"


class IngestionTaskSummary(BaseModel):
    task_id: str
    task_type: str
    status: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    pruned: int = 0
    prune_failed: int = 0
    current_file: str = ""
    errors: list[dict] = Field(default_factory=list)
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None


class IngestionTaskListResponse(BaseModel):
    total: int
    tasks: list[IngestionTaskSummary]
