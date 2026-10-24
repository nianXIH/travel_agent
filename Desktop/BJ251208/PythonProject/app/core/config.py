from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "旅游知识库"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=list)
    admin_token: str = ""

    data_dir: Path = Path("./数据")
    hf_home: Path = Path("./models")
    hf_endpoint: str = "https://hf-mirror.com"

    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.8-flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 60
    llm_enable_thinking: bool = False
    enrich_with_llm: bool = False

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_fp16: bool = True
    embedding_batch_size: int = 8
    embedding_max_length: int = 1024

    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512

    hybrid_enabled: bool = True
    retrieval_dense_top_k: int = 20
    retrieval_sparse_top_k: int = 20
    retrieval_candidate_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 5

    hyde_enabled: bool = True
    hyde_temperature: float = 0.3
    hyde_max_tokens: int = 320
    hyde_weight: float = 0.9

    web_search_enabled: bool = True
    web_search_mcp_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/mcps/EnhancedSearch/mcp"
    )
    web_search_tool_name: str = "search_pro"
    web_search_api_key: str = ""
    web_search_timeout_seconds: float = 8.0
    web_search_top_k: int = 5
    web_search_max_citations: int = 2
    web_search_min_citations: int = 1
    web_search_weight: float = 0.7

    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_db_name: str = "default"
    milvus_collection: str = "travel_chunks"
    milvus_consistency_level: str = "Strong"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "travel_kb"

    chat_history_max_messages: int = 12
    chat_summary_trigger: int = 20

    planner_temperature: float = 0.4
    planner_max_tokens: int = 4096
    planner_context_per_type: int = 3
    planner_max_citations: int = 12

    max_upload_mb: int = 50
    allowed_upload_suffixes: str = ".md,.markdown,.txt,.pdf,.docx"

    @property
    def data_path(self) -> Path:
        return _resolve(self.data_dir)

    @property
    def hf_home_path(self) -> Path:
        return _resolve(self.hf_home)

    @property
    def web_search_key(self) -> str:
        return self.web_search_api_key or self.llm_api_key

    @property
    def upload_suffixes(self) -> tuple[str, ...]:
        return tuple(
            suffix.strip().lower()
            for suffix in self.allowed_upload_suffixes.split(",")
            if suffix.strip()
        )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
