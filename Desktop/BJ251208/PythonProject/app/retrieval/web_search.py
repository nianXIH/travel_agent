import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from loguru import logger

from app.core.config import Settings

MCP_PROTOCOL_VERSION = "2025-03-26"


class WebSearchError(RuntimeError):
    pass


@dataclass
class WebSearchResult:
    hits: list[dict] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    elapsed_ms: int = 0


class WebSearchClient:
    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._http = http_client
        self._owns_http = http_client is None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._session_id: str | None = None
        self._protocol_version = MCP_PROTOCOL_VERSION

    @property
    def configured(self) -> bool:
        return bool(self._settings.web_search_enabled and self._settings.web_search_key)

    async def search(self, query: str, *, limit: int | None = None) -> WebSearchResult:
        started_at = time.perf_counter()
        if not self.configured:
            return WebSearchResult(status="disabled")
        if not query.strip():
            return WebSearchResult(status="empty")

        try:
            await self._ensure_initialized()
            pages = await self._call_tool(query.strip())
            hits = self._dedupe(pages, limit or self._settings.web_search_top_k)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("web search failed: {}", exc)
            return WebSearchResult(
                status="unavailable",
                error=_error_text(exc),
                elapsed_ms=_elapsed_ms(started_at),
            )

        return WebSearchResult(
            hits=hits,
            status="ok" if hits else "empty",
            elapsed_ms=_elapsed_ms(started_at),
        )

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            data, response = await self._post(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "travel-kb", "version": "0.1.0"},
                    },
                }
            )
            _raise_on_jsonrpc_error(data)
            result = data.get("result") or {}
            self._protocol_version = result.get("protocolVersion") or MCP_PROTOCOL_VERSION
            self._session_id = response.headers.get("mcp-session-id")

            try:
                await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
            except Exception:
                logger.debug("MCP initialized notification was not acknowledged")
            self._initialized = True

    async def _call_tool(self, query: str) -> list[dict]:
        data, _ = await self._post(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": self._settings.web_search_tool_name,
                    "arguments": {"query": query},
                },
            }
        )
        _raise_on_jsonrpc_error(data)
        result = data.get("result") or {}
        if result.get("isError"):
            raise WebSearchError(_content_text(result) or "search_pro returned an error")

        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return _pages_from_payload(structured)

        text = _content_text(result)
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WebSearchError("search_pro returned malformed JSON") from exc
        return _pages_from_payload(payload)

    async def _post(self, payload: dict) -> tuple[dict, httpx.Response]:
        response = await self.client.post(
            self._settings.web_search_mcp_url,
            headers=self._headers(),
            json=payload,
        )
        if response.status_code >= 400:
            raise WebSearchError(f"MCP HTTP {response.status_code}")
        return _parse_jsonrpc(response), response

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._settings.web_search_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._initialized:
            headers["MCP-Protocol-Version"] = self._protocol_version
        return headers

    @property
    def client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.web_search_timeout_seconds)
            )
        return self._http

    def _dedupe(self, pages: list[dict], limit: int) -> list[dict]:
        if limit <= 0:
            return []
        hits: list[dict] = []
        seen: set[str] = set()
        for page in pages:
            hit = _map_page(page)
            if hit is None:
                continue
            key = hit["url"] or hit["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            hits.append(hit)
            if len(hits) >= max(limit, 0):
                break
        return hits


def _parse_jsonrpc(response: httpx.Response) -> dict:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        data_lines = [
            line[5:].strip()
            for line in response.text.splitlines()
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
        if not data_lines:
            raise WebSearchError("empty MCP event stream")
        return json.loads(data_lines[-1])
    if not response.content:
        return {}
    return response.json()


def _raise_on_jsonrpc_error(data: dict) -> None:
    error = data.get("error")
    if not error:
        return
    message = error.get("message") if isinstance(error, dict) else str(error)
    raise WebSearchError(message or "MCP JSON-RPC error")


def _content_text(result: dict) -> str:
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _pages_from_payload(payload) -> list[dict]:
    if isinstance(payload, list):
        return [page for page in payload if isinstance(page, dict)]
    if isinstance(payload, dict):
        pages = payload.get("pages")
        if isinstance(pages, list):
            return [page for page in pages if isinstance(page, dict)]
    raise WebSearchError("unexpected search_pro payload")


def _map_page(page: dict) -> dict | None:
    url = str(page.get("url") or "").strip()
    title = str(page.get("title") or "").strip()
    snippet = str(page.get("snippet") or page.get("content") or "").strip()
    content = snippet or title
    if not content:
        return None

    site = str(page.get("hostname") or "").strip()
    if not site and url:
        site = urlparse(url).netloc

    source = url or f"{title}|{content}"
    chunk_id = "web:" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    published_at = page.get("published_at") or page.get("date") or page.get("time") or ""
    return {
        "chunk_id": chunk_id,
        "doc_id": "",
        "content": content,
        "score": 0.0,
        "source_type": "web",
        "title": title,
        "url": url,
        "site": site,
        "published_at": str(published_at),
        "retrieval_sources": ["web"],
        "content_type": "",
        "city": "",
        "scenic_name": "",
        "route_name": "",
        "hotel_name": "",
        "restaurant_name": "",
        "source_file": site or title,
        "source_path": url,
        "section_path": "",
    }


def _error_text(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message[:200]}" if message else type(exc).__name__


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)
