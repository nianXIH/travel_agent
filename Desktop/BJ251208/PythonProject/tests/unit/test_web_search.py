import json

import httpx

from app.core.config import Settings
from app.retrieval.web_search import WebSearchClient


def _settings(**overrides) -> Settings:
    values = {
        "llm_api_key": "sk-test",
        "web_search_enabled": True,
        "web_search_top_k": 5,
        "web_search_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def _text_result(pages: list[dict]) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": json.dumps({"pages": pages}, ensure_ascii=False)}],
            "isError": False,
        },
    }


def _handler(pages: list[dict], calls: list[dict]):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "BaiLianMcpServer", "version": "1.0.0"},
                    },
                },
            )
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        if body["method"] == "tools/call":
            return httpx.Response(200, json=_text_result(pages))
        return httpx.Response(400, json={"error": "unexpected method"})

    return handle


def _client(handler, settings: Settings) -> tuple[WebSearchClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WebSearchClient(settings, http_client=http), http


async def test_search_maps_pages_to_web_hits():
    calls: list[dict] = []
    pages = [
        {
            "title": "2026 杭州西湖门票",
            "url": "https://example.com/xihu",
            "hostname": "example.com",
            "snippet": "西湖环湖区域全天开放，部分景点需预约。",
            "images": ["https://example.com/a.jpg"],
        }
    ]
    client, http = _client(_handler(pages, calls), _settings())
    try:
        result = await client.search("杭州西湖门票")
    finally:
        await http.aclose()

    assert result.status == "ok"
    assert result.error is None
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit["chunk_id"].startswith("web:")
    assert hit["source_type"] == "web"
    assert hit["title"] == "2026 杭州西湖门票"
    assert hit["url"] == "https://example.com/xihu"
    assert hit["site"] == "example.com"
    assert hit["content"] == "西湖环湖区域全天开放，部分景点需预约。"
    assert hit["retrieval_sources"] == ["web"]


async def test_search_uses_bailian_key_and_search_pro_tool():
    calls: list[dict] = []
    client, http = _client(_handler([], calls), _settings())
    try:
        await client.search("杭州天气")
    finally:
        await http.aclose()

    initialize = next(call for call in calls if call["method"] == "initialize")
    tool_call = next(call for call in calls if call["method"] == "tools/call")
    assert initialize["params"]["clientInfo"]["name"]
    assert tool_call["params"]["name"] == "search_pro"
    assert tool_call["params"]["arguments"] == {"query": "杭州天气"}


async def test_search_dedupes_by_url_and_respects_top_k():
    pages = [
        {"title": "A", "url": "https://a.example/1", "snippet": "a"},
        {"title": "A duplicate", "url": "https://a.example/1", "snippet": "a again"},
        {"title": "B", "url": "https://b.example/2", "snippet": "b"},
        {"title": "C", "url": "https://c.example/3", "snippet": "c"},
    ]
    client, http = _client(_handler(pages, []), _settings(web_search_top_k=2))
    try:
        result = await client.search("杭州")
    finally:
        await http.aclose()

    assert result.status == "ok"
    assert [hit["title"] for hit in result.hits] == ["A", "B"]


async def test_search_returns_unavailable_on_transport_error():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client, http = _client(fail, _settings())
    try:
        result = await client.search("杭州")
    finally:
        await http.aclose()

    assert result.status == "unavailable"
    assert result.hits == []
    assert "ConnectError" in result.error


async def test_search_returns_unavailable_on_malformed_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"protocolVersion": "2024-11-05"},
                },
            )
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"content": [{"type": "text", "text": "not json"}]},
            },
        )

    client, http = _client(handler, _settings())
    try:
        result = await client.search("杭州")
    finally:
        await http.aclose()

    assert result.status == "unavailable"
    assert result.hits == []


async def test_search_is_disabled_without_api_key():
    calls: list[dict] = []
    client, http = _client(_handler([], calls), _settings(llm_api_key=""))
    try:
        result = await client.search("杭州")
    finally:
        await http.aclose()

    assert result.status == "disabled"
    assert calls == []


async def test_search_is_disabled_by_flag():
    calls: list[dict] = []
    client, http = _client(_handler([], calls), _settings(web_search_enabled=False))
    try:
        result = await client.search("杭州")
    finally:
        await http.aclose()

    assert result.status == "disabled"
    assert calls == []


async def test_search_parses_sse_response():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            payload = {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"protocolVersion": "2024-11-05"},
            }
        elif body["method"] == "notifications/initialized":
            return httpx.Response(202)
        else:
            payload = _text_result(
                [
                    {
                        "title": "SSE 结果",
                        "url": "https://sse.example/1",
                        "hostname": "sse.example",
                        "snippet": "来自 SSE 的搜索摘要",
                    }
                ]
            )
        return httpx.Response(
            200,
            text=f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client, http = _client(handler, _settings())
    try:
        result = await client.search("杭州")
    finally:
        await http.aclose()

    assert result.status == "ok"
    assert result.hits[0]["title"] == "SSE 结果"


async def test_initialize_runs_once_for_multiple_searches():
    calls: list[dict] = []
    client, http = _client(_handler([], calls), _settings())
    try:
        await client.search("第一次")
        await client.search("第二次")
    finally:
        await http.aclose()

    assert sum(call["method"] == "initialize" for call in calls) == 1
