import asyncio

from app.chat.nodes import ChatNodes
from app.chat.state import ChatRuntime
from app.core.config import Settings
from app.retrieval.three_way import RetrievalOutcome


class FakeMemory:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def add_message(self, **kwargs):
        self.messages.append(kwargs)
        return "assistant-1"


class FakeThreeWay:
    def __init__(self, outcome: RetrievalOutcome) -> None:
        self._outcome = outcome
        self.calls: list[dict] = []

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        city,
        content_type,
        on_stage=None,
    ) -> RetrievalOutcome:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "city": city,
                "content_type": content_type,
            }
        )
        if on_stage is not None:
            await on_stage("retrieving")
            await on_stage("fusing")
        return self._outcome


class FakeLLM:
    configured = True


async def test_retrieve_uses_three_way_and_emits_retrieval_meta():
    outcome = RetrievalOutcome(
        citations=[
            {
                "chunk_id": "web:1",
                "content": "西湖环湖区域全天开放。",
                "score": 0.0,
                "source_type": "web",
                "title": "2026 杭州西湖门票",
                "url": "https://example.com/xihu",
                "site": "example.com",
                "retrieval_sources": ["web"],
            }
        ],
        info={"web": {"status": "ok", "hits": 1}, "degraded": []},
    )
    memory = FakeMemory()
    three_way = FakeThreeWay(outcome)
    queue: asyncio.Queue = asyncio.Queue()
    runtime = ChatRuntime()
    nodes = ChatNodes(Settings(), memory, three_way, FakeLLM())
    config = {
        "configurable": {
            "runtime": runtime,
            "event_queue": queue,
        }
    }

    result = await nodes.retrieve(
        {
            "query": "西湖怎么玩",
            "session_id": "s1",
            "rewritten_query": "杭州西湖怎么玩",
            "top_k": 5,
            "filters": {"city": "杭州", "content_type": None},
            "intent": "景点推荐",
        },
        config,
    )

    events = []
    while not queue.empty():
        events.append(await queue.get())

    assert three_way.calls == [
        {
            "query": "杭州西湖怎么玩",
            "top_k": 5,
            "city": "杭州",
            "content_type": None,
        }
    ]
    assert result["assistant_message_id"] == "assistant-1"
    assert result["retrieval_info"] == outcome.info
    assert result["citations"][0]["source_type"] == "web"
    assert result["citations"][0]["url"] == "https://example.com/xihu"
    assert runtime.assistant_message_id == "assistant-1"
    assert runtime.citations == result["citations"]
    assert [event["event"] for event in events] == ["stage", "stage", "meta"]
    assert events[-1]["data"]["retrieval"] == outcome.info
    assert memory.messages[0]["citations"] == result["citations"]
