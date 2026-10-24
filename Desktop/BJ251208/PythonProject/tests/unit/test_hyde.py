from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.retrieval.hyde import HyDEGenerator


class FakeLLM:
    def __init__(
        self,
        *,
        configured: bool = True,
        content: str = "",
        error: Exception | None = None,
    ):
        self.configured = configured
        self._content = content
        self._error = error
        self.calls: list[dict] = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self._error is not None:
            raise self._error
        return AIMessage(content=self._content)


def _settings(**overrides) -> Settings:
    values = {
        "hyde_enabled": True,
        "hyde_temperature": 0.3,
        "hyde_max_tokens": 320,
    }
    values.update(overrides)
    return Settings(**values)


async def test_generate_returns_hypothetical_passage():
    llm = FakeLLM(content="  杭州西湖位于杭州市区西部，环湖区域全天开放。  ")
    generator = HyDEGenerator(_settings(), llm)

    result = await generator.generate("西湖怎么玩")

    assert result == "杭州西湖位于杭州市区西部，环湖区域全天开放。"


async def test_generate_includes_filters_in_prompt():
    llm = FakeLLM(content="杭州西湖景区介绍")
    generator = HyDEGenerator(_settings(), llm)

    await generator.generate("西湖怎么玩", city="杭州", content_type="景点介绍")

    human_content = llm.calls[0]["messages"][1].content
    assert "杭州" in human_content
    assert "景点介绍" in human_content


async def test_generate_passes_temperature_and_max_tokens():
    llm = FakeLLM(content="杭州西湖")
    generator = HyDEGenerator(_settings(), llm)

    await generator.generate("西湖怎么玩")

    assert llm.calls[0]["temperature"] == 0.3
    assert llm.calls[0]["max_tokens"] == 320


async def test_generate_returns_empty_on_llm_failure():
    llm = FakeLLM(error=RuntimeError("upstream down"))
    generator = HyDEGenerator(_settings(), llm)

    assert await generator.generate("西湖怎么玩") == ""


async def test_generate_returns_empty_on_blank_response():
    llm = FakeLLM(content="   ")
    generator = HyDEGenerator(_settings(), llm)

    assert await generator.generate("西湖怎么玩") == ""


async def test_generate_is_disabled_by_config():
    llm = FakeLLM(content="杭州西湖")
    generator = HyDEGenerator(_settings(hyde_enabled=False), llm)

    assert await generator.generate("西湖怎么玩") == ""
    assert llm.calls == []


async def test_generate_is_skipped_without_llm_key():
    llm = FakeLLM(configured=False, content="杭州西湖")
    generator = HyDEGenerator(_settings(), llm)

    assert await generator.generate("西湖怎么玩") == ""
    assert llm.calls == []
