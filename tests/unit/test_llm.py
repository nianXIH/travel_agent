from langchain_core.messages import AIMessage, HumanMessage

from app.chat.llm import ChatLLM
from app.core.config import Settings


class FakeClient:
    def __init__(self) -> None:
        self.bindings: list[dict] = []
        self.direct_calls = 0

    def bind(self, **kwargs):
        self.bindings.append(kwargs)
        return FakeBoundClient()

    async def ainvoke(self, messages):
        self.direct_calls += 1
        return AIMessage(content="direct")


class FakeBoundClient:
    async def ainvoke(self, messages):
        return AIMessage(content="bound")


async def test_ainvoke_binds_temperature_and_max_tokens():
    llm = ChatLLM(Settings(llm_api_key="sk-test"))
    client = FakeClient()
    llm._client = client

    response = await llm.ainvoke(
        [HumanMessage(content="hi")],
        temperature=0.2,
        max_tokens=123,
    )

    assert client.bindings == [{"temperature": 0.2, "max_tokens": 123}]
    assert response.content == "bound"


async def test_ainvoke_without_overrides_uses_client_directly():
    llm = ChatLLM(Settings(llm_api_key="sk-test"))
    client = FakeClient()
    llm._client = client

    response = await llm.ainvoke([HumanMessage(content="hi")])

    assert client.direct_calls == 1
    assert client.bindings == []
    assert response.content == "direct"
