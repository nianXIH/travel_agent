from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.exceptions import LLMNotConfiguredError


class ChatLLM:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: ChatOpenAI | None = None

    @property
    def configured(self) -> bool:
        return bool(self._settings.llm_api_key)

    @property
    def client(self) -> ChatOpenAI:
        if not self.configured:
            raise LLMNotConfiguredError()
        if self._client is None:
            extra_body = (
                None
                if self._settings.llm_enable_thinking
                else {"enable_thinking": False}
            )
            self._client = ChatOpenAI(
                model=self._settings.llm_model,
                openai_api_key=self._settings.llm_api_key,
                openai_api_base=self._settings.llm_base_url,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                request_timeout=self._settings.llm_timeout_seconds,
                stream_usage=True,
                extra_body=extra_body,
            )
        return self._client

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        model = self.client
        bindings = {}
        if temperature is not None:
            bindings["temperature"] = temperature
        if max_tokens is not None:
            bindings["max_tokens"] = max_tokens
        if bindings:
            model = model.bind(**bindings)
        return await model.ainvoke(messages)

    def astream(self, messages: list[BaseMessage]):
        return self.client.astream(messages)


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content or "")


def usage_dict(message) -> dict | None:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
