from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.chat.llm import ChatLLM, message_text
from app.core.config import Settings
from app.core.prompts import HYDE_SYSTEM_PROMPT, build_hyde_prompt


class HyDEGenerator:
    def __init__(self, settings: Settings, llm: ChatLLM) -> None:
        self._settings = settings
        self._llm = llm

    async def generate(
        self,
        query: str,
        *,
        city: str | None = None,
        content_type: str | None = None,
    ) -> str:
        if not self._settings.hyde_enabled or not self._llm.configured:
            return ""
        try:
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=HYDE_SYSTEM_PROMPT),
                    HumanMessage(content=build_hyde_prompt(query, city, content_type)),
                ],
                temperature=self._settings.hyde_temperature,
                max_tokens=self._settings.hyde_max_tokens,
            )
        except Exception as exc:
            logger.warning("HyDE generation failed: {}", type(exc).__name__)
            return ""
        return _clean(message_text(response))


def _clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith(("text", "markdown")):
            _, _, remainder = text.partition("\n")
            text = remainder.strip()
    for prefix in ("假设性资料：", "资料：", "回答："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text[:1200]
