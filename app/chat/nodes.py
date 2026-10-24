import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.chat.llm import ChatLLM, message_text, usage_dict
from app.chat.memory import ChatMemory
from app.chat.state import ChatRuntime, ChatState
from app.core.config import Settings
from app.core.prompts import (
    QA_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    build_context,
    build_rewrite_prompt,
)
from app.retrieval.three_way import ThreeWayRetriever

KNOWN_CITIES = ("三亚", "云南", "厦门", "张家界", "成都", "杭州")
KNOWN_CONTENT_TYPES = ("景点介绍", "线路推荐", "交通指南", "美食介绍", "酒店信息")

CITY_ALIASES = {
    "昆明": "云南",
    "大理": "云南",
    "丽江": "云南",
    "香格里拉": "云南",
    "西双版纳": "云南",
    "玉龙雪山": "云南",
    "洱海": "云南",
    "乌镇": "杭州",
    "西湖": "杭州",
    "灵隐": "杭州",
    "千岛湖": "杭州",
    "鼓浪屿": "厦门",
    "曾厝垵": "厦门",
    "武陵源": "张家界",
    "天门山": "张家界",
    "都江堰": "成都",
    "青城山": "成都",
    "宽窄巷子": "成都",
    "亚龙湾": "三亚",
    "海棠湾": "三亚",
    "蜈支洲": "三亚",
}

CONTENT_TYPE_HINTS = (
    ("酒店信息", ("酒店", "民宿", "住宿", "客栈", "住哪")),
    ("美食介绍", ("美食", "餐厅", "小吃", "吃什么", "好吃的", "菜")),
    ("交通指南", ("交通", "机场", "高铁", "火车", "航班", "地铁", "怎么去", "怎么到")),
    ("线路推荐", ("线路", "行程", "路线", "几日游", "日游", "安排几天")),
    ("景点介绍", ("景点", "景区", "门票", "打卡", "好玩")),
)

DURATION_RE = re.compile(r"\d+\s*天|[两三四五六七]\s*天|几天|几日")

NO_CONTEXT_ANSWER = (
    "知识库和联网检索都没有找到足够的参考资料来回答这个问题。"
    "可以尝试补充城市、景点或具体需求，例如“杭州三天怎么安排”。"
)


class ChatNodes:
    def __init__(
        self,
        settings: Settings,
        memory: ChatMemory,
        three_way: ThreeWayRetriever,
        llm: ChatLLM,
    ) -> None:
        self._settings = settings
        self._memory = memory
        self._three_way = three_way
        self._llm = llm

    async def load_history(self, state: ChatState, config: RunnableConfig) -> dict:
        session_id = state["session_id"]
        session = await self._memory.get_session(session_id)
        history = await self._memory.load_history(
            session_id,
            self._settings.chat_history_max_messages,
            exclude_message_id=state.get("user_message_id"),
        )
        return {"history": history, "summary": session.get("summary", "")}

    async def rewrite_and_route(self, state: ChatState, config: RunnableConfig) -> dict:
        query = state["query"]
        history = state.get("history", [])
        rewritten, city, content_type, intent = await self._rewrite(
            query, history, state.get("summary", "")
        )

        explicit_city = state.get("city")
        explicit_type = state.get("content_type")
        city = explicit_city or city
        content_type = explicit_type or content_type

        runtime = _runtime(config)
        if runtime is not None:
            runtime.rewritten_query = rewritten
            runtime.intent = intent

        filters = {
            "city": city,
            "content_type": content_type,
        }
        return {
            "rewritten_query": rewritten,
            "filters": filters,
            "intent": intent,
        }

    async def retrieve(self, state: ChatState, config: RunnableConfig) -> dict:
        filters = state.get("filters") or {}

        async def emit_stage(stage: str) -> None:
            await _emit(config, "stage", {"stage": stage})

        outcome = await self._three_way.search(
            state.get("rewritten_query") or state["query"],
            top_k=state.get("top_k", self._settings.rerank_top_n),
            city=filters.get("city"),
            content_type=filters.get("content_type"),
            on_stage=emit_stage,
        )
        citations = [_citation(hit) for hit in outcome.citations]
        message_id = await self._memory.add_message(
            session_id=state["session_id"],
            role="assistant",
            content="",
            citations=citations,
            intent=state.get("intent", ""),
            status="streaming",
        )

        runtime = _runtime(config)
        if runtime is not None:
            runtime.assistant_message_id = message_id
            runtime.citations = citations

        await _emit(
            config,
            "meta",
            {
                "session_id": state["session_id"],
                "message_id": message_id,
                "rewritten_query": state.get("rewritten_query", state["query"]),
                "intent": state.get("intent", ""),
                "citations": citations,
                "retrieval": outcome.info,
            },
        )
        return {
            "citations": citations,
            "retrieval_info": outcome.info,
            "assistant_message_id": message_id,
        }

    async def generate(self, state: ChatState, config: RunnableConfig) -> dict:
        citations = state.get("citations", [])
        messages = _answer_messages(state)

        if not citations:
            runtime = _runtime(config)
            if runtime is not None:
                runtime.answer = NO_CONTEXT_ANSWER
            await _emit(config, "delta", {"content": NO_CONTEXT_ANSWER})
            return {"answer": NO_CONTEXT_ANSWER, "token_usage": None}

        await _emit(config, "stage", {"stage": "generating"})
        runtime = _runtime(config)

        if config.get("configurable", {}).get("stream"):
            parts: list[str] = []
            usage: dict | None = None
            async for chunk in self._llm.astream(messages):
                text = message_text(chunk)
                if text:
                    parts.append(text)
                    if runtime is not None:
                        runtime.answer = "".join(parts)
                    await _emit(config, "delta", {"content": text})
                usage = usage_dict(chunk) or usage
            answer = "".join(parts)
            if runtime is not None:
                runtime.answer = answer
            return {"answer": answer, "token_usage": usage}

        response = await self._llm.ainvoke(messages)
        answer = message_text(response)
        if runtime is not None:
            runtime.answer = answer
        return {"answer": answer, "token_usage": usage_dict(response)}

    async def persist(self, state: ChatState, config: RunnableConfig) -> dict:
        runtime = _runtime(config)
        message_id = state.get("assistant_message_id") or (
            runtime.assistant_message_id if runtime else None
        )
        answer = state.get("answer", "")
        if message_id:
            await self._memory.update_message(
                message_id,
                content=answer,
                token_usage=state.get("token_usage"),
                status="completed",
            )
        if runtime is not None:
            runtime.persisted = True

        started_at = config.get("configurable", {}).get("started_at")
        elapsed_ms = int((time.perf_counter() - started_at) * 1000) if started_at else 0
        await _emit(
            config,
            "done",
            {
                "finish_reason": "stop",
                "elapsed_ms": elapsed_ms,
                "usage": state.get("token_usage"),
            },
        )
        return {}

    async def _rewrite(
        self,
        query: str,
        history: list[dict],
        summary: str,
    ) -> tuple[str, str | None, str | None, str]:
        fallback = _heuristic_route(query)
        if not self._llm.configured:
            return query, *fallback

        try:
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                    HumanMessage(content=build_rewrite_prompt(query, history, summary)),
                ],
                temperature=0.0,
            )
            data = _parse_json_object(message_text(response))
        except Exception as exc:
            logger.warning("query rewrite failed, using heuristic route: {}", exc)
            return query, *fallback

        if data is None:
            return query, *fallback

        rewritten = data.get("rewritten_query")
        rewritten = rewritten.strip() if isinstance(rewritten, str) and rewritten.strip() else query
        city = _validate_choice(data.get("city"), KNOWN_CITIES) or fallback[0]

        raw_type = data.get("content_type")
        content_type = _validate_choice(raw_type, KNOWN_CONTENT_TYPES)
        if content_type is None and isinstance(raw_type, str) and raw_type.strip():
            content_type = fallback[1]

        intent = data.get("intent")
        intent = intent.strip() if isinstance(intent, str) and intent.strip() else fallback[2]
        return rewritten, city, content_type, intent


def _answer_messages(state: ChatState) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=QA_SYSTEM_PROMPT)]
    summary = state.get("summary")
    if summary:
        messages.append(SystemMessage(content=f"会话摘要：{summary}"))

    for item in state.get("history", []):
        content = item.get("content") or ""
        if not content or item.get("status") in {"failed", "streaming"}:
            continue
        if item.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif item.get("role") == "assistant":
            messages.append(AIMessage(content=content))

    query = state.get("rewritten_query") or state["query"]
    context = build_context(state.get("citations", []))
    messages.append(
        HumanMessage(content=f"参考资料：\n{context}\n\n用户问题：{query}")
    )
    return messages


def _citation(hit: dict) -> dict:
    return {
        "chunk_id": hit.get("chunk_id") or "",
        "doc_id": hit.get("doc_id") or "",
        "content": hit.get("content") or "",
        "score": float(hit.get("score") or 0.0),
        "rerank_score": hit.get("rerank_score"),
        "content_type": hit.get("content_type") or "",
        "city": hit.get("city") or "",
        "scenic_name": hit.get("scenic_name") or "",
        "route_name": hit.get("route_name") or "",
        "hotel_name": hit.get("hotel_name") or "",
        "restaurant_name": hit.get("restaurant_name") or "",
        "source_file": hit.get("source_file") or "",
        "source_path": hit.get("source_path") or "",
        "section_path": hit.get("section_path") or "",
        "source_type": hit.get("source_type") or "kb",
        "title": hit.get("title") or hit.get("source_file") or "",
        "url": hit.get("url") or "",
        "site": hit.get("site") or "",
        "published_at": hit.get("published_at") or "",
        "retrieval_sources": list(hit.get("retrieval_sources") or []),
        "fusion_score": hit.get("fusion_score"),
    }


def _heuristic_route(query: str) -> tuple[str | None, str | None, str]:
    city = next((name for name in KNOWN_CITIES if name in query), None)
    if city is None:
        city = next(
            (target for alias, target in CITY_ALIASES.items() if alias in query),
            None,
        )

    content_type = None
    for name, keywords in CONTENT_TYPE_HINTS:
        if any(keyword in query for keyword in keywords):
            content_type = name
            break
    if content_type is None and DURATION_RE.search(query):
        content_type = "线路推荐"

    intent = {
        "酒店信息": "住宿选择",
        "美食介绍": "美食推荐",
        "交通指南": "交通方式",
        "线路推荐": "线路规划",
        "景点介绍": "景点推荐",
    }.get(content_type, "综合问答")
    return city, content_type, intent


def _validate_choice(value: Any, allowed: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in allowed else None


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _runtime(config: RunnableConfig) -> ChatRuntime | None:
    return config.get("configurable", {}).get("runtime")


async def _emit(config: RunnableConfig, event: str, data: dict) -> None:
    queue = config.get("configurable", {}).get("event_queue")
    if queue is not None:
        await queue.put({"event": event, "data": data})
