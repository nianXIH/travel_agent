import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import suppress

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.chat.graph import build_chat_graph
from app.chat.llm import ChatLLM, message_text
from app.chat.memory import ChatMemory
from app.chat.nodes import ChatNodes
from app.chat.state import ChatRuntime, ChatState
from app.chat.streaming import sse_event
from app.core.config import Settings
from app.core.exceptions import AppError, LLMNotConfiguredError, UpstreamError
from app.core.prompts import SUMMARY_SYSTEM_PROMPT, build_summary_prompt
from app.retrieval.three_way import ThreeWayRetriever
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.document import Citation


class ChatService:
    def __init__(
        self,
        settings: Settings,
        memory: ChatMemory,
        three_way: ThreeWayRetriever,
        llm: ChatLLM,
    ) -> None:
        self._settings = settings
        self._memory = memory
        self._llm = llm
        self._graph = build_chat_graph(ChatNodes(settings, memory, three_way, llm))

    async def chat(self, request: ChatRequest, tenant_id: str) -> ChatResponse:
        self._ensure_configured()
        session_id, user_message_id = await self._prepare(request, tenant_id)
        started_at = time.perf_counter()
        runtime = ChatRuntime()

        try:
            result = await self._graph.ainvoke(
                self._initial_state(request, session_id, user_message_id, tenant_id),
                {
                    "configurable": {
                        "runtime": runtime,
                        "stream": False,
                        "started_at": started_at,
                    }
                },
            )
        except Exception as exc:
            await self._mark_incomplete(
                tenant_id, runtime, "partial" if runtime.answer else "failed"
            )
            if isinstance(exc, AppError):
                raise
            logger.exception("chat failed for session {}", session_id)
            raise UpstreamError("生成回答失败，请稍后重试") from exc

        answer = result.get("answer", "")
        citations = [Citation.model_validate(item) for item in result.get("citations", [])]
        self._schedule_summary(tenant_id, session_id)
        return ChatResponse(
            session_id=session_id,
            message_id=result.get("assistant_message_id", ""),
            answer=answer,
            rewritten_query=result.get("rewritten_query", request.query),
            intent=result.get("intent", ""),
            citations=citations,
            retrieval=result.get("retrieval_info"),
            usage=result.get("token_usage"),
            elapsed_ms=_elapsed_ms(started_at),
        )

    async def stream(self, request: ChatRequest, tenant_id: str) -> AsyncIterator:
        self._ensure_configured()
        session_id, user_message_id = await self._prepare(request, tenant_id)
        started_at = time.perf_counter()
        runtime = ChatRuntime()
        state = self._initial_state(request, session_id, user_message_id, tenant_id)
        queue: asyncio.Queue = asyncio.Queue()
        config = {
            "configurable": {
                "runtime": runtime,
                "stream": True,
                "event_queue": queue,
                "started_at": started_at,
            }
        }

        async def run_graph() -> None:
            try:
                await self._graph.ainvoke(state, config)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("chat stream failed for session {}", session_id)
                code, message = _error_payload(exc)
                await queue.put(
                    {"event": "error", "data": {"code": code, "message": message}}
                )
            finally:
                await queue.put(None)

        async def events() -> AsyncIterator:
            task = asyncio.create_task(run_graph())
            completed = False
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if item["event"] == "done":
                        completed = True
                    yield sse_event(item["event"], item["data"])
            finally:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                if not runtime.persisted:
                    await self._mark_incomplete(
                        tenant_id, runtime, "partial" if runtime.answer else "failed"
                    )
                if completed:
                    self._schedule_summary(tenant_id, session_id)

        return events()

    async def _prepare(self, request: ChatRequest, tenant_id: str) -> tuple[str, str]:
        session = await self._memory.get_or_create_session(
            tenant_id,
            session_id=request.session_id,
            user_id=request.user_id,
            title=request.query,
        )
        session_id = session["session_id"]
        user_message_id = await self._memory.add_message(
            tenant_id=tenant_id,
            session_id=session_id,
            role="user",
            content=request.query,
        )
        return session_id, user_message_id

    def _initial_state(
        self,
        request: ChatRequest,
        session_id: str,
        user_message_id: str,
        tenant_id: str,
    ) -> ChatState:
        return {
            "tenant_id": tenant_id,
            "query": request.query,
            "session_id": session_id,
            "user_id": request.user_id,
            "user_message_id": user_message_id,
            "top_k": request.top_k,
            "city": request.city,
            "content_type": request.content_type,
        }

    def _ensure_configured(self) -> None:
        if not self._llm.configured:
            raise LLMNotConfiguredError()

    async def _mark_incomplete(
        self, tenant_id: str, runtime: ChatRuntime, status: str
    ) -> None:
        if not runtime.assistant_message_id:
            return
        try:
            await self._memory.update_message(
                tenant_id,
                runtime.assistant_message_id,
                content=runtime.answer,
                status=status,
            )
        except Exception:
            logger.exception(
                "failed to mark message {} as {}",
                runtime.assistant_message_id,
                status,
            )

    def _schedule_summary(self, tenant_id: str, session_id: str) -> None:
        if not self._llm.configured:
            return
        task = asyncio.create_task(self._refresh_summary(tenant_id, session_id))
        task.add_done_callback(_log_background_failure)

    async def _refresh_summary(self, tenant_id: str, session_id: str) -> None:
        session = await self._memory.get_session(tenant_id, session_id)
        if session.get("message_count", 0) <= self._settings.chat_summary_trigger:
            return

        history = await self._memory.load_history(tenant_id, session_id, limit=40)
        if not history:
            return
        response = await self._llm.ainvoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=build_summary_prompt(history)),
            ],
            temperature=0.2,
        )
        summary = message_text(response).strip()
        if summary:
            await self._memory.refresh_summary(tenant_id, session_id, summary[:800])


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _error_payload(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, AppError):
        return exc.code, exc.message
    return "UPSTREAM_ERROR", "生成回答时发生错误，请稍后重试"


def _log_background_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("background task failed: {}", exc)
