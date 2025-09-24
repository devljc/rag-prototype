# app/services/llm_stream.py
from __future__ import annotations

from typing import AsyncGenerator, List, Any, Optional, Dict

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from sse_starlette import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

from app.config import Config
from app.schemas import QueryBody
from app.services.core.llm_factory import get_model, LLMType
from app.services.core.stream_handler import CoalescingHandler
from app.services.rag import retrieve
from app.utils.cache import cache_key, get_cache, set_cache
from app.utils.helpers import list_sources, event, send, build_prompt


# =============================================================================
# Public entrypoint
# =============================================================================
async def stream(body: QueryBody) -> EventSourceResponse:
    """
    RAG → LLM 스트리밍(SSE)
    - 캐시 히트: 본문 한 번(data) + meta + end
    - 캐시 미스: 검색 → LLM 토큰 스트림(CoalescingHandler) → meta/end → 캐시
    """
    ckey = cache_key(body)
    cached = await get_cache(ckey)
    if cached:
        gen = _cached(cached)
    else:
        docs = _retrieve(body.question, body.top_k)

        model = get_model(stream=True, timeout=None)
        prompt = build_prompt(body.question, docs)
        handler = _handler(model, prompt)
        gen = _pipeline(handler, docs, cache_id=ckey)

    return EventSourceResponse(gen)


# =============================================================================
# Internals
# =============================================================================
def _retrieve(question: str, top_k: int) -> list[Document]:
    return retrieve(question, k=top_k)


def _handler(
        model: LLMType,
        prompt: list[SystemMessage | HumanMessage]
) -> CoalescingHandler:
    return CoalescingHandler(
        stream=model.astream(prompt),
        min_chars=Config.STREAM_MIN_CHARS,
        max_latency_ms=Config.STREAM_MAX_LAT_MS,
        heartbeat_secs=Config.HEARTBEAT_SECS,
    )


async def _pipeline(
        handler: CoalescingHandler,
        docs: List[Any],
        cache_id: Optional[str] = None,
) -> AsyncGenerator[ServerSentEvent, None]:
    async for frame in handler:
        yield frame

    sources = list_sources(docs)
    async for frame in _final(sources):
        yield frame

    if cache_id:
        await set_cache(cache_id, handler.get_answer(), sources)


def _cached(cached: Dict[str, Any]) -> AsyncGenerator[ServerSentEvent, None]:
    async def _gen():
        text = cached.get("answer") or ""
        sources = cached.get("sources")
        # 본문
        yield send({"answer": text})
        # meta/end
        async for frame in _final(sources):
            yield frame

    return _gen()


async def _final(sources: Optional[List[str]]) -> AsyncGenerator[Any, None]:
    if sources:
        yield event("meta", {"sources": sources})
    yield event("end", "[DONE]")
