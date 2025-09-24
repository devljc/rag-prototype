# app/services/llm_http.py
from typing import Dict, Any, List

from langchain_core.messages import BaseMessage

from app.config import Config
from app.schemas import QueryBody
from app.utils.cache import set_cache, cache_key, get_cache
from app.utils.helpers import list_sources, build_prompt
from app.services.core.llm_factory import get_model
from app.services.rag import retrieve


async def request(body: QueryBody) -> Dict[str, Any]:
    """
    단건(논-스트리밍) 처리:
      1) 캐시 히트 시 즉시 반환
      2) 미스면 RAG → LLM → 캐시 저장 후 반환
    """
    ckey = cache_key(body)

    cached = await get_cache(ckey)
    if cached:
        return cached

    docs = retrieve(body.question, k=body.top_k)
    result = await _request(body.question, docs)
    sources = list_sources(docs)
    await set_cache(ckey, result.content, sources)

    return {"answer": result.content, "sources": sources}


async def _request(question: str, docs: List[Any]) -> BaseMessage:
    model = get_model(stream=False, timeout=Config.REQUEST_TIMEOUT_S)
    prompt = build_prompt(question, docs)
    return await model.ainvoke(prompt)
