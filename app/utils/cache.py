# app/services/core/cache.py
from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

import redis.asyncio as redis

from app.config import Config
from app.deps.redis_client import get_redis
from app.schemas import QueryBody


# =========================
# High-level API
# =========================
def cache_key(body: QueryBody) -> str:
    user_tier = "admin" if body.admin else "user"
    return _key(body.question, body.top_k, user_tier)


async def get_cache(key: str) -> Optional[Dict[str, Any]]:
    r = await get_redis()
    await r.flushall()
    cached = await _get(r, key)
    return cached if isinstance(cached, dict) else None


async def set_cache(
        key: str,
        result: List[str] | str,
        sources: List[str],
) -> None:
    text = result.strip() if isinstance(result, str) else "".join(result).strip()
    if not text:
        return
    try:
        r = await get_redis()
        await _set(r, key, {"answer": text, "sources": sources}, ttl=Config.CACHE_TTL)
    except Exception:
        # 캐시 실패는 서비스 흐름을 막지 않음
        pass


async def invalidate(key: str) -> None:
    """해당 키 캐시 무효화."""
    try:
        r = await get_redis()
        await r.delete(key)
    except Exception:
        pass


# =========================
# Low-level Redis helpers
# =========================
def _key(question: str, top_k: int, role: str = "user") -> str:
    norm = " ".join((question or "").strip().split()).lower()
    return f"{Config.CACHE_PREFIX}{norm}|k={top_k}|role={role}"


async def _get(redis_: redis.Redis, key: str) -> Any:
    data = await redis_.get(key)
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None


async def _set(
        redis_: redis.Redis,
        key: str,
        value: Any,
        ttl: int | None = None,
) -> None:
    payload = json.dumps(value, ensure_ascii=False)
    await redis_.set(key, payload, ex=ttl or Config.CACHE_TTL)


__all__ = [
    "cache_key",
    "get_cache",
    "set_cache",
    "invalidate",
]
