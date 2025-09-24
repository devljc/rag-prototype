import redis.asyncio as redis
from app.config import Config

_redis: redis.Redis | None = None

async def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(Config.REDIS_URL, decode_responses=True)
    return _redis