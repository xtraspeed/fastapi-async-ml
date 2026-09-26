import hashlib
import json
from typing import Any, Optional
import redis

from app.config import settings
from app.core.logger import logger

_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    return _redis_client


def compute_cache_key(text: str) -> str:
    """Computes a deterministic SHA-256 hash for the given input payload."""
    clean_text = text.strip().lower()
    return f"cache:prediction:{hashlib.sha256(clean_text.encode('utf-8')).hexdigest()}"


def get_cached_prediction(cache_key: str) -> Optional[dict[str, Any]]:
    """Retrieves cached prediction result from Redis, or None if miss/error."""
    try:
        client = get_redis_client()
        cached = client.get(cache_key)
        if cached:
            logger.info(f"Cache HIT for key: {cache_key}")
            return json.loads(cached)
        return None
    except Exception as e:
        logger.warning(f"Redis get failed for {cache_key} (fallback to worker): {e}")
        return None


def set_cached_prediction(
    cache_key: str, data: dict[str, Any], ttl: int = settings.CACHE_TTL_SECONDS
) -> bool:
    """Stores prediction result in Redis with an expiration TTL."""
    try:
        client = get_redis_client()
        client.setex(cache_key, ttl, json.dumps(data))
        logger.info(f"Cached prediction for key: {cache_key} with TTL={ttl}s")
        return True
    except Exception as e:
        logger.warning(f"Redis set failed for {cache_key}: {e}")
        return False


def check_redis_health() -> bool:
    """Pings Redis to verify connectivity."""
    try:
        client = get_redis_client()
        return bool(client.ping())
    except Exception:
        return False
