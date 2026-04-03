from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from api.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    r = await get_redis()
    key_hash = hash_key(api_key)
    key_data = await r.hgetall(f"apikey:{key_hash}")

    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if key_data.get("disabled") == "1":
        raise HTTPException(status_code=403, detail="API key disabled")

    return key_hash


async def create_api_key(name: str) -> str:
    r = await get_redis()
    raw_key = f"pk_{secrets.token_urlsafe(32)}"
    key_hash = hash_key(raw_key)

    await r.hset(f"apikey:{key_hash}", mapping={
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "disabled": "0",
        "total_jobs": "0",
        "completed_jobs": "0",
        "failed_jobs": "0",
    })

    await r.sadd("apikeys", key_hash)
    return raw_key


async def increment_usage(key_hash: str, field: str = "total_jobs"):
    r = await get_redis()
    await r.hincrby(f"apikey:{key_hash}", field, 1)
    month = datetime.utcnow().strftime("%Y-%m")
    await r.hincrby(f"apikey:{key_hash}:usage:{month}", "count", 1)
