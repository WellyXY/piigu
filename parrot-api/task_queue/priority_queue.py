"""
Priority queue implementation using Redis Sorted Sets.

Priority levels:
  0 = critical (admin / emergency)
  1 = high (premium clients)
  2 = normal (default)
  3 = low (free tier)

Within same priority, FIFO ordering is maintained via timestamp scoring:
  score = priority * 1e12 + timestamp_ms
"""
from __future__ import annotations

import time
from typing import Optional

import redis.asyncio as aioredis

PRIORITY_QUEUE_KEY = "job_priority_queue"


def _compute_score(priority: int = 2) -> float:
    return priority * 1e12 + time.time() * 1000


async def enqueue_priority(r: aioredis.Redis, job_id: str, priority: int = 2):
    score = _compute_score(priority)
    await r.zadd(PRIORITY_QUEUE_KEY, {job_id: score})


async def dequeue_priority(r: aioredis.Redis) -> Optional[str]:
    """Pop the highest-priority (lowest-score) job atomically."""
    result = await r.zpopmin(PRIORITY_QUEUE_KEY, count=1)
    if result:
        return result[0][0]  # (member, score) tuple
    return None


async def dequeue_priority_blocking(r: aioredis.Redis, timeout: int = 5) -> Optional[str]:
    """
    Blocking dequeue with polling fallback.
    Redis BZPOPMIN is available in redis>=6.2.
    """
    try:
        result = await r.bzpopmin(PRIORITY_QUEUE_KEY, timeout=timeout)
        if result:
            return result[1]  # (key, member, score)
    except Exception:
        result = await r.zpopmin(PRIORITY_QUEUE_KEY, count=1)
        if result:
            return result[0][0]
    return None


async def get_priority_queue_length(r: aioredis.Redis) -> int:
    return await r.zcard(PRIORITY_QUEUE_KEY)


async def get_priority_queue_position(r: aioredis.Redis, job_id: str) -> int:
    rank = await r.zrank(PRIORITY_QUEUE_KEY, job_id)
    if rank is not None:
        return rank + 1
    return 0


async def remove_from_priority_queue(r: aioredis.Redis, job_id: str):
    await r.zrem(PRIORITY_QUEUE_KEY, job_id)
