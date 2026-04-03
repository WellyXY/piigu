"""
Lightweight metrics collection without requiring prometheus_client library.
Exposes /metrics in Prometheus text format.
"""
from __future__ import annotations

import time

import redis.asyncio as aioredis
from fastapi import APIRouter

from api.config import settings
from task_queue.job_manager import QUEUE_KEY

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics():
    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

    try:
        queue_len = await r.llen(QUEUE_KEY)

        processing = 0
        completed = 0
        failed = 0

        key_hashes = await r.smembers("apikeys")
        total_keys = len(key_hashes)

        for kh in key_hashes:
            data = await r.hgetall(f"apikey:{kh}")
            completed += int(data.get("completed_jobs", 0))
            failed += int(data.get("failed_jobs", 0))

        lines = [
            "# HELP parrot_queue_length Number of jobs in queue",
            "# TYPE parrot_queue_length gauge",
            f"parrot_queue_length {queue_len}",
            "",
            "# HELP parrot_jobs_completed_total Total completed jobs",
            "# TYPE parrot_jobs_completed_total counter",
            f"parrot_jobs_completed_total {completed}",
            "",
            "# HELP parrot_jobs_failed_total Total failed jobs",
            "# TYPE parrot_jobs_failed_total counter",
            f"parrot_jobs_failed_total {failed}",
            "",
            "# HELP parrot_api_keys_total Total API keys",
            "# TYPE parrot_api_keys_total gauge",
            f"parrot_api_keys_total {total_keys}",
            "",
            "# HELP parrot_gpu_workers_configured Number of GPU workers configured",
            "# TYPE parrot_gpu_workers_configured gauge",
            f"parrot_gpu_workers_configured {settings.NUM_GPUS}",
            "",
        ]

        return "\n".join(lines) + "\n"

    finally:
        await r.aclose()


@router.get("/v1/admin/stats")
async def admin_stats():
    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

    try:
        queue_len = await r.llen(QUEUE_KEY)
        key_hashes = await r.smembers("apikeys")

        total_completed = 0
        total_failed = 0
        total_submitted = 0

        for kh in key_hashes:
            data = await r.hgetall(f"apikey:{kh}")
            total_submitted += int(data.get("total_jobs", 0))
            total_completed += int(data.get("completed_jobs", 0))
            total_failed += int(data.get("failed_jobs", 0))

        return {
            "queue_length": queue_len,
            "total_api_keys": len(key_hashes),
            "total_submitted": total_submitted,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "gpu_workers": settings.NUM_GPUS,
            "estimated_throughput_per_hour": settings.NUM_GPUS * 24,
        }

    finally:
        await r.aclose()
