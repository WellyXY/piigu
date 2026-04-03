from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from api.auth import get_redis

router = APIRouter(prefix="/v1/admin")


@router.get("/billing")
async def billing_report(month: str | None = None):
    """
    Monthly billing report across all API keys.
    month format: YYYY-MM (defaults to current month)
    """
    r = await get_redis()
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")

    key_hashes = await r.smembers("apikeys")
    report = []

    for kh in sorted(key_hashes):
        data = await r.hgetall(f"apikey:{kh}")
        usage = await r.hgetall(f"apikey:{kh}:usage:{month}")
        count = int(usage.get("count", 0))

        report.append({
            "name": data.get("name", "?"),
            "key_hash_prefix": kh[:12],
            "month": month,
            "jobs_this_month": count,
            "total_jobs": int(data.get("total_jobs", 0)),
            "completed_jobs": int(data.get("completed_jobs", 0)),
            "failed_jobs": int(data.get("failed_jobs", 0)),
            "status": "disabled" if data.get("disabled") == "1" else "active",
        })

    return {
        "month": month,
        "total_clients": len(report),
        "total_jobs_this_month": sum(r["jobs_this_month"] for r in report),
        "clients": report,
    }


@router.post("/cleanup")
async def cleanup_expired_jobs(max_age_hours: int = 24):
    """Remove completed jobs older than max_age_hours from Redis."""
    r = await get_redis()
    import time

    cutoff = time.time() - (max_age_hours * 3600)
    cleaned = 0

    cursor = 0
    while True:
        cursor, keys = await r.scan(cursor, match="job:job_*", count=100)
        for key in keys:
            data = await r.hgetall(key)
            status = data.get("status", "")
            completed_at = float(data.get("completed_at", 0))

            if status in ("completed", "failed", "cancelled") and completed_at > 0 and completed_at < cutoff:
                await r.delete(key)
                cleaned += 1

                video_path = data.get("video_path", "")
                if video_path:
                    import os, shutil
                    job_dir = os.path.dirname(video_path)
                    if os.path.isdir(job_dir):
                        shutil.rmtree(job_dir, ignore_errors=True)

        if cursor == 0:
            break

    return {"cleaned_jobs": cleaned, "cutoff_hours": max_age_hours}
