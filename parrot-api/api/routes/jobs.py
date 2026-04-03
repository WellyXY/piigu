from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.auth import get_redis, verify_api_key
from api.config import settings
from api.models import JobMetadata, JobResponse, JobStatus
from task_queue.job_manager import cancel_job, get_job, get_queue_position

router = APIRouter()


def _ts_to_iso(ts: float) -> Optional[str]:
    if not ts or ts == 0:
        return None
    return datetime.utcfromtimestamp(ts).isoformat() + "Z"


def _build_video_url(job_id: str) -> str:
    return f"{settings.VIDEO_BASE_URL}/{job_id}/video"


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    key_hash: str = Depends(verify_api_key),
):
    r = await get_redis()
    job = await get_job(r, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("api_key_hash") != key_hash:
        raise HTTPException(status_code=403, detail="Access denied")

    status = job["status"]
    video_url = None
    if status == JobStatus.COMPLETED.value and job.get("video_path"):
        video_url = _build_video_url(job_id)

    return JobResponse(
        job_id=job_id,
        status=status,
        progress=job.get("progress", 0),
        created_at=_ts_to_iso(job["created_at"]) or "",
        started_at=_ts_to_iso(job["started_at"]),
        completed_at=_ts_to_iso(job["completed_at"]),
        video_url=video_url,
        error=job.get("error") or None,
        metadata=JobMetadata(
            position=job["position"],
            duration=int(job.get("duration", 10)),
            prompt=job.get("prompt", ""),
        ),
    )


@router.get("/jobs/{job_id}/video")
async def download_video(
    job_id: str,
    key_hash: str = Depends(verify_api_key),
):
    r = await get_redis()
    job = await get_job(r, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("api_key_hash") != key_hash:
        raise HTTPException(status_code=403, detail="Access denied")
    if job["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Video not ready yet")

    video_path = job.get("video_path", "")
    if not video_path:
        raise HTTPException(status_code=404, detail="Video file not found")

    import os
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file missing from storage")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


@router.delete("/jobs/{job_id}")
async def cancel_job_endpoint(
    job_id: str,
    key_hash: str = Depends(verify_api_key),
):
    r = await get_redis()
    job = await get_job(r, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("api_key_hash") != key_hash:
        raise HTTPException(status_code=403, detail="Access denied")

    success = await cancel_job(r, job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job in current state")

    return {"job_id": job_id, "status": "cancelled"}
