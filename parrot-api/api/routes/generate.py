from __future__ import annotations

import base64
import hashlib
import os
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException

from api.auth import get_redis, increment_usage, verify_api_key
from api.config import settings
from api.models import ErrorResponse, GenerateRequest, GenerateResponse, JobStatus
from task_queue.job_manager import create_job, get_queue_length

router = APIRouter()

SECONDS_PER_VIDEO = 150


async def _save_input_image(req: GenerateRequest) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:16]}.jpg"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)

    if req.image_base64:
        data = base64.b64decode(req.image_base64)
        with open(filepath, "wb") as f:
            f.write(data)
    elif req.image_url:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(req.image_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {resp.status_code}")
            with open(filepath, "wb") as f:
                f.write(resp.content)

    return filepath


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={400: {"model": ErrorResponse}},
)
async def submit_generate(
    req: GenerateRequest,
    key_hash: str = Depends(verify_api_key),
):
    r = await get_redis()

    image_path = await _save_input_image(req)

    job = await create_job(
        r,
        position=req.position,
        prompt=req.prompt or "",
        duration=req.duration,
        seed=req.seed,
        image_path=image_path,
        callback_url=req.callback_url,
        api_key_hash=key_hash,
    )

    await increment_usage(key_hash, "total_jobs")

    queue_len = await get_queue_length(r)
    estimated_wait = queue_len * SECONDS_PER_VIDEO // settings.NUM_GPUS

    return GenerateResponse(
        job_id=job["job_id"],
        status=JobStatus.QUEUED,
        position_in_queue=queue_len,
        estimated_wait_seconds=estimated_wait,
    )
