from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends

from api.auth import get_redis, verify_api_key
from api.models import UsageResponse

router = APIRouter()


@router.get("/account/usage", response_model=UsageResponse)
async def get_usage(key_hash: str = Depends(verify_api_key)):
    r = await get_redis()
    data = await r.hgetall(f"apikey:{key_hash}")
    month = datetime.utcnow().strftime("%Y-%m")

    return UsageResponse(
        api_key=data.get("name", "unknown"),
        total_jobs=int(data.get("total_jobs", 0)),
        completed_jobs=int(data.get("completed_jobs", 0)),
        failed_jobs=int(data.get("failed_jobs", 0)),
        month=month,
    )
