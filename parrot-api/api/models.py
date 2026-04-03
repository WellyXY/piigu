from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


AVAILABLE_POSITIONS = [
    "blow_job", "cowgirl", "doggy", "handjob",
    "lift_clothes", "masturbation", "missionary", "reverse_cowgirl",
]


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Request models ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    position: str
    prompt: Optional[str] = ""
    duration: int = Field(default=10, ge=5, le=10)
    seed: Optional[int] = None
    callback_url: Optional[str] = None

    @model_validator(mode="after")
    def check_image_source(self):
        if not self.image_url and not self.image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")
        if self.position not in AVAILABLE_POSITIONS:
            raise ValueError(
                f"Invalid position '{self.position}'. "
                f"Available: {', '.join(AVAILABLE_POSITIONS)}"
            )
        return self


# ── Response models ─────────────────────────────────────────────

class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    position_in_queue: int
    estimated_wait_seconds: int


class JobMetadata(BaseModel):
    position: str
    duration: int
    resolution: str = "720p"
    fps: int = 30
    prompt: str = ""


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: JobMetadata


class PositionInfo(BaseModel):
    name: str
    description: str = ""


class PositionsResponse(BaseModel):
    positions: list[PositionInfo]


class UsageResponse(BaseModel):
    api_key: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    month: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class WebhookPayload(BaseModel):
    event: str
    job_id: str
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[JobMetadata] = None
