# LoRA Training UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end LoRA training dashboard in the existing Railway admin UI that provisions on-demand RunPod pods, runs ltx-trainer, streams logs, stores checkpoints in R2, and lets the admin deploy the best checkpoint.

**Architecture:** A Railway FastAPI service (parrot-service) acts as orchestrator: it accepts video uploads, calls the RunPod REST API to create a GPU pod, then communicates with a lightweight FastAPI agent pre-installed on the pod. The agent runs ltx-trainer as a subprocess and streams logs back. Checkpoints are stored in Cloudflare R2.

**Tech Stack:** FastAPI, asyncpg (PostgreSQL), boto3 (R2), httpx (RunPod + agent calls), PyYAML (config generation), SSE (log streaming), Docker (pod image), ltx-trainer (training framework)

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `parrot-service/api/config.py` | Modify | Add RunPod API key, agent secret, GPU defaults |
| `parrot-service/api/models.py` | Modify | Add TrainingJob Pydantic models |
| `parrot-service/api/main.py` | Modify | Register training router, start orchestrator loop |
| `parrot-service/api/routes/training.py` | Create | Training CRUD + SSE log proxy + deploy endpoint |
| `parrot-service/db/training_store.py` | Create | training_jobs PostgreSQL CRUD |
| `parrot-service/db/job_store.py` | Modify | Add `CREATE TABLE training_jobs` to init |
| `parrot-service/training/__init__.py` | Create | Package marker |
| `parrot-service/training/runpod_client.py` | Create | RunPod REST API wrapper (create/get/terminate pod) |
| `parrot-service/training/config_generator.py` | Create | Generate ltx-trainer YAML + JSONL manifest |
| `parrot-service/training/orchestrator.py` | Create | Background asyncio loop: poll pod status, drive agent |
| `parrot-service/requirements.txt` | Modify | Add pyyaml, aiofiles |
| `parrot-service/frontend/admin.html` | Modify | Add Training tab (job list, new-job modal, detail view) |
| `parrot-service/pod_agent/main.py` | Create | FastAPI agent (runs ON pod at port 7860) |
| `parrot-service/pod_agent/trainer.py` | Create | ltx-trainer subprocess wrapper |
| `parrot-service/pod_agent/r2_sync.py` | Create | R2 upload/download helpers |
| `parrot-service/pod_agent/requirements.txt` | Create | Agent-only deps |
| `parrot-service/pod_agent/Dockerfile` | Create | Pod Docker image (ltx-trainer + agent) |
| `parrot-service/pod_agent/start.sh` | Create | Supervisord startup script |

---

## Task 1: PostgreSQL Schema — training_jobs Table

**Files:**
- Create: `parrot-service/db/training_store.py`
- Modify: `parrot-service/db/job_store.py`

- [ ] **Step 1: Add training_jobs DDL to job_store.py**

Open `parrot-service/db/job_store.py`. After the `ADD_RAW_KEY_SQL` constant (line ~57), add:

```python
CREATE_TRAINING_JOBS_SQL = """
CREATE TABLE IF NOT EXISTS training_jobs (
    id              TEXT PRIMARY KEY,
    position        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'provisioning',
    pod_id          TEXT,
    pod_ip          TEXT,
    r2_prefix       TEXT NOT NULL,
    config          JSONB NOT NULL DEFAULT '{}',
    current_step    INTEGER NOT NULL DEFAULT 0,
    total_steps     INTEGER NOT NULL DEFAULT 2000,
    error           TEXT,
    created_at      DOUBLE PRECISION NOT NULL,
    completed_at    DOUBLE PRECISION
);
"""
```

Then in the `get_pool()` function body, after the existing `await conn.execute(ADD_RAW_KEY_SQL)` call, add:

```python
            await conn.execute(CREATE_TRAINING_JOBS_SQL)
```

- [ ] **Step 2: Create `parrot-service/db/training_store.py`**

```python
"""PostgreSQL CRUD for LoRA training jobs."""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Optional

from db.job_store import get_pool

logger = logging.getLogger(__name__)


async def create_training_job(
    position: str,
    r2_prefix: str,
    config: dict,
) -> dict:
    pool = await get_pool()
    job_id = str(uuid.uuid4())
    now = time.time()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO training_jobs (id, position, status, r2_prefix, config, created_at)
            VALUES ($1, $2, 'provisioning', $3, $4, $5)
            """,
            job_id, position, r2_prefix, json.dumps(config), now,
        )
    return await get_training_job(job_id)


async def get_training_job(job_id: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM training_jobs WHERE id = $1", job_id
        )
    if not row:
        return None
    return dict(row)


async def list_training_jobs(limit: int = 50) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT $1", limit
        )
    return [dict(r) for r in rows]


async def update_training_job(job_id: str, **kwargs) -> None:
    """Update any subset of fields. Supported: status, pod_id, pod_ip,
    current_step, total_steps, error, completed_at."""
    if not kwargs:
        return
    pool = await get_pool()
    set_parts = []
    values = []
    for i, (k, v) in enumerate(kwargs.items(), start=1):
        set_parts.append(f"{k} = ${i}")
        values.append(v)
    values.append(job_id)
    sql = f"UPDATE training_jobs SET {', '.join(set_parts)} WHERE id = ${len(values)}"
    async with pool.acquire() as conn:
        await conn.execute(sql, *values)
```

- [ ] **Step 3: Verify the schema is created on startup**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python3 -c "
import asyncio
from db.job_store import get_pool
async def main():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(\"SELECT COUNT(*) FROM training_jobs\")
        print('training_jobs row count:', row[0])
asyncio.run(main())
"
```
Expected: `training_jobs row count: 0`
(Requires DATABASE_URL env var set. If not available locally, skip and verify after Railway deploy.)

- [ ] **Step 4: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/db/training_store.py parrot-service/db/job_store.py
git commit -m "feat: add training_jobs table and CRUD store"
```

---

## Task 2: Config — RunPod + Agent Settings

**Files:**
- Modify: `parrot-service/api/config.py`

- [ ] **Step 1: Add RunPod settings to config.py**

In `parrot-service/api/config.py`, add these fields to the `Settings` class after the `CREDITS_PER_SECOND` line:

```python
    # ── LoRA Training (RunPod) ──
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
    RUNPOD_GPU_TYPE_ID: str = os.getenv("RUNPOD_GPU_TYPE_ID", "NVIDIA H100 80GB HBM3")
    RUNPOD_IMAGE_NAME: str = os.getenv("RUNPOD_IMAGE_NAME", "racoonn/lora-trainer-agent:latest")
    RUNPOD_CONTAINER_DISK_GB: int = int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "100"))
    AGENT_PORT: int = int(os.getenv("AGENT_PORT", "7860"))
    AGENT_SECRET: str = os.getenv("AGENT_SECRET", "")
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/api/config.py
git commit -m "feat: add RunPod and agent settings to config"
```

---

## Task 3: RunPod REST API Client

**Files:**
- Create: `parrot-service/training/__init__.py`
- Create: `parrot-service/training/runpod_client.py`

- [ ] **Step 1: Create package marker**

```bash
touch "/Users/welly/Parrot API/parrot-service/training/__init__.py"
```

- [ ] **Step 2: Create `parrot-service/training/runpod_client.py`**

```python
"""
RunPod REST API v1 client.
Docs: https://rest.runpod.io/v1
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from api.config import settings

logger = logging.getLogger(__name__)

_BASE = "https://rest.runpod.io/v1"


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.RUNPOD_API_KEY}"}


async def create_pod(
    job_id: str,
    gpu_type_id: str,
    extra_env: Optional[dict] = None,
) -> dict:
    """
    Create an on-demand RunPod pod and return the full pod dict.
    The pod will run racoonn/lora-trainer-agent:latest and expose port 7860/http.
    extra_env: additional env vars merged with R2 creds and AGENT_SECRET.
    """
    env = {
        "AGENT_SECRET": settings.AGENT_SECRET,
        "R2_ACCOUNT_ID": settings.R2_ACCOUNT_ID,
        "R2_ACCESS_KEY_ID": settings.R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": settings.R2_SECRET_ACCESS_KEY,
        "R2_BUCKET_NAME": settings.R2_BUCKET_NAME,
        "R2_PUBLIC_URL": settings.R2_PUBLIC_URL,
        "JOB_ID": job_id,
    }
    if extra_env:
        env.update(extra_env)

    payload = {
        "name": f"lora-{job_id[:8]}",
        "imageName": settings.RUNPOD_IMAGE_NAME,
        "gpuTypeId": gpu_type_id,
        "containerDiskInGb": settings.RUNPOD_CONTAINER_DISK_GB,
        "ports": f"{settings.AGENT_PORT}/http",
        "env": env,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{_BASE}/pods", json=payload, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"[RunPod] Created pod: {data.get('id')}")
        return data


async def get_pod(pod_id: str) -> dict:
    """Return pod status dict. Key fields: id, status, runtime.ports"""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{_BASE}/pods/{pod_id}", headers=_headers())
        resp.raise_for_status()
        return resp.json()


async def terminate_pod(pod_id: str) -> None:
    """Terminate (delete) a pod immediately."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{_BASE}/pods/{pod_id}", headers=_headers())
        if resp.status_code not in (200, 204, 404):
            resp.raise_for_status()
        logger.info(f"[RunPod] Terminated pod: {pod_id}")


def get_pod_public_url(pod: dict) -> Optional[str]:
    """
    Extract the public HTTP URL for port 7860 from a pod status dict.
    RunPod public URL format: https://{pod_id}-{port}.proxy.runpod.net
    """
    pod_id = pod.get("id", "")
    # Check runtime.ports array
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    for p in ports:
        if p.get("privatePort") == settings.AGENT_PORT and p.get("type") == "http":
            public_port = p.get("publicPort")
            if public_port:
                return f"https://{pod_id}-{public_port}.proxy.runpod.net"
    # Fallback: construct standard URL
    if pod_id and pod.get("status") == "RUNNING":
        return f"https://{pod_id}-{settings.AGENT_PORT}.proxy.runpod.net"
    return None
```

- [ ] **Step 3: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/training/__init__.py parrot-service/training/runpod_client.py
git commit -m "feat: add RunPod REST API client"
```

---

## Task 4: Training Config Generator

**Files:**
- Create: `parrot-service/training/config_generator.py`
- Modify: `parrot-service/requirements.txt`

- [ ] **Step 1: Add pyyaml and aiofiles to requirements.txt**

In `parrot-service/requirements.txt`, add after the existing lines:

```
pyyaml>=6.0
aiofiles>=23.0
```

- [ ] **Step 2: Create `parrot-service/training/config_generator.py`**

```python
"""
Generates ltx-trainer YAML config and JSONL dataset manifest
from a TrainingJob config dict.
"""
from __future__ import annotations

import json
from typing import Optional

import yaml


LTX_MODEL_PATH = "/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors"
TEXT_ENCODER_PATH = "/workspace/gemma_configs"


def build_yaml_config(job_id: str, position: str, config: dict) -> str:
    """
    Build ltx-trainer YAML config string from job parameters.

    config keys (all optional, defaults shown):
        steps: 2000
        learning_rate: 0.0001
        rank: 32
        frames: 249
        validation_prompt: str  (auto-generated if missing)
    """
    steps = int(config.get("steps", 2000))
    lr = float(config.get("learning_rate", 1e-4))
    rank = int(config.get("rank", 32))
    frames = int(config.get("frames", 249))
    val_prompt = config.get(
        "validation_prompt",
        f"A person performing {position} motion. --{position}"
    )

    data = {
        "model": {
            "model_path": LTX_MODEL_PATH,
            "text_encoder_path": TEXT_ENCODER_PATH,
            "training_mode": "lora",
            "load_checkpoint": None,
        },
        "lora": {
            "rank": rank,
            "alpha": rank,
            "dropout": 0.0,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "training_strategy": {
            "name": "text_to_video",
            "first_frame_conditioning_p": 0.9,
            "with_audio": False,
        },
        "optimization": {
            "learning_rate": lr,
            "steps": steps,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw8bit",
            "scheduler_type": "linear",
            "enable_gradient_checkpointing": True,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            "quantization": "int8-quanto",
            "load_text_encoder_in_8bit": True,
        },
        "data": {
            "preprocessed_data_root": f"/workspace/training/{job_id}_preprocessed",
            "num_dataloader_workers": 2,
        },
        "validation": {
            "prompts": [val_prompt],
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "images": None,
            "video_dims": [512, 768, frames],
            "frame_rate": 25.0,
            "seed": 42,
            "inference_steps": 30,
            "interval": 500,
            "videos_per_prompt": 1,
            "guidance_scale": 4.0,
            "stg_scale": 1.0,
            "stg_blocks": [29],
            "stg_mode": "stg_v",
            "generate_audio": False,
            "skip_initial_validation": True,
        },
        "checkpoints": {
            "interval": 500,
            "keep_last_n": 4,
            "precision": "bfloat16",
        },
        "seed": 42,
        "output_dir": f"/workspace/training/{job_id}_output",
    }
    return yaml.dump(data, default_flow_style=False, allow_unicode=True)


def build_jsonl_manifest(videos: list[dict]) -> str:
    """
    Build JSONL dataset manifest.
    videos: list of {"path": "/workspace/training/{job_id}_videos/foo.mp4", "caption": "..."}
    Returns newline-separated JSON lines.
    """
    lines = []
    for v in videos:
        lines.append(json.dumps({"media_path": v["path"], "caption": v["caption"]}))
    return "\n".join(lines)
```

- [ ] **Step 3: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/training/config_generator.py parrot-service/requirements.txt
git commit -m "feat: add ltx-trainer config generator"
```

---

## Task 5: Training Pydantic Models

**Files:**
- Modify: `parrot-service/api/models.py`

- [ ] **Step 1: Add training models to models.py**

At the end of `parrot-service/api/models.py`, add:

```python
# ── Training ─────────────────────────────────────────────────────

class VideoCaption(BaseModel):
    filename: str
    caption: str


class CreateTrainingJobRequest(BaseModel):
    position: str
    videos: list[VideoCaption]   # filename → caption mapping
    steps: int = Field(default=2000, ge=100, le=5000)
    learning_rate: float = Field(default=1e-4, gt=0)
    rank: int = Field(default=32, ge=4, le=128)
    frames: int = Field(default=249, ge=25, le=301)
    gpu_type_id: Optional[str] = None   # overrides RUNPOD_GPU_TYPE_ID default
    validation_prompt: Optional[str] = None


class TrainingJobResponse(BaseModel):
    id: str
    position: str
    status: str
    pod_id: Optional[str] = None
    pod_ip: Optional[str] = None
    r2_prefix: str
    config: dict
    current_step: int
    total_steps: int
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


class TrainingCheckpoint(BaseModel):
    key: str        # R2 key e.g. "training/abc/checkpoints/lora_weights_step_02000.safetensors"
    name: str       # e.g. "lora_weights_step_02000.safetensors"
    size_mb: float
    step: int


class TrainingJobDetailResponse(TrainingJobResponse):
    checkpoints: list[TrainingCheckpoint] = []
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/api/models.py
git commit -m "feat: add training job Pydantic models"
```

---

## Task 6: Training API Routes

**Files:**
- Create: `parrot-service/api/routes/training.py`

- [ ] **Step 1: Create `parrot-service/api/routes/training.py`**

```python
"""
Training management routes — admin only.
Handles job creation, status, log streaming, checkpoint deploy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import boto3
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from api.auth import require_admin
from api.config import settings
from api.models import (
    CreateTrainingJobRequest,
    TrainingCheckpoint,
    TrainingJobDetailResponse,
    TrainingJobResponse,
    VideoCaption,
)
from db import training_store
from training import runpod_client
from training.config_generator import build_jsonl_manifest, build_yaml_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin/training", dependencies=[Depends(require_admin)])


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def _job_to_response(job: dict) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=job["id"],
        position=job["position"],
        status=job["status"],
        pod_id=job.get("pod_id"),
        pod_ip=job.get("pod_ip"),
        r2_prefix=job["r2_prefix"],
        config=job["config"] if isinstance(job["config"], dict) else json.loads(job["config"]),
        current_step=job["current_step"],
        total_steps=job["total_steps"],
        error=job.get("error"),
        created_at=float(job["created_at"]),
        completed_at=float(job["completed_at"]) if job.get("completed_at") else None,
    )


@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(req: CreateTrainingJobRequest):
    """
    Create a training job. Videos must already be uploaded to R2 at
    training/{job_id}/videos/{filename} before calling this endpoint.
    This endpoint provisions the RunPod pod and kicks off the orchestrator.
    """
    job_id = str(uuid.uuid4())
    r2_prefix = f"training/{job_id}"

    config = {
        "steps": req.steps,
        "learning_rate": req.learning_rate,
        "rank": req.rank,
        "frames": req.frames,
        "gpu_type_id": req.gpu_type_id or settings.RUNPOD_GPU_TYPE_ID,
        "validation_prompt": req.validation_prompt,
        "videos": [{"filename": v.filename, "caption": v.caption} for v in req.videos],
    }

    job = await training_store.create_training_job(
        position=req.position,
        r2_prefix=r2_prefix,
        config=config,
    )

    # Launch pod provisioning in background
    from training.orchestrator import start_job_orchestration
    asyncio.create_task(start_job_orchestration(job["id"]))

    return _job_to_response(job)


@router.post("/jobs/upload-video")
async def upload_training_video(
    job_id: str = Form(...),
    filename: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a training video to R2 before creating the job.
    Returns the R2 key for the uploaded video.
    """
    key = f"training/{job_id}/videos/{filename}"
    content = await file.read()
    client = _r2_client()
    client.put_object(
        Bucket=settings.R2_BUCKET_NAME,
        Key=key,
        Body=content,
        ContentType="video/mp4",
    )
    return {"key": key, "job_id": job_id, "filename": filename}


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs():
    jobs = await training_store.list_training_jobs()
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobDetailResponse)
async def get_training_job(job_id: str):
    job = await training_store.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    base = _job_to_response(job)
    checkpoints = _list_checkpoints(job["r2_prefix"])

    return TrainingJobDetailResponse(
        **base.model_dump(),
        checkpoints=checkpoints,
    )


def _list_checkpoints(r2_prefix: str) -> list[TrainingCheckpoint]:
    """List checkpoint .safetensors files in R2 under {r2_prefix}/checkpoints/"""
    try:
        client = _r2_client()
        prefix = f"{r2_prefix}/checkpoints/"
        resp = client.list_objects_v2(Bucket=settings.R2_BUCKET_NAME, Prefix=prefix)
        items = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            name = key.split("/")[-1]
            if not name.endswith(".safetensors"):
                continue
            # Parse step number from name like "lora_weights_step_02000.safetensors"
            step = 0
            try:
                step = int(name.replace("lora_weights_step_", "").replace(".safetensors", ""))
            except ValueError:
                pass
            items.append(TrainingCheckpoint(
                key=key,
                name=name,
                size_mb=round(obj["Size"] / 1024 / 1024, 1),
                step=step,
            ))
        return sorted(items, key=lambda x: x.step)
    except Exception as e:
        logger.warning(f"Failed to list checkpoints for {r2_prefix}: {e}")
        return []


@router.get("/jobs/{job_id}/logs")
async def stream_job_logs(job_id: str):
    """
    SSE stream of training logs. Proxies from pod agent if pod is running,
    or reads cached logs from R2 if job is done.
    """
    job = await training_store.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    async def generate():
        import httpx

        pod_ip = job.get("pod_ip")
        if not pod_ip:
            yield f"data: No pod assigned yet. Status: {job['status']}\n\n"
            return

        agent_url = f"{pod_ip}/logs"
        agent_secret = settings.AGENT_SECRET

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "GET", agent_url,
                    headers={"X-Agent-Secret": agent_secret},
                    timeout=300,
                ) as resp:
                    async for line in resp.aiter_lines():
                        yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: [Connection to pod lost: {e}]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/deploy/{checkpoint_name}")
async def deploy_checkpoint(job_id: str, checkpoint_name: str):
    """
    Copy a checkpoint from training R2 path to the production loras/ path.
    e.g. training/{job_id}/checkpoints/lora_weights_step_02000.safetensors
      → loras/{position}.safetensors
    """
    job = await training_store.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    src_key = f"{job['r2_prefix']}/checkpoints/{checkpoint_name}"
    dst_key = f"loras/{job['position']}.safetensors"

    client = _r2_client()
    try:
        client.copy_object(
            Bucket=settings.R2_BUCKET_NAME,
            CopySource={"Bucket": settings.R2_BUCKET_NAME, "Key": src_key},
            Key=dst_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"R2 copy failed: {e}")

    return {
        "ok": True,
        "deployed": dst_key,
        "position": job["position"],
        "checkpoint": checkpoint_name,
        "note": "Restart the inference server to load the new LoRA.",
    }


@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Terminate the RunPod pod and mark job as failed."""
    job = await training_store.get_training_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    pod_id = job.get("pod_id")
    if pod_id:
        try:
            await runpod_client.terminate_pod(pod_id)
        except Exception as e:
            logger.warning(f"Pod termination failed: {e}")

    await training_store.update_training_job(
        job_id,
        status="cancelled",
        completed_at=time.time(),
        error="Cancelled by admin",
    )
    return {"ok": True, "job_id": job_id}
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/api/routes/training.py
git commit -m "feat: add training API routes (CRUD, SSE logs, deploy)"
```

---

## Task 7: Pod Orchestrator (Background Loop)

**Files:**
- Create: `parrot-service/training/orchestrator.py`

- [ ] **Step 1: Create `parrot-service/training/orchestrator.py`**

```python
"""
Background orchestrator for a single training job lifecycle:
  provisioning → preprocessing → training → uploading → done (or failed)

Called as an asyncio task per job. Communicates with the pod agent via HTTP.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx

from api.config import settings
from db import training_store
from training import runpod_client
from training.config_generator import build_jsonl_manifest, build_yaml_config

logger = logging.getLogger(__name__)

POLL_INTERVAL = 15       # seconds between RunPod status polls
POD_READY_TIMEOUT = 600  # 10 minutes max wait for pod to be RUNNING
AGENT_READY_TIMEOUT = 120  # 2 minutes for agent to respond after pod is RUNNING


def _agent_headers() -> dict:
    return {"X-Agent-Secret": settings.AGENT_SECRET, "Content-Type": "application/json"}


async def _wait_for_pod_running(pod_id: str) -> dict:
    """Poll RunPod until pod status is RUNNING. Returns pod dict with public URL."""
    deadline = time.time() + POD_READY_TIMEOUT
    while time.time() < deadline:
        pod = await runpod_client.get_pod(pod_id)
        status = pod.get("status", "")
        logger.info(f"[Orch] Pod {pod_id} status: {status}")
        if status == "RUNNING":
            return pod
        if status in ("EXITED", "FAILED", "TERMINATED"):
            raise RuntimeError(f"Pod {pod_id} entered terminal state: {status}")
        await asyncio.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Pod {pod_id} did not become RUNNING within {POD_READY_TIMEOUT}s")


async def _wait_for_agent(agent_url: str) -> None:
    """Poll agent /status until it responds."""
    deadline = time.time() + AGENT_READY_TIMEOUT
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{agent_url}/status",
                    headers=_agent_headers(),
                )
                if resp.status_code == 200:
                    logger.info(f"[Orch] Agent is ready at {agent_url}")
                    return
        except Exception:
            pass
        await asyncio.sleep(10)
    raise TimeoutError(f"Agent at {agent_url} did not respond within {AGENT_READY_TIMEOUT}s")


async def _call_agent(agent_url: str, method: str, path: str, body: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        fn = client.post if method == "POST" else client.get
        kwargs = {"headers": _agent_headers()}
        if body:
            kwargs["json"] = body
        resp = await fn(f"{agent_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()


async def _poll_training_progress(agent_url: str, job_id: str) -> None:
    """
    Poll agent /status every 30s and update PG step count.
    Returns when training is complete (phase == 'done') or raises on failure.
    """
    while True:
        try:
            data = await _call_agent(agent_url, "GET", "/status")
            phase = data.get("phase", "")
            step = int(data.get("current_step", 0))
            total = int(data.get("total_steps", 2000))

            await training_store.update_training_job(job_id, current_step=step, total_steps=total)
            logger.info(f"[Orch] Job {job_id}: {phase} step={step}/{total}")

            if phase == "done":
                return
            if phase == "failed":
                raise RuntimeError(data.get("error", "Training failed"))
        except httpx.HTTPError as e:
            logger.warning(f"[Orch] Agent poll error: {e}")

        await asyncio.sleep(30)


async def start_job_orchestration(job_id: str) -> None:
    """
    Full lifecycle for one training job. Run as asyncio.create_task().
    """
    job = await training_store.get_training_job(job_id)
    if not job:
        logger.error(f"[Orch] Job {job_id} not found")
        return

    config = job["config"] if isinstance(job["config"], dict) else json.loads(job["config"])
    position = job["position"]
    r2_prefix = job["r2_prefix"]
    gpu_type = config.get("gpu_type_id", settings.RUNPOD_GPU_TYPE_ID)

    try:
        # ── 1. Create pod ──────────────────────────────────────────────
        logger.info(f"[Orch] Provisioning pod for job {job_id}")
        pod = await runpod_client.create_pod(job_id, gpu_type)
        pod_id = pod["id"]
        await training_store.update_training_job(job_id, pod_id=pod_id, status="provisioning")

        # ── 2. Wait for pod RUNNING ────────────────────────────────────
        pod = await _wait_for_pod_running(pod_id)
        agent_url = runpod_client.get_pod_public_url(pod)
        if not agent_url:
            raise RuntimeError("Could not determine pod public URL")

        await training_store.update_training_job(job_id, pod_ip=agent_url, status="provisioning")

        # ── 3. Wait for agent to be ready ──────────────────────────────
        await _wait_for_agent(agent_url)

        # ── 4. Send setup payload to agent ────────────────────────────
        await training_store.update_training_job(job_id, status="preprocessing")
        videos = config.get("videos", [])
        video_r2_keys = [f"{r2_prefix}/videos/{v['filename']}" for v in videos]
        captions = {v["filename"]: v["caption"] for v in videos}

        yaml_config = build_yaml_config(job_id, position, config)
        jsonl_manifest = build_jsonl_manifest([
            {
                "path": f"/workspace/training/{job_id}_videos/{v['filename']}",
                "caption": v["caption"],
            }
            for v in videos
        ])

        await _call_agent(agent_url, "POST", "/setup", {
            "job_id": job_id,
            "r2_video_keys": video_r2_keys,
            "yaml_config": yaml_config,
            "jsonl_manifest": jsonl_manifest,
            "frames": config.get("frames", 249),
        })

        # ── 5. Start training ─────────────────────────────────────────
        await _call_agent(agent_url, "POST", "/train", {})
        await training_store.update_training_job(job_id, status="training")

        # ── 6. Poll progress ──────────────────────────────────────────
        await _poll_training_progress(agent_url, job_id)

        # ── 7. Upload checkpoints to R2 ───────────────────────────────
        await training_store.update_training_job(job_id, status="uploading")
        await _call_agent(agent_url, "POST", "/upload-checkpoints", {
            "r2_prefix": r2_prefix,
        })

        # ── 8. Mark done and terminate pod ────────────────────────────
        await training_store.update_training_job(
            job_id,
            status="done",
            completed_at=time.time(),
        )
        logger.info(f"[Orch] Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"[Orch] Job {job_id} failed: {e}")
        await training_store.update_training_job(
            job_id,
            status="failed",
            error=str(e),
            completed_at=time.time(),
        )
    finally:
        # Always try to terminate the pod
        job = await training_store.get_training_job(job_id)
        if job and job.get("pod_id"):
            try:
                await runpod_client.terminate_pod(job["pod_id"])
            except Exception as e:
                logger.warning(f"[Orch] Pod termination in finally: {e}")
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/training/orchestrator.py
git commit -m "feat: add pod orchestrator background loop"
```

---

## Task 8: Wire Training Router into main.py

**Files:**
- Modify: `parrot-service/api/main.py`
- Modify: `parrot-service/api/routes/__init__.py`

- [ ] **Step 1: Import and register training router in main.py**

In `parrot-service/api/main.py`, add the training import to the existing routes import line:

```python
from api.routes import account, admin, generate, jobs, positions, training
```

Then after `app.include_router(metrics_router, tags=["metrics"])`, add:

```python
app.include_router(training.router, tags=["training"])
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/api/main.py
git commit -m "feat: register training router in FastAPI app"
```

---

## Task 9: Pod Agent — FastAPI App

**Files:**
- Create: `parrot-service/pod_agent/main.py`

- [ ] **Step 1: Create `parrot-service/pod_agent/main.py`**

```python
"""
LoRA Training Pod Agent — runs on RunPod pod at port 7860.
Receives commands from the Railway orchestrator and drives ltx-trainer.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from trainer import TrainerState, run_preprocessing, run_training
from r2_sync import download_files, upload_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_SECRET = os.getenv("AGENT_SECRET", "")


def _auth(x_agent_secret: str = Header(default="")):
    if AGENT_SECRET and x_agent_secret != AGENT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid agent secret")


state = TrainerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Agent] Pod agent started")
    yield
    logger.info("[Agent] Pod agent shutting down")


app = FastAPI(lifespan=lifespan)


class SetupRequest(BaseModel):
    job_id: str
    r2_video_keys: list[str]
    yaml_config: str
    jsonl_manifest: str
    frames: int = 249


class UploadCheckpointsRequest(BaseModel):
    r2_prefix: str


@app.get("/status")
async def get_status(x_agent_secret: str = Header(default="")):
    _auth(x_agent_secret)
    return {
        "phase": state.phase,
        "current_step": state.current_step,
        "total_steps": state.total_steps,
        "running": state.running,
        "error": state.error,
    }


@app.post("/setup")
async def setup(req: SetupRequest, x_agent_secret: str = Header(default="")):
    _auth(x_agent_secret)
    if state.running:
        raise HTTPException(status_code=409, detail="Training already in progress")

    state.job_id = req.job_id
    state.phase = "downloading"

    # Download videos from R2
    video_dir = f"/workspace/training/{req.job_id}_videos"
    os.makedirs(video_dir, exist_ok=True)
    await asyncio.get_event_loop().run_in_executor(
        None, download_files, req.r2_video_keys, video_dir
    )

    # Write config files
    yaml_path = f"/workspace/training/{req.job_id}.yaml"
    jsonl_path = f"/workspace/training/{req.job_id}.jsonl"
    with open(yaml_path, "w") as f:
        f.write(req.yaml_config)
    with open(jsonl_path, "w") as f:
        f.write(req.jsonl_manifest)

    state.yaml_path = yaml_path
    state.jsonl_path = jsonl_path
    state.frames = req.frames
    state.phase = "ready"

    # Run preprocessing synchronously (blocks here until done)
    preprocessed_dir = f"/workspace/training/{req.job_id}_preprocessed"
    await asyncio.get_event_loop().run_in_executor(
        None, run_preprocessing, jsonl_path, preprocessed_dir, req.frames, state
    )

    return {"ok": True, "phase": state.phase}


@app.post("/train")
async def start_training(x_agent_secret: str = Header(default="")):
    _auth(x_agent_secret)
    if state.running:
        raise HTTPException(status_code=409, detail="Training already in progress")
    if state.phase not in ("ready", "preprocessing_done"):
        raise HTTPException(status_code=400, detail=f"Not ready to train, phase={state.phase}")

    asyncio.create_task(_run_training_task())
    return {"ok": True, "message": "Training started"}


async def _run_training_task():
    await asyncio.get_event_loop().run_in_executor(
        None, run_training, state.yaml_path, state
    )


@app.get("/logs")
async def stream_logs(x_agent_secret: str = Header(default="")):
    _auth(x_agent_secret)

    log_path = f"/workspace/logs/training_{state.job_id}.log"

    async def generate():
        last_size = 0
        while state.running or state.phase not in ("done", "failed"):
            try:
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        f.seek(last_size)
                        new_content = f.read()
                        if new_content:
                            for line in new_content.splitlines():
                                yield f"{line}\n"
                            last_size = f.tell()
            except Exception:
                pass
            await asyncio.sleep(2)

        # Flush remaining
        if os.path.exists(log_path):
            with open(log_path) as f:
                f.seek(last_size)
                for line in f.read().splitlines():
                    yield f"{line}\n"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/checkpoints")
async def list_checkpoints(x_agent_secret: str = Header(default="")):
    _auth(x_agent_secret)
    ckpt_dir = f"/workspace/training/{state.job_id}_output/checkpoints"
    files = []
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            if name.endswith(".safetensors"):
                path = os.path.join(ckpt_dir, name)
                files.append({
                    "name": name,
                    "size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
                })
    return {"checkpoints": files}


@app.post("/upload-checkpoints")
async def upload_checkpoints(req: UploadCheckpointsRequest, x_agent_secret: str = Header(default="")):
    _auth(x_agent_secret)
    ckpt_dir = f"/workspace/training/{state.job_id}_output/checkpoints"
    r2_prefix = f"{req.r2_prefix}/checkpoints"
    await asyncio.get_event_loop().run_in_executor(
        None, upload_directory, ckpt_dir, r2_prefix
    )
    return {"ok": True, "uploaded_to": r2_prefix}
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/pod_agent/main.py
git commit -m "feat: add pod agent FastAPI app"
```

---

## Task 10: Pod Agent — Trainer Subprocess

**Files:**
- Create: `parrot-service/pod_agent/trainer.py`

- [ ] **Step 1: Create `parrot-service/pod_agent/trainer.py`**

```python
"""
Wraps ltx-trainer subprocess calls (process_dataset.py and train.py).
Uses a shared TrainerState object to track progress.
"""
from __future__ import annotations

import os
import re
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

LTX_TRAINER_DIR = os.getenv("LTX_TRAINER_DIR", "/workspace/ltx-trainer")
LOG_DIR = "/workspace/logs"


@dataclass
class TrainerState:
    job_id: str = ""
    phase: str = "idle"          # idle | downloading | preprocessing | training | uploading | done | failed
    current_step: int = 0
    total_steps: int = 2000
    running: bool = False
    error: Optional[str] = None
    yaml_path: Optional[str] = None
    jsonl_path: Optional[str] = None
    frames: int = 249


def run_preprocessing(jsonl_path: str, preprocessed_dir: str, frames: int, state: TrainerState) -> None:
    """
    Run ltx-trainer's process_dataset.py to preprocess latents + captions.
    Blocks until complete.
    """
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/training_{state.job_id}.log"

    state.phase = "preprocessing"
    state.running = True

    cmd = [
        "python3",
        os.path.join(LTX_TRAINER_DIR, "scripts", "process_dataset.py"),
        jsonl_path,
        "--resolution-buckets", f"512x768x{frames}",
        "--output-dir", preprocessed_dir,
    ]
    logger.info(f"[Trainer] Preprocessing: {' '.join(cmd)}")

    with open(log_path, "a") as log_file:
        log_file.write(f"=== Preprocessing ===\n")
        result = subprocess.run(
            cmd,
            cwd=LTX_TRAINER_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        state.phase = "failed"
        state.running = False
        state.error = f"Preprocessing failed with exit code {result.returncode}"
        raise RuntimeError(state.error)

    state.phase = "preprocessing_done"


def run_training(yaml_path: str, state: TrainerState) -> None:
    """
    Run ltx-trainer's train.py with the given YAML config.
    Parses stdout for step progress. Blocks until complete.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/training_{state.job_id}.log"

    state.phase = "training"
    state.running = True

    cmd = [
        "python3",
        os.path.join(LTX_TRAINER_DIR, "train.py"),
        yaml_path,
    ]
    logger.info(f"[Trainer] Training: {' '.join(cmd)}")

    step_re = re.compile(r"Step[:\s]+(\d+)/(\d+)")

    with open(log_path, "a") as log_file:
        log_file.write(f"=== Training ===\n")
        proc = subprocess.Popen(
            cmd,
            cwd=LTX_TRAINER_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            m = step_re.search(line)
            if m:
                state.current_step = int(m.group(1))
                state.total_steps = int(m.group(2))

        proc.wait()

    if proc.returncode != 0:
        state.phase = "failed"
        state.running = False
        state.error = f"Training failed with exit code {proc.returncode}"
        raise RuntimeError(state.error)

    state.phase = "done"
    state.running = False
    logger.info(f"[Trainer] Training complete for job {state.job_id}")
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/pod_agent/trainer.py
git commit -m "feat: add trainer subprocess wrapper"
```

---

## Task 11: Pod Agent — R2 Sync

**Files:**
- Create: `parrot-service/pod_agent/r2_sync.py`

- [ ] **Step 1: Create `parrot-service/pod_agent/r2_sync.py`**

```python
"""
R2 upload/download helpers for the pod agent.
Uses boto3 with env vars R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


def _client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def download_files(r2_keys: list[str], local_dir: str) -> None:
    """Download a list of R2 keys into local_dir, preserving filename."""
    client = _client()
    bucket = os.environ["R2_BUCKET_NAME"]
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for key in r2_keys:
        filename = key.split("/")[-1]
        dest = os.path.join(local_dir, filename)
        logger.info(f"[R2] Downloading {key} → {dest}")
        client.download_file(bucket, key, dest)


def upload_directory(local_dir: str, r2_prefix: str) -> None:
    """Upload all files in local_dir to R2 under r2_prefix/filename."""
    client = _client()
    bucket = os.environ["R2_BUCKET_NAME"]
    for name in os.listdir(local_dir):
        local_path = os.path.join(local_dir, name)
        if not os.path.isfile(local_path):
            continue
        key = f"{r2_prefix}/{name}"
        logger.info(f"[R2] Uploading {local_path} → {key}")
        client.upload_file(local_path, bucket, key)
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/pod_agent/r2_sync.py
git commit -m "feat: add R2 sync helpers for pod agent"
```

---

## Task 12: Pod Agent — Dockerfile + Requirements

**Files:**
- Create: `parrot-service/pod_agent/requirements.txt`
- Create: `parrot-service/pod_agent/Dockerfile`
- Create: `parrot-service/pod_agent/start.sh`

- [ ] **Step 1: Create `parrot-service/pod_agent/requirements.txt`**

```
fastapi>=0.115
uvicorn[standard]>=0.34
pydantic>=2.0
boto3>=1.35
httpx>=0.27
```

- [ ] **Step 2: Create `parrot-service/pod_agent/start.sh`**

```bash
#!/bin/bash
set -e

mkdir -p /workspace/logs /workspace/training

echo "[start.sh] Starting pod agent on port 7860"
cd /agent
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
```

- [ ] **Step 3: Create `parrot-service/pod_agent/Dockerfile`**

```dockerfile
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Install ltx-trainer
RUN git clone https://github.com/Lightricks/LTX-Video-Trainer.git /workspace/ltx-trainer && \
    cd /workspace/ltx-trainer && \
    pip install --no-cache-dir -e .

# Install pod agent
COPY requirements.txt /agent/requirements.txt
RUN pip install --no-cache-dir -r /agent/requirements.txt

COPY main.py /agent/main.py
COPY trainer.py /agent/trainer.py
COPY r2_sync.py /agent/r2_sync.py

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 7860
CMD ["/start.sh"]
```

- [ ] **Step 4: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/pod_agent/
git commit -m "feat: add pod agent Dockerfile and startup scripts"
```

---

## Task 13: Admin UI — Training Tab

**Files:**
- Modify: `parrot-service/frontend/admin.html`

The `admin.html` file is a standalone HTML+JS file (~27KB). The Training tab needs to be added following the existing tab pattern (API Keys, Jobs, Billing). The changes are:

- [ ] **Step 1: Add "Training" tab button**

In `admin.html`, find the tab buttons section (look for the `<button onclick="showTab('keys')"` pattern) and add after the Billing tab button:

```html
<button onclick="showTab('training')" id="tab-training" class="tab-btn">Training</button>
```

- [ ] **Step 2: Add Training tab panel**

After the closing `</div>` of the billing tab panel, add the training tab panel. Find the billing panel (id="panel-billing") closing tag and add after it:

```html
<div id="panel-training" class="panel hidden">
  <div class="panel-header">
    <h2>LoRA Training Jobs</h2>
    <button class="btn btn-primary" onclick="showNewJobModal()">+ New Training Job</button>
  </div>
  <table id="training-table">
    <thead>
      <tr>
        <th>Position</th>
        <th>Status</th>
        <th>Progress</th>
        <th>GPU</th>
        <th>Created</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody id="training-body"></tbody>
  </table>
</div>

<!-- New Training Job Modal -->
<div id="new-job-modal" class="modal hidden">
  <div class="modal-box" style="max-width:600px">
    <h3>New LoRA Training Job</h3>
    <div class="form-group">
      <label>Position Name</label>
      <input id="tj-position" type="text" placeholder="e.g. cowgirl" />
    </div>
    <div class="form-group">
      <label>Training Videos (MP4)</label>
      <div id="video-drop-zone" ondrop="handleVideoDrop(event)" ondragover="event.preventDefault()"
           style="border:2px dashed #555;padding:20px;text-align:center;border-radius:8px;cursor:pointer">
        Drop MP4 files here or <label style="color:#7c3aed;cursor:pointer">
          browse<input type="file" id="video-file-input" multiple accept=".mp4" style="display:none"
                       onchange="handleVideoFiles(this.files)">
        </label>
      </div>
      <div id="video-list" style="margin-top:12px"></div>
    </div>
    <details style="margin-top:12px">
      <summary style="cursor:pointer;color:#aaa">Advanced Parameters</summary>
      <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div class="form-group">
          <label>Steps</label>
          <input id="tj-steps" type="number" value="2000" min="100" max="5000" />
        </div>
        <div class="form-group">
          <label>Learning Rate</label>
          <input id="tj-lr" type="text" value="0.0001" />
        </div>
        <div class="form-group">
          <label>LoRA Rank</label>
          <input id="tj-rank" type="number" value="32" min="4" max="128" />
        </div>
        <div class="form-group">
          <label>Frames</label>
          <input id="tj-frames" type="number" value="249" min="25" max="301" />
        </div>
        <div class="form-group" style="grid-column:span 2">
          <label>GPU Type</label>
          <select id="tj-gpu">
            <option value="NVIDIA H100 80GB HBM3">H100 80GB (recommended)</option>
            <option value="NVIDIA A100-SXM4-80GB">A100 80GB</option>
            <option value="NVIDIA GeForce RTX 4090">RTX 4090</option>
          </select>
        </div>
      </div>
    </details>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal('new-job-modal')">Cancel</button>
      <button class="btn btn-primary" onclick="submitNewJob()">Start Training</button>
    </div>
  </div>
</div>

<!-- Training Job Detail Modal -->
<div id="job-detail-modal" class="modal hidden">
  <div class="modal-box" style="max-width:800px">
    <h3 id="detail-title">Training Job</h3>
    <div style="display:flex;gap:12px;margin-bottom:12px">
      <span id="detail-status" class="badge"></span>
      <span id="detail-progress" style="color:#aaa"></span>
    </div>
    <div id="log-panel"
         style="background:#0a0a0a;border:1px solid #333;padding:12px;height:300px;overflow-y:auto;font-family:monospace;font-size:12px;white-space:pre-wrap"></div>
    <h4 style="margin-top:16px">Checkpoints</h4>
    <div id="checkpoint-list"></div>
    <div class="modal-actions">
      <button class="btn btn-danger" id="cancel-job-btn" onclick="cancelTrainingJob()">Cancel Job</button>
      <button class="btn" onclick="closeModal('job-detail-modal')">Close</button>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Add Training JavaScript**

Find the closing `</script>` tag in `admin.html` and add before it:

```javascript
// ── Training Tab ──────────────────────────────────────────────────

let trainingJobs = [];
let selectedJobId = null;
let logEventSource = null;
let pendingVideos = []; // [{file, caption}]

async function loadTraining() {
  const data = await apiFetch('GET', '/v1/admin/training/jobs');
  trainingJobs = Array.isArray(data) ? data : [];
  renderTrainingTable();
}

function renderTrainingTable() {
  const tbody = document.getElementById('training-body');
  if (!trainingJobs.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666">No training jobs yet</td></tr>';
    return;
  }
  tbody.innerHTML = trainingJobs.map(j => {
    const pct = j.total_steps > 0 ? Math.round((j.current_step / j.total_steps) * 100) : 0;
    const statusColor = { done: '#10b981', failed: '#ef4444', training: '#7c3aed', provisioning: '#f59e0b', cancelled: '#6b7280' }[j.status] || '#aaa';
    return `<tr>
      <td>${j.position}</td>
      <td><span style="color:${statusColor}">${j.status}</span></td>
      <td>
        <div style="background:#222;border-radius:4px;height:8px;width:120px">
          <div style="background:#7c3aed;height:8px;border-radius:4px;width:${pct}%"></div>
        </div>
        <span style="font-size:11px;color:#aaa">${j.current_step}/${j.total_steps}</span>
      </td>
      <td style="font-size:12px;color:#aaa">${(j.config && j.config.gpu_type_id) ? j.config.gpu_type_id.replace('NVIDIA ', '') : '—'}</td>
      <td style="font-size:12px;color:#aaa">${new Date(j.created_at * 1000).toLocaleString()}</td>
      <td>
        <button class="btn btn-sm" onclick="openJobDetail('${j.id}')">Logs</button>
        ${j.status === 'done' ? `<button class="btn btn-sm btn-primary" onclick="openJobDetail('${j.id}')">Deploy</button>` : ''}
      </td>
    </tr>`;
  }).join('');
}

function showNewJobModal() {
  pendingVideos = [];
  document.getElementById('video-list').innerHTML = '';
  document.getElementById('tj-position').value = '';
  document.getElementById('new-job-modal').classList.remove('hidden');
}

function handleVideoDrop(e) {
  e.preventDefault();
  handleVideoFiles(e.dataTransfer.files);
}

function handleVideoFiles(files) {
  for (const file of files) {
    if (!file.name.endsWith('.mp4')) continue;
    const caption = file.name.replace('.mp4', '').replace(/_/g, ' ');
    pendingVideos.push({ file, caption });
  }
  renderVideoList();
}

function renderVideoList() {
  const el = document.getElementById('video-list');
  el.innerHTML = pendingVideos.map((v, i) => `
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
      <span style="min-width:160px;font-size:13px;color:#aaa;overflow:hidden;text-overflow:ellipsis">${v.file.name}</span>
      <input type="text" value="${v.caption}" style="flex:1"
             onchange="pendingVideos[${i}].caption = this.value" placeholder="Caption..." />
      <button class="btn btn-sm" onclick="pendingVideos.splice(${i},1);renderVideoList()">✕</button>
    </div>
  `).join('');
}

async function submitNewJob() {
  const position = document.getElementById('tj-position').value.trim();
  if (!position) return alert('Position name is required');
  if (!pendingVideos.length) return alert('Upload at least one video');

  const jobId = crypto.randomUUID();

  // Upload videos first
  showToast('Uploading videos...');
  for (const v of pendingVideos) {
    const fd = new FormData();
    fd.append('job_id', jobId);
    fd.append('filename', v.file.name);
    fd.append('file', v.file);
    const resp = await fetch('/v1/admin/training/jobs/upload-video', {
      method: 'POST',
      headers: { 'X-Admin-Password': adminToken },
      body: fd,
    });
    if (!resp.ok) return alert('Video upload failed: ' + v.file.name);
  }

  // Create job
  const body = {
    position,
    videos: pendingVideos.map(v => ({ filename: v.file.name, caption: v.caption })),
    steps: parseInt(document.getElementById('tj-steps').value),
    learning_rate: parseFloat(document.getElementById('tj-lr').value),
    rank: parseInt(document.getElementById('tj-rank').value),
    frames: parseInt(document.getElementById('tj-frames').value),
    gpu_type_id: document.getElementById('tj-gpu').value,
  };

  // Override job_id by pre-uploading — but the create route generates its own UUID.
  // We need to pass job_id so the videos match. Use a workaround: re-upload with real job_id.
  // Simpler: create job first (no videos), get job_id, then upload. But our route does both.
  // For now: the upload uses a client-generated jobId, and we pass it as a custom header.
  // ACTUAL FLOW: POST /v1/admin/training/jobs returns job_id; videos are uploaded with that id.
  // The submit flow here is: create empty job first → get id → upload → patch job.
  // To keep it simple, we create the job and trust the server to use the pre-uploaded video keys.

  const result = await apiFetch('POST', '/v1/admin/training/jobs', body);
  if (result.id) {
    showToast('Training job created: ' + result.id.slice(0, 8));
    closeModal('new-job-modal');
    loadTraining();
  } else {
    alert('Failed to create job: ' + JSON.stringify(result));
  }
}

async function openJobDetail(jobId) {
  selectedJobId = jobId;
  const job = await apiFetch('GET', `/v1/admin/training/jobs/${jobId}`);

  document.getElementById('detail-title').textContent = `Training: ${job.position}`;
  const statusEl = document.getElementById('detail-status');
  statusEl.textContent = job.status;
  document.getElementById('detail-progress').textContent =
    `Step ${job.current_step} / ${job.total_steps}`;
  document.getElementById('job-detail-modal').classList.remove('hidden');

  // Checkpoints
  const ckEl = document.getElementById('checkpoint-list');
  if (job.checkpoints && job.checkpoints.length) {
    ckEl.innerHTML = job.checkpoints.map(ck => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:8px;background:#111;border-radius:6px;margin-bottom:6px">
        <span style="font-size:13px">${ck.name} <span style="color:#aaa">(${ck.size_mb} MB)</span></span>
        <button class="btn btn-sm btn-primary" onclick="deployCheckpoint('${jobId}','${ck.name}')">Deploy</button>
      </div>
    `).join('');
  } else {
    ckEl.innerHTML = '<p style="color:#666">No checkpoints yet</p>';
  }

  // Start log stream
  if (logEventSource) logEventSource.close();
  const logPanel = document.getElementById('log-panel');
  logPanel.textContent = '';
  const response = await fetch(`/v1/admin/training/jobs/${jobId}/logs`, {
    headers: { 'X-Admin-Password': adminToken },
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  (async () => {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      logPanel.textContent += decoder.decode(value);
      logPanel.scrollTop = logPanel.scrollHeight;
    }
  })();
}

async function deployCheckpoint(jobId, checkpointName) {
  if (!confirm(`Deploy ${checkpointName}? This will overwrite the current production LoRA for this position.`)) return;
  const result = await apiFetch('POST', `/v1/admin/training/jobs/${jobId}/deploy/${checkpointName}`);
  if (result.ok) {
    showToast(`Deployed: ${result.deployed}. Restart inference server to apply.`);
  } else {
    alert('Deploy failed: ' + JSON.stringify(result));
  }
}

async function cancelTrainingJob() {
  if (!selectedJobId) return;
  if (!confirm('Cancel this training job and terminate the pod?')) return;
  await apiFetch('DELETE', `/v1/admin/training/jobs/${selectedJobId}`);
  closeModal('job-detail-modal');
  loadTraining();
}
```

- [ ] **Step 4: Wire loadTraining into the showTab function**

In `admin.html`, find the `showTab` function and add a case for training:

```javascript
// Inside showTab function, after the existing cases:
if (tab === 'training') loadTraining();
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/frontend/admin.html
git commit -m "feat: add Training tab to admin UI"
```

---

## Task 14: Fix Video Upload Flow + Deploy to Railway

The video upload flow has a sequencing issue: videos are uploaded with a client-generated `job_id`, but the server creates its own `job_id`. Fix by splitting the route into two steps.

**Files:**
- Modify: `parrot-service/api/routes/training.py`

- [ ] **Step 1: Add a `POST /jobs/prepare` route that reserves a job_id**

In `parrot-service/api/routes/training.py`, add before the existing `create_training_job` route:

```python
@router.post("/jobs/prepare")
async def prepare_training_job():
    """Reserve a job_id so the client can upload videos before creating the job."""
    job_id = str(uuid.uuid4())
    return {"job_id": job_id}
```

- [ ] **Step 2: Modify `create_training_job` to accept an optional `job_id`**

Replace the first two lines of `create_training_job`:

```python
@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(req: CreateTrainingJobRequest):
    job_id = req.job_id if req.job_id else str(uuid.uuid4())
    r2_prefix = f"training/{job_id}"
```

And add `job_id: Optional[str] = None` to `CreateTrainingJobRequest` in `models.py`:

```python
class CreateTrainingJobRequest(BaseModel):
    job_id: Optional[str] = None   # pre-reserved via /jobs/prepare
    position: str
    videos: list[VideoCaption]
    steps: int = Field(default=2000, ge=100, le=5000)
    learning_rate: float = Field(default=1e-4, gt=0)
    rank: int = Field(default=32, ge=4, le=128)
    frames: int = Field(default=249, ge=25, le=301)
    gpu_type_id: Optional[str] = None
    validation_prompt: Optional[str] = None
```

- [ ] **Step 3: Update the admin UI submitNewJob to use the two-step flow**

In `admin.html`, replace the `submitNewJob` function body:

```javascript
async function submitNewJob() {
  const position = document.getElementById('tj-position').value.trim();
  if (!position) return alert('Position name is required');
  if (!pendingVideos.length) return alert('Upload at least one video');

  // Step 1: Reserve job_id
  const prep = await apiFetch('POST', '/v1/admin/training/jobs/prepare');
  const jobId = prep.job_id;

  // Step 2: Upload videos
  showToast('Uploading videos...');
  for (const v of pendingVideos) {
    const fd = new FormData();
    fd.append('job_id', jobId);
    fd.append('filename', v.file.name);
    fd.append('file', v.file);
    const resp = await fetch('/v1/admin/training/jobs/upload-video', {
      method: 'POST',
      headers: { 'X-Admin-Password': adminToken },
      body: fd,
    });
    if (!resp.ok) return alert('Video upload failed: ' + v.file.name);
  }

  // Step 3: Create job with pre-reserved job_id
  const body = {
    job_id: jobId,
    position,
    videos: pendingVideos.map(v => ({ filename: v.file.name, caption: v.caption })),
    steps: parseInt(document.getElementById('tj-steps').value),
    learning_rate: parseFloat(document.getElementById('tj-lr').value),
    rank: parseInt(document.getElementById('tj-rank').value),
    frames: parseInt(document.getElementById('tj-frames').value),
    gpu_type_id: document.getElementById('tj-gpu').value,
  };

  const result = await apiFetch('POST', '/v1/admin/training/jobs', body);
  if (result.id) {
    showToast('Training job created: ' + result.id.slice(0, 8));
    closeModal('new-job-modal');
    loadTraining();
  } else {
    alert('Failed to create job: ' + JSON.stringify(result));
  }
}
```

- [ ] **Step 4: Add Railway environment variables**

In Railway dashboard for `parrot-service`, add:
```
RUNPOD_API_KEY=<your RunPod API key from https://www.runpod.io/console/user/settings>
RUNPOD_GPU_TYPE_ID=NVIDIA H100 80GB HBM3
RUNPOD_IMAGE_NAME=racoonn/lora-trainer-agent:latest
AGENT_SECRET=<generate a random 32-char string>
RUNPOD_CONTAINER_DISK_GB=100
AGENT_PORT=7860
```

- [ ] **Step 5: Commit and push**

```bash
cd "/Users/welly/Parrot API"
git add parrot-service/api/routes/training.py parrot-service/api/models.py parrot-service/frontend/admin.html
git commit -m "fix: two-step job creation flow (prepare → upload → create)"
git push origin main
```

---

## Task 15: Build and Push Docker Image

The pod agent Docker image must be built and pushed to Docker Hub as `racoonn/lora-trainer-agent:latest`.

- [ ] **Step 1: Build the Docker image locally (requires Docker)**

```bash
cd "/Users/welly/Parrot API/parrot-service/pod_agent"
docker build -t racoonn/lora-trainer-agent:latest .
```
Expected: Build completes successfully. This will take 10-20 minutes due to ltx-trainer deps.

- [ ] **Step 2: Push to Docker Hub**

```bash
docker login  # enter Docker Hub credentials
docker push racoonn/lora-trainer-agent:latest
```
Expected: `latest: digest: sha256:... size: ...`

- [ ] **Step 3: Verify image is accessible**

```bash
docker pull racoonn/lora-trainer-agent:latest
```
Expected: Image pulled successfully.

- [ ] **Step 4: Commit image reference note**

```bash
cd "/Users/welly/Parrot API"
git commit --allow-empty -m "chore: pod agent image published as racoonn/lora-trainer-agent:latest"
```

---

## Task 16: End-to-End Smoke Test

- [ ] **Step 1: Verify training routes are accessible after Railway deploy**

```bash
curl -s -o /dev/null -w "%{http_code}" \
  -H "X-Admin-Password: adminPiigu888" \
  https://www.racoonn.me/v1/admin/training/jobs
```
Expected: `200`

- [ ] **Step 2: Test prepare endpoint**

```bash
curl -s -X POST \
  -H "X-Admin-Password: adminPiigu888" \
  https://www.racoonn.me/v1/admin/training/jobs/prepare
```
Expected: `{"job_id": "<uuid>"}`

- [ ] **Step 3: Test job creation without a real pod (RUNPOD_API_KEY not set = should return error)**

```bash
curl -s -X POST \
  -H "X-Admin-Password: adminPiigu888" \
  -H "Content-Type: application/json" \
  -d '{"position":"test","videos":[{"filename":"test.mp4","caption":"test"}]}' \
  https://www.racoonn.me/v1/admin/training/jobs
```
Expected: Job is created with status `provisioning`, then quickly transitions to `failed` (RunPod API key not configured or pod creation fails — this confirms the orchestrator runs and handles errors correctly).

- [ ] **Step 4: Verify Training tab appears in admin UI**

Open `https://www.racoonn.me/admin` in browser, log in with `adminPiigu888`. Confirm "Training" tab is visible and shows empty state.

- [ ] **Step 5: Full end-to-end test (requires RUNPOD_API_KEY set in Railway)**

Set `RUNPOD_API_KEY` in Railway env vars, then use the UI to:
1. Click "Training" tab → "+ New Training Job"
2. Enter position: `test_position`
3. Drop a short MP4 file
4. Enter a caption
5. Select RTX 4090 (cheapest)
6. Click "Start Training"
7. Open job detail → observe logs streaming as pod provisions

Expected: Pod is created in RunPod dashboard, logs appear within 10 minutes once agent starts.
