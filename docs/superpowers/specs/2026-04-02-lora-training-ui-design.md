# LoRA Training UI Design

## Overview

A web-based end-to-end LoRA training interface integrated into the existing `parrot-service` admin panel. Users upload training videos through the browser, configure training parameters, and the system provisions an on-demand RunPod GPU pod, runs ltx-trainer, streams progress, and stores checkpoints to R2.

**Goal:** Replace the manual SSH/SCP training workflow with a fully automated UI so non-technical users can train position LoRAs.

---

## Architecture

```
Browser (admin UI)
      ↕
Railway: parrot-service (FastAPI)
      ↕                    ↕
  PostgreSQL             R2 (videos + checkpoints)
      ↕
  RunPod REST API
      ↕
  RunPod Pod (on-demand H100/A100)
      ↕
  Pod Agent (FastAPI, port 7860)
      ↕
  ltx-trainer scripts
```

**Data flow:**
1. Admin fills training form (position name, captions, params) and uploads MP4s
2. Railway uploads videos to R2 under `training/{job_id}/videos/`
3. Railway calls RunPod API → creates pod from custom Docker image
4. Railway polls RunPod API until pod status = `RUNNING` with public IP
5. Railway calls pod agent `POST /setup` — agent downloads videos from R2, generates JSONL + YAML config
6. Railway calls pod agent `POST /train` — agent spawns ltx-trainer subprocess
7. UI polls `GET /v1/admin/training/jobs/{id}/logs` (SSE) → Railway proxies from pod agent
8. On completion, agent uploads checkpoints to R2 under `training/{job_id}/checkpoints/`
9. Railway marks job `done`, stores checkpoint list in PG
10. Admin selects best checkpoint, clicks Deploy → Railway SSH-copies checkpoint to inference server LoRA directory

---

## Components

### 1. Railway Orchestrator (new routes in parrot-service)

**Routes:**
```
POST   /v1/admin/training/jobs                    — create job, upload videos, provision pod
GET    /v1/admin/training/jobs                    — list all training jobs
GET    /v1/admin/training/jobs/{id}               — job detail + checkpoint list
GET    /v1/admin/training/jobs/{id}/logs          — SSE log stream (proxied from pod agent)
POST   /v1/admin/training/jobs/{id}/deploy/{ckpt} — copy checkpoint to inference server
DELETE /v1/admin/training/jobs/{id}               — terminate pod + cleanup
```

**Background task (polling loop):**
- Every 10s, polls RunPod API for pod status
- On `RUNNING`: calls agent `/setup` then `/train`
- On agent training complete: calls agent `/upload-checkpoints`, updates PG, terminates pod
- On error: marks job failed, terminates pod

### 2. Pod Agent (FastAPI, runs on RunPod pod)

Pre-installed in Docker image, starts automatically at pod boot on port 7860.

**Endpoints:**
```
POST /setup               — download videos from R2, write YAML + JSONL
POST /train               — spawn ltx-trainer, return immediately (async)
GET  /status              — { step, total_steps, phase, vram_gb, running }
GET  /logs                — SSE stream of training log file
GET  /checkpoints         — list .safetensors files with sizes
POST /upload-checkpoints  — push all checkpoints to R2
DELETE /shutdown          — terminate pod via RunPod API then exit
```

**Training phases tracked:**
`idle → preprocessing → training → uploading → done`

### 3. Custom Docker Image

Based on `runpod/pytorch:2.1.0-py3.10-cuda12.1-devel`.

Pre-installed:
- ltx-trainer + all Python deps (torch, transformers, diffusers, etc.)
- Pod agent + uvicorn
- boto3 (for R2 access)
- supervisord (manages agent startup)

Startup script:
```bash
supervisord -c /etc/supervisor/conf.d/agent.conf
```

Published to Docker Hub as `racoonn/lora-trainer-agent:latest`.

### 4. PostgreSQL Schema

New table `training_jobs`:
```sql
CREATE TABLE training_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'provisioning',
    -- status values: provisioning | preprocessing | training | uploading | done | failed
    pod_id      TEXT,
    pod_ip      TEXT,
    r2_prefix   TEXT NOT NULL,   -- e.g. "training/abc123"
    config      JSONB NOT NULL,  -- steps, lr, rank, frames, gpu_type
    current_step INT DEFAULT 0,
    total_steps  INT DEFAULT 2000,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);
```

### 5. Admin UI Extension

New tab "Training" added to existing `frontend/admin.html`.

**Training tab contains:**
- **Jobs table**: position, status badge, step progress bar (`current_step/total_steps`), GPU cost estimate, created time, actions
- **New Job button** → modal form:
  - Position name (text input)
  - Video upload zone (drag & drop, shows file list with caption inputs per file)
  - Training params (collapsible): steps (2000), learning rate (1e-4), rank (32), frames (249)
  - GPU type selector: H100 SXM (recommended) / A100 / RTX 4090
- **Job detail view** (click row):
  - Real-time log panel (auto-scrolls, monospace)
  - Step progress bar with phase label
  - VRAM usage gauge
  - Checkpoints list with Download + Deploy buttons

---

## Training Config Generation

From UI form inputs, Railway generates the ltx-trainer YAML:

```yaml
model:
  model_path: "/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors"
  text_encoder_path: "/workspace/gemma_configs"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: {rank}          # from UI, default 32
  alpha: {rank}
  dropout: 0.0
  target_modules: [to_k, to_q, to_v, to_out.0]

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.9
  with_audio: false

optimization:
  learning_rate: {lr}   # from UI, default 1e-4
  steps: {steps}        # from UI, default 2000
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "linear"
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"
  load_text_encoder_in_8bit: true

data:
  preprocessed_data_root: "/workspace/training/{job_id}_preprocessed"
  num_dataloader_workers: 2

validation:
  prompts: ["{position caption} --{position}"]
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 768, {frames}]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: 500
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: 500
  keep_last_n: 4
  precision: "bfloat16"

seed: 42
output_dir: "/workspace/training/{job_id}_output"
```

---

## Environment Variables

Added to Railway `parrot-service`:
```
RUNPOD_API_KEY=<runpod api key>
RUNPOD_GPU_TYPE_ID=NVIDIA H100 80GB HBM3   # default, overridable per job
RUNPOD_TEMPLATE_ID=<custom docker template id>
RUNPOD_NETWORK_VOLUME_ID=<existing /workspace volume id>
AGENT_PORT=7860
AGENT_SECRET=<shared secret for agent auth>
```

---

## Security

- Pod agent validates `X-Agent-Secret` header on all requests (set at pod startup via env var)
- Railway passes `AGENT_SECRET` when creating the pod via RunPod API env vars
- R2 credentials passed to pod agent via RunPod env vars at pod creation time
- Admin UI protected by existing `X-Admin-Password` mechanism

---

## Error Handling

| Failure point | Recovery |
|---|---|
| Pod fails to start | Mark job failed, show RunPod error |
| Agent unreachable after 5 min | Mark job failed, terminate pod |
| ltx-trainer OOM | Agent captures exit code, marks phase failed |
| R2 upload fails | Retry 3x, then mark job failed |
| Pod terminated mid-training | Polling loop detects pod gone, marks job failed |

---

## Out of Scope

- Multi-user training (single admin only, same as current system)
- Training on multiple positions simultaneously (one job at a time per pod)
- Automatic hyperparameter tuning
- Validation video preview in UI (checkpoints only)
