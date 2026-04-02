# Credits System & Admin Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-key credits system (0.035 credits/second) with PostgreSQL storage and a password-protected Admin web UI to parrot-service.

**Architecture:** API keys move from Redis-only to PostgreSQL (`api_keys` table) with Redis as a 60-second cache. Credits are checked at submit time (no deduct) and atomically deducted in PG only on successful job completion. Admin UI is a single HTML file served at `/admin`.

**Tech Stack:** FastAPI, asyncpg (PostgreSQL), redis-asyncio, vanilla JS + fetch API (Admin HTML)

---

## File Map

| File | Change |
|------|--------|
| `parrot-service/db/job_store.py` | Add `api_keys` CRUD + `deduct_credits` + `get_jobs_paginated` + `get_key_jobs` |
| `parrot-service/api/config.py` | Add `ADMIN_PASSWORD`, `CREDITS_PER_SECOND` |
| `parrot-service/api/auth.py` | `verify_api_key` reads PG+Redis cache; add `require_admin`; update `create_api_key` to write PG |
| `parrot-service/api/models.py` | Add `InsufficientCreditsError`, `AccountJobsResponse`, `AdminKeyResponse`, `AdminJobResponse` |
| `parrot-service/api/routes/generate.py` | Check credits balance before queuing |
| `parrot-service/workers/gpu_worker.py` | Call `deduct_credits` on job completion |
| `parrot-service/api/routes/account.py` | Add `GET /v1/account/jobs`; update `/account/usage` to include credits |
| `parrot-service/api/routes/admin.py` | Expand endpoints; add `require_admin` protection |
| `parrot-service/api/main.py` | Mount `/admin` to serve `admin.html` |
| `parrot-service/frontend/admin.html` | New Admin UI (login + 4 tabs) |

---

## Task 1: Database Schema — `api_keys` table + `credits_charged` column

**Files:**
- Modify: `parrot-service/db/job_store.py`

- [ ] **Step 1: Add schema constants** to `db/job_store.py` — add these two SQL strings right after the existing `CREATE_TABLE_SQL`:

```python
CREATE_API_KEYS_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      DOUBLE PRECISION NOT NULL,
    disabled        BOOLEAN NOT NULL DEFAULT FALSE,
    credits         NUMERIC(12,4) NOT NULL DEFAULT 0,
    credits_used    NUMERIC(12,4) NOT NULL DEFAULT 0,
    total_jobs      INTEGER NOT NULL DEFAULT 0,
    completed_jobs  INTEGER NOT NULL DEFAULT 0,
    failed_jobs     INTEGER NOT NULL DEFAULT 0
);
"""

ADD_CREDITS_CHARGED_SQL = """
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS credits_charged NUMERIC(12,4) DEFAULT 0;
"""
```

- [ ] **Step 2: Run migrations on startup** — in `get_pool()`, after the existing `CREATE TABLE IF NOT EXISTS jobs` call, add:

```python
async with _pool.acquire() as conn:
    await conn.execute(CREATE_TABLE_SQL)
    await conn.execute(CREATE_API_KEYS_SQL)
    await conn.execute(ADD_CREDITS_CHARGED_SQL)
```

Replace the existing single-execute block with the above three lines.

- [ ] **Step 3: Add `upsert_api_key` function** — add at the end of `db/job_store.py`:

```python
async def upsert_api_key(key_hash: str, name: str, created_at: float, credits: float = 0.0):
    """Insert a new API key (called by create_api_key). No-op if already exists."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO api_keys (key_hash, name, created_at, credits)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (key_hash) DO NOTHING
            """,
            key_hash, name, created_at, credits,
        )


async def get_api_key(key_hash: str) -> Optional[dict]:
    """Fetch key metadata from PostgreSQL."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM api_keys WHERE key_hash = $1", key_hash
        )
    if not row:
        return None
    return dict(row)


async def list_api_keys() -> list[dict]:
    """List all API keys for admin."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM api_keys ORDER BY created_at DESC"
        )
    return [dict(r) for r in rows]


async def update_api_key(key_hash: str, *, disabled: Optional[bool] = None, add_credits: Optional[float] = None):
    """Admin: disable/enable key or top up credits."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if disabled is not None:
            await conn.execute(
                "UPDATE api_keys SET disabled = $1 WHERE key_hash = $2",
                disabled, key_hash,
            )
        if add_credits is not None:
            await conn.execute(
                "UPDATE api_keys SET credits = credits + $1 WHERE key_hash = $2",
                add_credits, key_hash,
            )


async def check_credits(key_hash: str) -> float:
    """Return current credits balance. Returns -1 if key not found."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT credits FROM api_keys WHERE key_hash = $1", key_hash
        )
    if not row:
        return -1.0
    return float(row["credits"])


async def deduct_credits(key_hash: str, amount: float, job_id: str) -> bool:
    """
    Atomically deduct credits on job completion.
    Returns True if deducted, False if insufficient (should not happen — checked at submit).
    Also writes credits_charged to the jobs table.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                UPDATE api_keys
                SET credits = credits - $1,
                    credits_used = credits_used + $1,
                    completed_jobs = completed_jobs + 1
                WHERE key_hash = $2 AND credits >= $1
                RETURNING credits
                """,
                amount, key_hash,
            )
            if row is None:
                return False
            await conn.execute(
                "UPDATE jobs SET credits_charged = $1 WHERE job_id = $2",
                amount, job_id,
            )
            return True


async def increment_api_key_jobs(key_hash: str, field: str):
    """Increment total_jobs or failed_jobs counter."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE api_keys SET {field} = {field} + 1 WHERE key_hash = $1",
            key_hash,
        )


async def get_jobs_paginated(page: int, limit: int, key_hash: Optional[str] = None, status: Optional[str] = None) -> tuple[list[dict], int]:
    """Admin: paginated job list with optional filters."""
    pool = await get_pool()
    conditions = []
    params = []
    idx = 1

    if key_hash:
        conditions.append(f"j.api_key_hash = ${idx}")
        params.append(key_hash)
        idx += 1
    if status:
        conditions.append(f"j.status = ${idx}")
        params.append(status)
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM jobs j {where}", *params
        )
        rows = await conn.fetch(
            f"""
            SELECT j.*, k.name as key_name
            FROM jobs j
            LEFT JOIN api_keys k ON j.api_key_hash = k.key_hash
            {where}
            ORDER BY j.created_at DESC
            LIMIT ${idx} OFFSET ${idx+1}
            """,
            *params, limit, (page - 1) * limit,
        )
    return [dict(r) for r in rows], total


async def get_key_jobs(key_hash: str, page: int, limit: int, status: Optional[str] = None) -> tuple[list[dict], int]:
    """User: paginated job list for a specific key."""
    pool = await get_pool()
    conditions = ["api_key_hash = $1"]
    params: list = [key_hash]
    idx = 2

    if status:
        conditions.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    where = "WHERE " + " AND ".join(conditions)

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM jobs {where}", *params
        )
        rows = await conn.fetch(
            f"""
            SELECT job_id, position, duration, status, credits_charged,
                   video_url, created_at, completed_at, prompt, seed, callback_url
            FROM jobs {where}
            ORDER BY created_at DESC
            LIMIT ${idx} OFFSET ${idx+1}
            """,
            *params, limit, (page - 1) * limit,
        )
    return [dict(r) for r in rows], total
```

- [ ] **Step 4: Verify the file parses correctly**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from db import job_store; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd "/Users/welly/Parrot API/parrot-service"
git add db/job_store.py
git commit -m "feat: add api_keys table, credits columns, and CRUD functions to job_store"
```

---

## Task 2: Config — add `ADMIN_PASSWORD` and `CREDITS_PER_SECOND`

**Files:**
- Modify: `parrot-service/api/config.py`

- [ ] **Step 1: Add two settings** to the `Settings` class in `api/config.py`, after `QUEUE_EXPIRE_SECONDS`:

```python
    # ── Admin ──
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")

    # ── Credits ──
    CREDITS_PER_SECOND: float = float(os.getenv("CREDITS_PER_SECOND", "0.035"))
```

- [ ] **Step 2: Verify**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from api.config import settings; print(settings.CREDITS_PER_SECOND, settings.ADMIN_PASSWORD)"
```

Expected: `0.035 ` (empty string if env not set)

- [ ] **Step 3: Commit**

```bash
git add api/config.py
git commit -m "feat: add ADMIN_PASSWORD and CREDITS_PER_SECOND to config"
```

---

## Task 3: Auth — PG+Redis cache verify, `require_admin`, update `create_api_key`

**Files:**
- Modify: `parrot-service/api/auth.py`

- [ ] **Step 1: Replace `api/auth.py` entirely** with the following:

```python
from __future__ import annotations

import hashlib
import secrets
import time
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from api.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_redis: Optional[aioredis.Redis] = None

CACHE_TTL = 60  # seconds


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def _cache_key_data(r: aioredis.Redis, key_hash: str, data: dict):
    """Write key metadata to Redis cache with TTL."""
    await r.hset(f"apikey:{key_hash}", mapping={
        "name": data.get("name", ""),
        "disabled": "1" if data.get("disabled") else "0",
        "credits": str(data.get("credits", 0)),
        "total_jobs": str(data.get("total_jobs", 0)),
        "completed_jobs": str(data.get("completed_jobs", 0)),
        "failed_jobs": str(data.get("failed_jobs", 0)),
    })
    await r.expire(f"apikey:{key_hash}", CACHE_TTL)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    r = await get_redis()
    key_hash = hash_key(api_key)

    # Try Redis cache first
    cache_ttl = await r.ttl(f"apikey:{key_hash}")
    if cache_ttl > 0:
        key_data = await r.hgetall(f"apikey:{key_hash}")
        if key_data:
            if key_data.get("disabled") == "1":
                raise HTTPException(status_code=403, detail="API key disabled")
            return key_hash

    # Cache miss — query PostgreSQL
    if not settings.DATABASE_URL:
        # Fallback: accept any key that exists in Redis (legacy mode)
        key_data = await r.hgetall(f"apikey:{key_hash}")
        if not key_data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        if key_data.get("disabled") == "1":
            raise HTTPException(status_code=403, detail="API key disabled")
        return key_hash

    from db import job_store
    pg_data = await job_store.get_api_key(key_hash)
    if not pg_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if pg_data.get("disabled"):
        raise HTTPException(status_code=403, detail="API key disabled")

    # Populate cache
    await _cache_key_data(r, key_hash, pg_data)
    return key_hash


async def create_api_key(name: str, credits: float = 0.0) -> str:
    """Create a new API key — writes to PostgreSQL (and optionally Redis)."""
    raw_key = f"pk_{secrets.token_urlsafe(32)}"
    key_hash = hash_key(raw_key)
    now = time.time()

    if settings.DATABASE_URL:
        from db import job_store
        await job_store.upsert_api_key(key_hash, name, now, credits)

    # Also write to Redis set for legacy billing report
    r = await get_redis()
    await r.sadd("apikeys", key_hash)
    await _cache_key_data(r, key_hash, {
        "name": name,
        "disabled": False,
        "credits": credits,
        "total_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
    })

    return raw_key


async def increment_usage(key_hash: str, field: str = "total_jobs"):
    """Increment Redis usage counters (kept for backwards compat)."""
    r = await get_redis()
    await r.hincrby(f"apikey:{key_hash}", field, 1)
    month = __import__("datetime").datetime.utcnow().strftime("%Y-%m")
    await r.hincrby(f"apikey:{key_hash}:usage:{month}", "count", 1)
    # Invalidate cache so next verify re-reads fresh data
    await r.expire(f"apikey:{key_hash}", 1)


async def require_admin(x_admin_password: Optional[str] = None):
    """FastAPI dependency for admin routes."""
    from fastapi import Header
    return x_admin_password


def make_require_admin():
    """Returns a FastAPI dependency that checks X-Admin-Password header."""
    from fastapi import Header

    async def _require_admin(x_admin_password: Optional[str] = Header(default=None)):
        if not settings.ADMIN_PASSWORD:
            raise HTTPException(status_code=500, detail="ADMIN_PASSWORD not configured")
        if not x_admin_password:
            raise HTTPException(status_code=401, detail="Missing X-Admin-Password header")
        if x_admin_password != settings.ADMIN_PASSWORD:
            raise HTTPException(status_code=403, detail="Invalid admin password")

    return _require_admin

require_admin = make_require_admin()
```

- [ ] **Step 2: Verify**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from api.auth import verify_api_key, create_api_key, require_admin; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/auth.py
git commit -m "feat: verify_api_key reads PG+Redis cache; add require_admin; create_api_key writes PG"
```

---

## Task 4: Models — add credits and admin Pydantic models

**Files:**
- Modify: `parrot-service/api/models.py`

- [ ] **Step 1: Add new models** at the end of `api/models.py`:

```python

# ── Credits ─────────────────────────────────────────────────────

class InsufficientCreditsResponse(BaseModel):
    error: str = "Insufficient credits"
    required: float
    available: float


class AccountUsageResponse(BaseModel):
    api_key: str
    credits: float
    credits_used: float
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    month: str


class AccountJobItem(BaseModel):
    job_id: str
    position: str
    duration: int
    status: str
    credits_charged: float
    video_url: Optional[str]
    created_at: float
    completed_at: Optional[float]
    prompt: str
    seed: int
    callback_url: str


class AccountJobsResponse(BaseModel):
    jobs: list[AccountJobItem]
    total: int
    page: int
    limit: int


# ── Admin ────────────────────────────────────────────────────────

class AdminKeyItem(BaseModel):
    key_hash: str
    name: str
    created_at: float
    disabled: bool
    credits: float
    credits_used: float
    total_jobs: int
    completed_jobs: int
    failed_jobs: int


class AdminJobItem(BaseModel):
    job_id: str
    key_name: Optional[str]
    api_key_hash: str
    position: str
    duration: int
    status: str
    credits_charged: float
    video_url: Optional[str]
    created_at: float
    completed_at: Optional[float]
    prompt: str
    seed: int
    callback_url: str


class AdminJobsResponse(BaseModel):
    jobs: list[AdminJobItem]
    total: int
    page: int
    limit: int


class CreateKeyRequest(BaseModel):
    name: str
    credits: float = 0.0


class TopUpRequest(BaseModel):
    add_credits: float


class DisableKeyRequest(BaseModel):
    disabled: bool
```

- [ ] **Step 2: Also update existing `UsageResponse`** — it's still used by old code, keep it. The new `AccountUsageResponse` is the replacement. No action needed.

- [ ] **Step 3: Verify**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from api.models import InsufficientCreditsResponse, AdminJobItem, CreateKeyRequest; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add api/models.py
git commit -m "feat: add credits and admin Pydantic models"
```

---

## Task 5: Generate route — credits balance check before queuing

**Files:**
- Modify: `parrot-service/api/routes/generate.py`

- [ ] **Step 1: Add credits check** in `submit_generate`, right after `key_hash: str = Depends(verify_api_key)` resolves and before `_save_input_image`. Insert after line `r = await get_redis()`:

```python
    # ── Credits check ──────────────────────────────────────────
    cost = req.duration * settings.CREDITS_PER_SECOND
    if settings.DATABASE_URL:
        from db import job_store as _js
        available = await _js.check_credits(key_hash)
        if available >= 0 and available < cost:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Insufficient credits",
                    "required": round(cost, 4),
                    "available": round(available, 4),
                },
            )
```

Place this block immediately after `r = await get_redis()` on line 64.

- [ ] **Step 2: Add `settings` import** — at the top of `generate.py`, the import `from api.config import settings` already exists. Confirm it's there (line 13). If not, add it.

- [ ] **Step 3: Verify the file parses**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from api.routes.generate import router; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Manual test — credits check**

Start the service locally (requires Redis + PG running):
```bash
ADMIN_PASSWORD=test123 DATABASE_URL=postgresql://... uvicorn api.main:app --port 8000
```

Then try submitting with a key that has 0 credits:
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "X-API-Key: pk_..." \
  -H "Content-Type: application/json" \
  -d '{"image_url":"http://example.com/img.jpg","position":"cowgirl","duration":10}'
```

Expected response: `HTTP 402` with `{"detail":{"error":"Insufficient credits","required":0.35,"available":0.0}}`

- [ ] **Step 5: Commit**

```bash
git add api/routes/generate.py
git commit -m "feat: check credits balance before queuing generate job (402 on insufficient)"
```

---

## Task 6: Worker — deduct credits on job completion

**Files:**
- Modify: `parrot-service/workers/gpu_worker.py`

- [ ] **Step 1: Add `_deduct_credits` helper** — add this function after `_write_pg_fail` (around line 76):

```python
async def _deduct_credits(job: dict):
    """Deduct credits from api_keys table on successful completion."""
    if not settings.DATABASE_URL:
        return
    key_hash = job.get("api_key_hash", "")
    duration = int(job.get("duration", 10))
    job_id = job.get("job_id", "")
    if not key_hash or not job_id:
        return
    cost = duration * settings.CREDITS_PER_SECOND
    try:
        from db import job_store
        ok = await job_store.deduct_credits(key_hash, cost, job_id)
        if not ok:
            logger.warning(f"Credits deduct returned False for {job_id} (balance may be 0)")
    except Exception as e:
        logger.warning(f"Credits deduct failed for {job_id}: {e}")
```

- [ ] **Step 2: Call `_deduct_credits`** — in `process_job`, after `await _write_pg_complete(...)` (around line 152), add:

```python
        await _deduct_credits(job)
```

- [ ] **Step 3: Remove old `increment_usage` call for `completed_jobs`** — find this block near line 154:

```python
        from api.auth import increment_usage
        await increment_usage(job["api_key_hash"], "completed_jobs")
```

Keep it — `increment_usage` still updates the Redis monthly usage counter. The PG `completed_jobs` is now handled inside `deduct_credits`. Both can coexist.

- [ ] **Step 4: Verify parse**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from workers.gpu_worker import process_job; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add workers/gpu_worker.py
git commit -m "feat: deduct credits from PG on job completion"
```

---

## Task 7: Account routes — add `/account/jobs` and update `/account/usage`

**Files:**
- Modify: `parrot-service/api/routes/account.py`

- [ ] **Step 1: Replace `api/routes/account.py` entirely**:

```python
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.auth import get_redis, verify_api_key
from api.models import AccountUsageResponse, AccountJobsResponse, AccountJobItem

router = APIRouter()


@router.get("/account/usage", response_model=AccountUsageResponse)
async def get_usage(key_hash: str = Depends(verify_api_key)):
    r = await get_redis()
    data = await r.hgetall(f"apikey:{key_hash}")
    month = datetime.utcnow().strftime("%Y-%m")

    credits = 0.0
    credits_used = 0.0
    if __import__("api.config", fromlist=["settings"]).settings.DATABASE_URL:
        from db import job_store
        pg = await job_store.get_api_key(key_hash)
        if pg:
            credits = float(pg["credits"])
            credits_used = float(pg["credits_used"])

    return AccountUsageResponse(
        api_key=data.get("name", "unknown"),
        credits=credits,
        credits_used=credits_used,
        total_jobs=int(data.get("total_jobs", 0)),
        completed_jobs=int(data.get("completed_jobs", 0)),
        failed_jobs=int(data.get("failed_jobs", 0)),
        month=month,
    )


@router.get("/account/jobs", response_model=AccountJobsResponse)
async def get_my_jobs(
    key_hash: str = Depends(verify_api_key),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
):
    from db import job_store
    jobs, total = await job_store.get_key_jobs(key_hash, page, limit, status)

    items = [
        AccountJobItem(
            job_id=j["job_id"],
            position=j["position"],
            duration=int(j["duration"]),
            status=j["status"],
            credits_charged=float(j["credits_charged"] or 0),
            video_url=j.get("video_url") or None,
            created_at=float(j["created_at"]),
            completed_at=float(j["completed_at"]) if j.get("completed_at") else None,
            prompt=j.get("prompt", ""),
            seed=int(j.get("seed", 0)),
            callback_url=j.get("callback_url", ""),
        )
        for j in jobs
    ]
    return AccountJobsResponse(jobs=items, total=total, page=page, limit=limit)
```

- [ ] **Step 2: Verify parse**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from api.routes.account import router; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/routes/account.py
git commit -m "feat: add GET /v1/account/jobs; update /account/usage to include credits"
```

---

## Task 8: Admin routes — expand endpoints + auth protection

**Files:**
- Modify: `parrot-service/api/routes/admin.py`

- [ ] **Step 1: Replace `api/routes/admin.py` entirely**:

```python
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import create_api_key, get_redis, require_admin
from api.models import (
    AdminJobItem, AdminJobsResponse, AdminKeyItem,
    CreateKeyRequest, DisableKeyRequest, TopUpRequest,
)

router = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_admin)])


@router.post("/keys")
async def create_key(req: CreateKeyRequest):
    key = await create_api_key(req.name, req.credits)
    return {"name": req.name, "api_key": key, "credits": req.credits}


@router.get("/keys")
async def list_keys():
    from db import job_store
    keys = await job_store.list_api_keys()
    return {
        "keys": [
            AdminKeyItem(
                key_hash=k["key_hash"],
                name=k["name"],
                created_at=float(k["created_at"]),
                disabled=bool(k["disabled"]),
                credits=float(k["credits"]),
                credits_used=float(k["credits_used"]),
                total_jobs=int(k["total_jobs"]),
                completed_jobs=int(k["completed_jobs"]),
                failed_jobs=int(k["failed_jobs"]),
            )
            for k in keys
        ]
    }


@router.patch("/keys/{key_hash}/topup")
async def topup_credits(key_hash: str, req: TopUpRequest):
    if req.add_credits <= 0:
        raise HTTPException(status_code=400, detail="add_credits must be positive")
    from db import job_store
    await job_store.update_api_key(key_hash, add_credits=req.add_credits)
    # Invalidate Redis cache
    r = await get_redis()
    await r.expire(f"apikey:{key_hash}", 1)
    return {"ok": True, "added": req.add_credits}


@router.patch("/keys/{key_hash}/disable")
async def set_key_disabled(key_hash: str, req: DisableKeyRequest):
    from db import job_store
    await job_store.update_api_key(key_hash, disabled=req.disabled)
    r = await get_redis()
    await r.expire(f"apikey:{key_hash}", 1)
    return {"ok": True, "disabled": req.disabled}


@router.get("/jobs", response_model=AdminJobsResponse)
async def list_jobs(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    key_hash: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
):
    from db import job_store
    jobs, total = await job_store.get_jobs_paginated(page, limit, key_hash, status)
    items = [
        AdminJobItem(
            job_id=j["job_id"],
            key_name=j.get("key_name"),
            api_key_hash=j["api_key_hash"],
            position=j["position"],
            duration=int(j["duration"]),
            status=j["status"],
            credits_charged=float(j.get("credits_charged") or 0),
            video_url=j.get("video_url") or None,
            created_at=float(j["created_at"]),
            completed_at=float(j["completed_at"]) if j.get("completed_at") else None,
            prompt=j.get("prompt", ""),
            seed=int(j.get("seed", 0)),
            callback_url=j.get("callback_url", ""),
        )
        for j in jobs
    ]
    return AdminJobsResponse(jobs=items, total=total, page=page, limit=limit)


@router.get("/billing")
async def billing_report(month: Optional[str] = None):
    r = await get_redis()
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")

    from db import job_store
    keys = await job_store.list_api_keys()
    report = []
    for k in keys:
        kh = k["key_hash"]
        usage = await r.hgetall(f"apikey:{kh}:usage:{month}")
        count = int(usage.get("count", 0))
        report.append({
            "name": k["name"],
            "key_hash_prefix": kh[:12],
            "month": month,
            "jobs_this_month": count,
            "total_jobs": int(k["total_jobs"]),
            "completed_jobs": int(k["completed_jobs"]),
            "failed_jobs": int(k["failed_jobs"]),
            "credits": float(k["credits"]),
            "credits_used": float(k["credits_used"]),
            "status": "disabled" if k["disabled"] else "active",
        })

    return {
        "month": month,
        "total_clients": len(report),
        "total_jobs_this_month": sum(r_["jobs_this_month"] for r_ in report),
        "total_credits_used": sum(r_["credits_used"] for r_ in report),
        "clients": report,
    }


@router.post("/cleanup")
async def cleanup_expired_jobs(max_age_hours: int = 24):
    r = await get_redis()
    import time as _time
    cutoff = _time.time() - (max_age_hours * 3600)
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
        if cursor == 0:
            break
    return {"cleaned_jobs": cleaned, "cutoff_hours": max_age_hours}
```

- [ ] **Step 2: Verify parse**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from api.routes.admin import router; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/routes/admin.py
git commit -m "feat: expand admin routes with auth, key management, jobs list, topup"
```

---

## Task 9: Admin HTML — create `frontend/admin.html`

**Files:**
- Create: `parrot-service/frontend/admin.html`

- [ ] **Step 1: Create `frontend/admin.html`** with the full Admin UI:

```html
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Piigu Admin</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #07090f; --surface: #0d1117; --panel: #111827;
      --border: #1e2738; --border2: #263248;
      --accent: #6366f1; --accent2: #818cf8; --glow: rgba(99,102,241,.18);
      --green: #10b981; --red: #f43f5e; --amber: #f59e0b; --sky: #38bdf8;
      --txt1: #f1f5f9; --txt2: #94a3b8; --txt3: #475569; --radius: 12px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--txt1); font-family: 'Inter', sans-serif; font-size: 14px; min-height: 100vh; }

    /* Login */
    #login-screen { display: flex; align-items: center; justify-content: center; min-height: 100vh; }
    .login-box { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 40px; width: 360px; }
    .login-box h1 { font-size: 22px; font-weight: 600; margin-bottom: 8px; }
    .login-box p { color: var(--txt2); margin-bottom: 24px; }
    .login-box input { width: 100%; padding: 10px 14px; background: var(--panel); border: 1px solid var(--border2); border-radius: 8px; color: var(--txt1); font-size: 14px; margin-bottom: 12px; outline: none; }
    .login-box input:focus { border-color: var(--accent); }
    .btn { display: inline-flex; align-items: center; gap: 6px; padding: 9px 18px; border-radius: 8px; border: none; cursor: pointer; font-size: 13px; font-weight: 500; transition: opacity .15s; }
    .btn:hover { opacity: .85; }
    .btn-primary { background: var(--accent); color: #fff; width: 100%; justify-content: center; }
    .btn-sm { padding: 5px 12px; font-size: 12px; }
    .btn-outline { background: transparent; border: 1px solid var(--border2); color: var(--txt2); }
    .btn-danger { background: var(--red); color: #fff; }
    .btn-green { background: var(--green); color: #fff; }
    #login-error { color: var(--red); font-size: 13px; margin-top: 8px; display: none; }

    /* App shell */
    #app { display: none; flex-direction: column; min-height: 100vh; }
    .topbar { height: 56px; border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 24px; gap: 12px; background: rgba(13,17,23,.9); backdrop-filter: blur(12px); position: sticky; top: 0; z-index: 100; }
    .topbar-brand { font-weight: 700; font-size: 16px; letter-spacing: -.3px; }
    .topbar-brand span { color: var(--accent2); }
    .topbar-spacer { flex: 1; }
    .logout-btn { background: transparent; border: 1px solid var(--border2); color: var(--txt2); padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }

    /* Tabs */
    .tabs { display: flex; gap: 4px; padding: 16px 24px 0; border-bottom: 1px solid var(--border); }
    .tab { padding: 10px 18px; border-radius: 8px 8px 0 0; cursor: pointer; font-size: 13px; font-weight: 500; color: var(--txt2); border: 1px solid transparent; border-bottom: none; transition: all .15s; }
    .tab.active { background: var(--surface); color: var(--txt1); border-color: var(--border); }
    .tab:hover:not(.active) { color: var(--txt1); }

    /* Content */
    .content { padding: 24px; flex: 1; }
    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
    .panel-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--border); }
    .panel-title { font-weight: 600; font-size: 15px; }

    /* Table */
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; padding: 10px 16px; font-size: 12px; font-weight: 500; color: var(--txt2); text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--border); background: var(--panel); }
    td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 13px; vertical-align: top; }
    tr:last-child td { border-bottom: none; }
    tr.expandable { cursor: pointer; }
    tr.expandable:hover td { background: rgba(255,255,255,.02); }

    /* Badges */
    .badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    .badge-green { background: rgba(16,185,129,.15); color: var(--green); }
    .badge-red { background: rgba(244,63,94,.15); color: var(--red); }
    .badge-amber { background: rgba(245,158,11,.15); color: var(--amber); }
    .badge-sky { background: rgba(56,189,248,.15); color: var(--sky); }
    .badge-gray { background: rgba(148,163,184,.1); color: var(--txt2); }

    /* Modal */
    .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.6); display: flex; align-items: center; justify-content: center; z-index: 200; display: none; }
    .modal-overlay.open { display: flex; }
    .modal { background: var(--surface); border: 1px solid var(--border2); border-radius: var(--radius); padding: 28px; width: 400px; }
    .modal h2 { font-size: 17px; font-weight: 600; margin-bottom: 20px; }
    .form-field { margin-bottom: 14px; }
    .form-field label { display: block; font-size: 12px; font-weight: 500; color: var(--txt2); margin-bottom: 6px; }
    .form-field input { width: 100%; padding: 9px 13px; background: var(--panel); border: 1px solid var(--border2); border-radius: 8px; color: var(--txt1); font-size: 14px; outline: none; }
    .form-field input:focus { border-color: var(--accent); }
    .modal-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px; }

    /* Detail row */
    .detail-row td { background: var(--panel); padding: 16px 20px; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
    .detail-item label { display: block; font-size: 11px; color: var(--txt2); margin-bottom: 3px; }
    .detail-item value { font-size: 13px; word-break: break-all; }
    video { width: 100%; max-height: 360px; border-radius: 8px; background: #000; margin-top: 10px; }

    /* Stats cards */
    .stat-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 20px; }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; }
    .stat-card .label { font-size: 12px; color: var(--txt2); margin-bottom: 6px; }
    .stat-card .value { font-size: 24px; font-weight: 700; }

    /* Pagination */
    .pagination { display: flex; gap: 8px; align-items: center; justify-content: flex-end; padding: 14px 20px; border-top: 1px solid var(--border); }
    .page-info { color: var(--txt2); font-size: 13px; }

    /* Filter row */
    .filter-row { display: flex; gap: 10px; padding: 14px 20px; border-bottom: 1px solid var(--border); align-items: center; }
    .filter-row select, .filter-row input { padding: 7px 12px; background: var(--panel); border: 1px solid var(--border2); border-radius: 8px; color: var(--txt1); font-size: 13px; outline: none; }

    .empty { text-align: center; padding: 48px; color: var(--txt2); }
    .loading { text-align: center; padding: 48px; color: var(--txt3); }
    .toast { position: fixed; bottom: 24px; right: 24px; background: var(--panel); border: 1px solid var(--border2); border-radius: 8px; padding: 12px 18px; font-size: 13px; z-index: 300; opacity: 0; transition: opacity .2s; pointer-events: none; }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>

<!-- Login -->
<div id="login-screen">
  <div class="login-box">
    <h1>Piigu Admin</h1>
    <p>請輸入管理員密碼</p>
    <input type="password" id="pw-input" placeholder="Admin password" />
    <button class="btn btn-primary" onclick="doLogin()">登入</button>
    <div id="login-error">密碼錯誤，請重試</div>
  </div>
</div>

<!-- App -->
<div id="app">
  <div class="topbar">
    <div class="topbar-brand">Piigu <span>Admin</span></div>
    <div class="topbar-spacer"></div>
    <button class="logout-btn" onclick="doLogout()">登出</button>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('keys')">API Keys</div>
    <div class="tab" onclick="switchTab('jobs')">Jobs</div>
    <div class="tab" onclick="switchTab('billing')">Billing</div>
  </div>

  <div class="content">
    <!-- Keys Tab -->
    <div id="tab-keys">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">API Keys</div>
          <button class="btn btn-primary btn-sm" onclick="openNewKeyModal()">+ 新建 Key</button>
        </div>
        <div id="keys-table-wrap"><div class="loading">載入中...</div></div>
      </div>
    </div>

    <!-- Jobs Tab -->
    <div id="tab-jobs" style="display:none">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Generation Jobs</div>
        </div>
        <div class="filter-row">
          <select id="jobs-key-filter" onchange="loadJobs(1)">
            <option value="">所有 Keys</option>
          </select>
          <select id="jobs-status-filter" onchange="loadJobs(1)">
            <option value="">所有狀態</option>
            <option value="queued">Queued</option>
            <option value="processing">Processing</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>
        <div id="jobs-table-wrap"><div class="loading">載入中...</div></div>
        <div class="pagination" id="jobs-pagination"></div>
      </div>
    </div>

    <!-- Billing Tab -->
    <div id="tab-billing" style="display:none">
      <div class="filter-row" style="border:none;padding-bottom:0">
        <input type="month" id="billing-month" onchange="loadBilling()" />
      </div>
      <div id="billing-content"><div class="loading">載入中...</div></div>
    </div>
  </div>
</div>

<!-- New Key Modal -->
<div class="modal-overlay" id="modal-new-key">
  <div class="modal">
    <h2>新建 API Key</h2>
    <div class="form-field">
      <label>名稱 / 客戶名稱</label>
      <input type="text" id="new-key-name" placeholder="e.g. Client A" />
    </div>
    <div class="form-field">
      <label>初始 Credits</label>
      <input type="number" id="new-key-credits" value="100" min="0" step="0.01" />
    </div>
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closeModal('modal-new-key')">取消</button>
      <button class="btn btn-primary" onclick="createKey()">建立</button>
    </div>
  </div>
</div>

<!-- Topup Modal -->
<div class="modal-overlay" id="modal-topup">
  <div class="modal">
    <h2>充值 Credits</h2>
    <div class="form-field">
      <label>充值金額</label>
      <input type="number" id="topup-amount" value="100" min="0.01" step="0.01" />
    </div>
    <input type="hidden" id="topup-key-hash" />
    <div class="modal-actions">
      <button class="btn btn-outline" onclick="closeModal('modal-topup')">取消</button>
      <button class="btn btn-green" onclick="doTopup()">確認充值</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const API = '';
let adminToken = localStorage.getItem('adminToken') || '';
let allKeys = [];
let jobsPage = 1;

// ── Auth ──────────────────────────────────────────────────────
function doLogin() {
  adminToken = document.getElementById('pw-input').value.trim();
  apiFetch('/v1/admin/keys').then(() => {
    localStorage.setItem('adminToken', adminToken);
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('app').style.display = 'flex';
    document.getElementById('app').style.flexDirection = 'column';
    initApp();
  }).catch(() => {
    document.getElementById('login-error').style.display = 'block';
  });
}
document.getElementById('pw-input').addEventListener('keydown', e => { if(e.key==='Enter') doLogin(); });

function doLogout() {
  localStorage.removeItem('adminToken');
  location.reload();
}

function apiFetch(path, opts={}) {
  return fetch(API + path, {
    ...opts,
    headers: { 'X-Admin-Password': adminToken, 'Content-Type': 'application/json', ...(opts.headers||{}) }
  }).then(async r => {
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  });
}

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2500);
}

// ── Init ──────────────────────────────────────────────────────
function initApp() {
  if (adminToken) {
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('app').style.display = 'flex';
    document.getElementById('app').style.flexDirection = 'column';
  }
  loadKeys();
  // Set billing month to current
  const now = new Date();
  document.getElementById('billing-month').value = now.toISOString().slice(0,7);
}

if (adminToken) initApp();

// ── Tabs ──────────────────────────────────────────────────────
function switchTab(name) {
  ['keys','jobs','billing'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t===name ? '' : 'none';
    document.querySelectorAll('.tab')[['keys','jobs','billing'].indexOf(t)].classList.toggle('active', t===name);
  });
  if (name==='keys') loadKeys();
  if (name==='jobs') loadJobs(1);
  if (name==='billing') loadBilling();
}

// ── Keys Tab ──────────────────────────────────────────────────
function loadKeys() {
  apiFetch('/v1/admin/keys').then(data => {
    allKeys = data.keys || [];
    renderKeysTable(allKeys);
    // Populate jobs filter
    const sel = document.getElementById('jobs-key-filter');
    sel.innerHTML = '<option value="">所有 Keys</option>';
    allKeys.forEach(k => {
      sel.innerHTML += `<option value="${k.key_hash}">${k.name}</option>`;
    });
  }).catch(e => {
    document.getElementById('keys-table-wrap').innerHTML = `<div class="empty">載入失敗: ${e.message}</div>`;
  });
}

function renderKeysTable(keys) {
  if (!keys.length) { document.getElementById('keys-table-wrap').innerHTML = '<div class="empty">尚無 API Keys</div>'; return; }
  let html = `<table><thead><tr>
    <th>名稱</th><th>Credits 餘額</th><th>已用 Credits</th><th>Total Jobs</th><th>完成</th><th>失敗</th><th>狀態</th><th>操作</th>
  </tr></thead><tbody>`;
  keys.forEach(k => {
    const status = k.disabled ? `<span class="badge badge-red">停用</span>` : `<span class="badge badge-green">啟用</span>`;
    html += `<tr>
      <td><strong>${k.name}</strong><br><small style="color:var(--txt3)">${k.key_hash.slice(0,12)}...</small></td>
      <td style="font-weight:600;color:var(--accent2)">${Number(k.credits).toFixed(4)}</td>
      <td>${Number(k.credits_used).toFixed(4)}</td>
      <td>${k.total_jobs}</td>
      <td>${k.completed_jobs}</td>
      <td>${k.failed_jobs}</td>
      <td>${status}</td>
      <td>
        <button class="btn btn-sm btn-green" onclick="openTopup('${k.key_hash}')">充值</button>
        <button class="btn btn-sm ${k.disabled?'btn-primary':'btn-danger'}" style="margin-left:6px" onclick="toggleDisable('${k.key_hash}',${!k.disabled})">${k.disabled?'啟用':'停用'}</button>
      </td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('keys-table-wrap').innerHTML = html;
}

function openNewKeyModal() { document.getElementById('modal-new-key').classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }

function createKey() {
  const name = document.getElementById('new-key-name').value.trim();
  const credits = parseFloat(document.getElementById('new-key-credits').value) || 0;
  if (!name) return;
  apiFetch('/v1/admin/keys', { method: 'POST', body: JSON.stringify({name, credits}) }).then(data => {
    closeModal('modal-new-key');
    toast(`Key 已建立: ${data.api_key}`);
    navigator.clipboard?.writeText(data.api_key);
    loadKeys();
  }).catch(e => toast('建立失敗: ' + e.message));
}

function openTopup(keyHash) {
  document.getElementById('topup-key-hash').value = keyHash;
  document.getElementById('modal-topup').classList.add('open');
}

function doTopup() {
  const keyHash = document.getElementById('topup-key-hash').value;
  const amount = parseFloat(document.getElementById('topup-amount').value);
  apiFetch(`/v1/admin/keys/${keyHash}/topup`, { method: 'PATCH', body: JSON.stringify({add_credits: amount}) }).then(() => {
    closeModal('modal-topup');
    toast(`充值 ${amount} credits 成功`);
    loadKeys();
  }).catch(e => toast('充值失敗: ' + e.message));
}

function toggleDisable(keyHash, disabled) {
  apiFetch(`/v1/admin/keys/${keyHash}/disable`, { method: 'PATCH', body: JSON.stringify({disabled}) }).then(() => {
    toast(disabled ? 'Key 已停用' : 'Key 已啟用');
    loadKeys();
  }).catch(e => toast('操作失敗: ' + e.message));
}

// ── Jobs Tab ──────────────────────────────────────────────────
let expandedJobId = null;

function loadJobs(page) {
  jobsPage = page || jobsPage;
  const keyHash = document.getElementById('jobs-key-filter').value;
  const status = document.getElementById('jobs-status-filter').value;
  let url = `/v1/admin/jobs?page=${jobsPage}&limit=20`;
  if (keyHash) url += `&key_hash=${keyHash}`;
  if (status) url += `&status=${status}`;
  document.getElementById('jobs-table-wrap').innerHTML = '<div class="loading">載入中...</div>';
  apiFetch(url).then(data => renderJobsTable(data)).catch(e => {
    document.getElementById('jobs-table-wrap').innerHTML = `<div class="empty">載入失敗: ${e.message}</div>`;
  });
}

function statusBadge(s) {
  const map = { queued:'badge-amber', processing:'badge-sky', postprocessing:'badge-sky', completed:'badge-green', failed:'badge-red', cancelled:'badge-gray' };
  return `<span class="badge ${map[s]||'badge-gray'}">${s}</span>`;
}

function fmtTime(ts) {
  if (!ts) return '—';
  return new Date(ts*1000).toLocaleString('zh-TW');
}

function renderJobsTable(data) {
  const jobs = data.jobs || [];
  if (!jobs.length) { document.getElementById('jobs-table-wrap').innerHTML = '<div class="empty">沒有 Jobs</div>'; return; }

  let html = `<table><thead><tr>
    <th>Job ID</th><th>Key</th><th>Position</th><th>Duration</th><th>Credits</th><th>狀態</th><th>建立時間</th>
  </tr></thead><tbody>`;

  jobs.forEach(j => {
    html += `<tr class="expandable" onclick="toggleJobDetail('${j.job_id}')">
      <td style="font-family:monospace;font-size:12px">${j.job_id}</td>
      <td>${j.key_name || j.api_key_hash.slice(0,8)+'...'}</td>
      <td>${j.position}</td>
      <td>${j.duration}s</td>
      <td>${Number(j.credits_charged).toFixed(4)}</td>
      <td>${statusBadge(j.status)}</td>
      <td>${fmtTime(j.created_at)}</td>
    </tr>`;
    // Detail row (hidden by default)
    html += `<tr class="detail-row" id="detail-${j.job_id}" style="display:none"><td colspan="7">
      <div class="detail-grid">
        <div class="detail-item"><label>Prompt</label><value>${j.prompt || '—'}</value></div>
        <div class="detail-item"><label>Seed</label><value>${j.seed}</value></div>
        <div class="detail-item"><label>完成時間</label><value>${fmtTime(j.completed_at)}</value></div>
        <div class="detail-item"><label>Callback URL</label><value>${j.callback_url || '—'}</value></div>
      </div>
      ${j.video_url ? `<video controls src="${j.video_url}"></video>` : '<div style="color:var(--txt3);font-size:13px">尚無影片</div>'}
    </td></tr>`;
  });

  html += '</tbody></table>';
  document.getElementById('jobs-table-wrap').innerHTML = html;

  // Pagination
  const total = data.total;
  const pages = Math.ceil(total / 20);
  let pg = `<span class="page-info">共 ${total} 筆</span>`;
  if (jobsPage > 1) pg += `<button class="btn btn-outline btn-sm" onclick="loadJobs(${jobsPage-1})">上一頁</button>`;
  pg += `<span class="page-info">${jobsPage} / ${pages}</span>`;
  if (jobsPage < pages) pg += `<button class="btn btn-outline btn-sm" onclick="loadJobs(${jobsPage+1})">下一頁</button>`;
  document.getElementById('jobs-pagination').innerHTML = pg;
}

function toggleJobDetail(jobId) {
  const row = document.getElementById('detail-'+jobId);
  if (!row) return;
  if (expandedJobId && expandedJobId !== jobId) {
    const prev = document.getElementById('detail-'+expandedJobId);
    if (prev) prev.style.display = 'none';
  }
  const isOpen = row.style.display !== 'none';
  row.style.display = isOpen ? 'none' : '';
  expandedJobId = isOpen ? null : jobId;
}

// ── Billing Tab ───────────────────────────────────────────────
function loadBilling() {
  const month = document.getElementById('billing-month').value;
  apiFetch(`/v1/admin/billing?month=${month}`).then(data => renderBilling(data)).catch(e => {
    document.getElementById('billing-content').innerHTML = `<div class="empty">載入失敗: ${e.message}</div>`;
  });
}

function renderBilling(data) {
  let html = `<div class="stat-cards">
    <div class="stat-card"><div class="label">Total Jobs</div><div class="value">${data.total_jobs_this_month}</div></div>
    <div class="stat-card"><div class="label">Credits Used</div><div class="value">${Number(data.total_credits_used).toFixed(4)}</div></div>
    <div class="stat-card"><div class="label">Active Clients</div><div class="value">${data.clients.filter(c=>c.status==='active').length}</div></div>
  </div>
  <div class="panel"><table><thead><tr>
    <th>名稱</th><th>本月 Jobs</th><th>Credits 餘額</th><th>累計已用</th><th>完成</th><th>失敗</th><th>狀態</th>
  </tr></thead><tbody>`;

  data.clients.forEach(c => {
    const st = c.status==='active' ? '<span class="badge badge-green">啟用</span>' : '<span class="badge badge-red">停用</span>';
    html += `<tr>
      <td><strong>${c.name}</strong><br><small style="color:var(--txt3)">${c.key_hash_prefix}...</small></td>
      <td>${c.jobs_this_month}</td>
      <td style="color:var(--accent2);font-weight:600">${Number(c.credits).toFixed(4)}</td>
      <td>${Number(c.credits_used).toFixed(4)}</td>
      <td>${c.completed_jobs}</td>
      <td>${c.failed_jobs}</td>
      <td>${st}</td>
    </tr>`;
  });
  html += '</tbody></table></div>';
  document.getElementById('billing-content').innerHTML = html;
}
</script>
</body>
</html>
```

- [ ] **Step 2: Verify file exists**

```bash
ls -la "/Users/welly/Parrot API/parrot-service/frontend/admin.html"
```

Expected: file shown with non-zero size

- [ ] **Step 3: Commit**

```bash
cd "/Users/welly/Parrot API/parrot-service"
git add frontend/admin.html
git commit -m "feat: add admin web UI with login, Keys/Jobs/Billing tabs"
```

---

## Task 10: Mount `/admin` route in `api/main.py`

**Files:**
- Modify: `parrot-service/api/main.py`

- [ ] **Step 1: Add the `/admin` route** — at the end of `api/main.py`, after the `/health` endpoint, add:

```python

@app.get("/admin", include_in_schema=False)
async def admin_ui():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "admin.html")
    return FileResponse(path, media_type="text/html")
```

`FileResponse` and `os` are already imported at the top of main.py.

- [ ] **Step 2: Verify parse**

```bash
cd "/Users/welly/Parrot API/parrot-service"
python -c "from api.main import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Full smoke test**

```bash
ADMIN_PASSWORD=test123 DATABASE_URL="postgresql://..." REDIS_URL="redis://localhost:6379" \
  uvicorn api.main:app --port 8000 --reload
```

Then open `http://localhost:8000/admin` — should see login screen.

Try logging in with `test123` → should load Keys tab.

Try creating a key: name=`test`, credits=`50` → should see new key in table, and copy the `pk_...` key from toast.

Try submitting a generate with 0-credit key:
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "X-API-Key: pk_..." \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/img.jpg","position":"cowgirl","duration":10}'
```
Expected: `HTTP 402` with `{"detail":{"error":"Insufficient credits","required":0.35,"available":0.0}}`

Try with a key that has 50 credits → should get `HTTP 200` with job_id.

- [ ] **Step 4: Commit**

```bash
git add api/main.py
git commit -m "feat: mount /admin route to serve admin UI"
```

---

## Task 11: Environment variable setup

- [ ] **Step 1: Add to `.env.example`** — open `parrot-service/.env.example` and add:

```bash
# Admin
ADMIN_PASSWORD=change-me-in-production

# Credits
CREDITS_PER_SECOND=0.035
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API/parrot-service"
git add .env.example
git commit -m "chore: add ADMIN_PASSWORD and CREDITS_PER_SECOND to .env.example"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Credits at 0.035/sec → `config.CREDITS_PER_SECOND`, `cost = duration * settings.CREDITS_PER_SECOND`
- ✅ Credits in PostgreSQL → `api_keys` table in `db/job_store.py`
- ✅ PG+Redis cache auth → Task 3 `verify_api_key`
- ✅ Balance check at submit, deduct at completion → Tasks 5 + 6
- ✅ 402 with required/available → Task 5
- ✅ No deduct on failure → `deduct_credits` only called in completion path
- ✅ Admin password → `require_admin` dependency on all admin routes
- ✅ Create key with credits → `POST /v1/admin/keys` with `CreateKeyRequest`
- ✅ List keys → `GET /v1/admin/keys`
- ✅ Top up credits → `PATCH /v1/admin/keys/{hash}/topup`
- ✅ Disable/enable → `PATCH /v1/admin/keys/{hash}/disable`
- ✅ Admin jobs list paginated → `GET /v1/admin/jobs`
- ✅ User job history → `GET /v1/account/jobs`
- ✅ Admin UI with 4 tabs (Keys/Jobs/Billing/Login) → Task 9
- ✅ Video playback in job detail → `<video>` tag in detail row
- ✅ `/admin` route → Task 10
- ✅ `.env.example` updated → Task 11

**Type consistency:**
- `deduct_credits(key_hash, amount, job_id)` defined in Task 1, called in Task 6 ✅
- `get_jobs_paginated` returns `tuple[list[dict], int]`, consumed in Task 8 ✅
- `get_key_jobs` returns `tuple[list[dict], int]`, consumed in Task 7 ✅
- `AdminKeyItem`, `CreateKeyRequest`, `TopUpRequest`, `DisableKeyRequest` defined in Task 4, used in Task 8 ✅
- `AccountJobsResponse`, `AccountJobItem`, `AccountUsageResponse` defined in Task 4, used in Task 7 ✅
