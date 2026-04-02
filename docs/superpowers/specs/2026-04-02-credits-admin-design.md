# Credits System & Admin Dashboard — Design Spec

**Date:** 2026-04-02  
**Scope:** `parrot-service/` only (parrot-api on-premise excluded)  
**Status:** Approved

---

## 1. Overview

Add a per-key credits system and a web-based Admin dashboard to parrot-service. Every API key has a credit balance stored in PostgreSQL. Each successful video generation deducts credits based on duration (0.035 credits/second). Admins can manage keys and view all generation activity through a password-protected web UI.

---

## 2. Database Schema

### New table: `api_keys`

```sql
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
```

### Modified table: `jobs`

```sql
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS credits_charged NUMERIC(12,4) DEFAULT 0;
```

`credits_charged` is written by the worker on job completion. It is 0 for failed/cancelled jobs.

### Credits deduction (atomic, prevents overdraft)

```sql
UPDATE api_keys
SET credits = credits - $1,
    credits_used = credits_used + $1,
    completed_jobs = completed_jobs + 1
WHERE key_hash = $2
  AND credits >= $1
RETURNING credits;
```

Zero rows affected = insufficient credits at deduction time (edge case: worker still charges correctly).

---

## 3. Credits Billing Logic

- **Rate:** 0.035 credits per second
- **Calculation:** `cost = duration * 0.035` (e.g. 10s = 0.35 credits, 5s = 0.175 credits)
- **Timing — two-phase approach:**
  1. **At submit:** Check `credits >= cost`. If not, reject with `402`. Do NOT deduct.
  2. **At job completion (worker):** Deduct credits atomically in PG, write `credits_charged` to jobs table.
  3. **At job failure:** No deduction. `credits_charged` remains 0.
- **Concurrency:** The PG atomic UPDATE prevents overdraft even with parallel submissions.

---

## 4. Authentication Changes

### API Key verification (`api/auth.py`)

Updated flow:
1. Extract key from `X-API-Key` header
2. Check Redis cache `apikey:{hash}` (TTL 60s) — if hit, use cached data
3. On cache miss: query PG `api_keys` table, write result to Redis
4. Reject if `disabled = true` or key not found
5. Return `key_hash` and key metadata (including credits) to caller

### Admin authentication (`api/routes/admin.py`)

- New FastAPI dependency `require_admin` checks `X-Admin-Password` header against `settings.ADMIN_PASSWORD`
- All `/v1/admin/*` routes protected by this dependency
- Returns `401` if header missing, `403` if password wrong

### New config (`api/config.py`)

```python
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")
CREDITS_PER_SECOND: float = 0.035
```

---

## 5. API Changes

### Modified: `POST /v1/generate`

Before queuing:
1. Calculate `cost = req.duration * settings.CREDITS_PER_SECOND`
2. Query PG for current credits balance
3. If `credits < cost` → `402 Payment Required`:
   ```json
   { "error": "Insufficient credits", "required": 0.35, "available": 0.12 }
   ```
4. Otherwise enqueue job normally (no deduction yet)

### New: `GET /v1/account/jobs`

Auth: `X-API-Key` (existing user key)

Query params: `page` (default 1), `limit` (default 20, max 100), `status` (optional filter)

Response:
```json
{
  "jobs": [
    {
      "job_id": "...",
      "position": "cowgirl",
      "duration": 10,
      "status": "completed",
      "credits_charged": 0.35,
      "video_url": "https://...",
      "created_at": 1234567890.0,
      "completed_at": 1234567950.0
    }
  ],
  "total": 42,
  "page": 1,
  "limit": 20
}
```

### Modified: `GET /v1/account`

Add `credits` and `credits_used` fields to existing response.

### Admin Endpoints (all require `X-Admin-Password`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/admin/keys` | Create key with `name` + `credits` |
| `GET` | `/v1/admin/keys` | List all keys with balances |
| `PATCH` | `/v1/admin/keys/{key_hash}` | Top up credits or disable key |
| `GET` | `/v1/admin/jobs` | Paginated job list, filterable by `key_hash` |
| `GET` | `/v1/admin/billing` | Monthly billing report (existing, now requires auth) |

---

## 6. Worker Changes (`workers/gpu_worker.py`)

On job completion:
```python
cost = job["duration"] * CREDITS_PER_SECOND
await db.deduct_credits(job["api_key_hash"], cost, job["job_id"])
```

On job failure: no credits action.

---

## 7. Admin Web UI (`frontend/admin.html`)

Standalone HTML file served at `/admin`. Style matches existing `frontend/index.html` (dark theme, Inter font, indigo accent).

### Login Screen
- Full-page password prompt
- On submit: stores password in `localStorage` as `adminToken`
- All subsequent API calls include `X-Admin-Password: {adminToken}` header
- On 401/403 response: clear token, redirect to login

### Tab: Keys
- Table: Name | Credits Remaining | Credits Used | Total Jobs | Status | Actions
- Actions: "Top Up" (modal, enter amount) | "Disable/Enable"
- Button: "New Key" → modal with Name + Initial Credits fields
- Shows key hash prefix (first 12 chars) for identification

### Tab: Jobs
- Paginated table: Key Name | Position | Duration | Credits | Status | Created At
- Status badges with colors (queued=amber, processing=blue, completed=green, failed=red)
- Click row → expand detail panel:
  - Full prompt, seed, callback_url
  - Inline `<video>` player if `video_url` is present
- Filter by key (dropdown) and status

### Tab: Billing
- Month picker (YYYY-MM)
- Summary cards: Total Jobs | Total Credits Consumed | Active Keys
- Per-key breakdown table

---

## 8. Files Modified / Created

| File | Change |
|------|--------|
| `db/job_store.py` | Add `api_keys` CRUD, `deduct_credits`, `get_jobs_paginated` |
| `api/auth.py` | Update `verify_api_key` to use PG + Redis cache; add `require_admin` |
| `api/config.py` | Add `ADMIN_PASSWORD`, `CREDITS_PER_SECOND` |
| `api/models.py` | Add request/response models for credits, admin endpoints |
| `api/routes/generate.py` | Add pre-submit credits balance check |
| `api/routes/admin.py` | Expand endpoints + auth middleware |
| `api/routes/account.py` | Add `GET /v1/account/jobs` endpoint |
| `api/main.py` | Mount `/admin` route to serve `admin.html` |
| `frontend/admin.html` | New Admin UI (create from scratch) |

**Not changed:** `parrot-api/`, worker inference logic, R2 storage, webhook sender.

---

## 9. Migration Strategy

On service startup (`db/job_store.py` `get_pool()`):
1. `CREATE TABLE IF NOT EXISTS api_keys ...` — safe no-op if exists
2. `ALTER TABLE jobs ADD COLUMN IF NOT EXISTS credits_charged ...` — safe no-op if exists
3. Existing Redis-only keys continue to work; they simply won't appear in Admin UI until recreated via the new flow. No forced migration of existing keys.

---

## 10. Out of Scope

- Stripe / payment gateway integration
- Automatic credit top-up / subscription billing
- Email notifications for low balance
- parrot-api (on-premise) credits support
- Rate limiting beyond credits check
