from __future__ import annotations

import ipaddress
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import settings
from api.metrics import router as metrics_router
from api.routes import account, admin, generate, jobs, positions

# https://www.cloudflare.com/ips/
CLOUDFLARE_IP_RANGES = [
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
    "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
    "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
    "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32",
    "2405:b500::/32", "2405:8100::/32", "2a06:98c0::/29", "2c0f:f248::/32",
]
CF_NETWORKS = [ipaddress.ip_network(cidr) for cidr in CLOUDFLARE_IP_RANGES]
CLOUDFLARE_ENABLED = os.getenv("CLOUDFLARE_ENABLED", "false").lower() == "true"


class CloudflareMiddleware(BaseHTTPMiddleware):
    """Reject requests not coming from Cloudflare edge when CF mode is on."""

    async def dispatch(self, request: Request, call_next):
        if not CLOUDFLARE_ENABLED:
            return await call_next(request)

        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        client_ip = request.client.host
        try:
            addr = ipaddress.ip_address(client_ip)
            if not any(addr in net for net in CF_NETWORKS):
                return JSONResponse(status_code=403, content={"error": "Direct access forbidden"})
        except ValueError:
            pass

        return await call_next(request)


async def _queue_reaper(interval: int = 30):
    """Background task: expire jobs that have been queued too long."""
    import asyncio
    import time
    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    while True:
        try:
            await asyncio.sleep(interval)
            queue_items = await r.lrange("job_queue", 0, -1)
            now = time.time()
            expired = 0
            for job_id in queue_items:
                data = await r.hgetall(f"job:{job_id}")
                if not data:
                    await r.lrem("job_queue", 1, job_id)
                    continue
                created = float(data.get("created_at", 0))
                if now - created > settings.QUEUE_EXPIRE_SECONDS:
                    await r.lrem("job_queue", 1, job_id)
                    await r.hset(f"job:{job_id}", mapping={
                        "status": "failed",
                        "error": f"Queue timeout: waited {now - created:.0f}s",
                        "completed_at": str(now),
                    })
                    expired += 1
            if expired:
                print(f"[Reaper] Expired {expired} jobs from queue")
        except Exception as e:
            print(f"[Reaper] Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        await r.ping()
        print(f"[API] Redis connected: {settings.REDIS_URL}")
    except Exception as e:
        print(f"[API] WARNING: Redis not reachable: {e}")
    finally:
        await r.aclose()

    mode = "Cloudflare proxy" if CLOUDFLARE_ENABLED else "direct"
    print(f"[API] Running in {mode} mode")
    print(f"[API] Queue expire: {settings.QUEUE_EXPIRE_SECONDS}s")

    reaper_task = asyncio.create_task(_queue_reaper())
    yield
    reaper_task.cancel()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(CloudflareMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, prefix="/v1", tags=["generate"])
app.include_router(jobs.router, prefix="/v1", tags=["jobs"])
app.include_router(positions.router, prefix="/v1", tags=["positions"])
app.include_router(account.router, prefix="/v1", tags=["account"])
app.include_router(admin.router, tags=["admin"])
app.include_router(metrics_router, tags=["metrics"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}
