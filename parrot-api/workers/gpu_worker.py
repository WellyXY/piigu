"""
GPU Worker — one process per GPU.

Startup:
  1. Load base model into VRAM
  2. Pre-load Fast LoRA (shared across all positions)
  3. Enter main loop: BRPOP from Redis queue, process jobs

Usage:
  python -m workers.gpu_worker --gpu_id 0
  python -m workers.gpu_worker --gpu_id 1
  ...
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time

import redis.asyncio as aioredis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from api.models import JobStatus
from task_queue.job_manager import get_job, pop_job, update_job
from storage.file_storage import save_video
from webhook.sender import send_webhook
from workers.inference_engine import InferenceEngine
from workers.postprocess import run_postprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GPU%(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SHUTDOWN = False


def handle_signal(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received, finishing current job...")
    SHUTDOWN = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


async def process_job(engine: InferenceEngine, r: aioredis.Redis, job_id: str):
    job = await get_job(r, job_id)
    if not job:
        logger.warning(f" {engine.gpu_id}] Job {job_id} not found, skipping")
        return

    wait_time = time.time() - float(job.get("created_at", 0))
    if wait_time > settings.QUEUE_EXPIRE_SECONDS:
        logger.info(f" {engine.gpu_id}] Job {job_id} expired (waited {wait_time:.0f}s > {settings.QUEUE_EXPIRE_SECONDS}s), killing")
        await update_job(r, job_id,
                         status=JobStatus.FAILED.value,
                         error=f"Queue timeout: waited {wait_time:.0f}s",
                         completed_at=time.time())
        callback_url = job.get("callback_url", "")
        if callback_url:
            await send_webhook(callback_url, {
                "event": "job.expired",
                "job_id": job_id,
                "error": f"Job expired after {wait_time:.0f}s in queue",
            })
        return

    position = job["position"]
    logger.info(f" {engine.gpu_id}] Processing {job_id}: position={position} (waited {wait_time:.0f}s)")

    await update_job(r, job_id,
                     status=JobStatus.PROCESSING.value,
                     started_at=time.time(),
                     progress=0.1)

    try:
        raw_video_path, gen_time = engine.generate(
            position=position,
            image_path=job["image_path"],
            prompt=job.get("prompt", ""),
            duration=int(job.get("duration", 10)),
            seed=int(job.get("seed", 42)),
        )
        await update_job(r, job_id, progress=0.7)

        if settings.POSTPROCESS_ENABLED:
            await update_job(r, job_id, status=JobStatus.POSTPROCESSING.value, progress=0.75)

            final_path = raw_video_path.replace(".mp4", "_final.mp4")
            pp_time = run_postprocess(
                raw_video_path, final_path,
                target_fps=settings.TARGET_FPS,
                upscale_factor=settings.UPSCALE_FACTOR,
            )

            if os.path.isfile(raw_video_path) and raw_video_path != final_path:
                os.unlink(raw_video_path)

            video_path = save_video(job_id, final_path)
        else:
            pp_time = 0
            video_path = save_video(job_id, raw_video_path)

        await update_job(r, job_id,
                         status=JobStatus.COMPLETED.value,
                         progress=1.0,
                         completed_at=time.time(),
                         video_path=video_path)

        from api.auth import increment_usage
        await increment_usage(job["api_key_hash"], "completed_jobs")

        logger.info(
            f" {engine.gpu_id}] Completed {job_id}: "
            f"gen={gen_time:.1f}s, pp={pp_time:.1f}s, total={gen_time + pp_time:.1f}s"
        )

        callback_url = job.get("callback_url", "")
        if callback_url:
            await send_webhook(callback_url, {
                "event": "job.completed",
                "job_id": job_id,
                "video_url": f"{settings.VIDEO_BASE_URL}/{job_id}/video",
                "metadata": {
                    "position": position,
                    "duration": int(job.get("duration", 10)),
                    "resolution": settings.OUTPUT_RESOLUTION,
                    "fps": settings.OUTPUT_FPS,
                },
            })

    except Exception as e:
        logger.error(f" {engine.gpu_id}] Job {job_id} failed: {e}", exc_info=True)
        await update_job(r, job_id,
                         status=JobStatus.FAILED.value,
                         error=str(e),
                         completed_at=time.time())

        from api.auth import increment_usage
        await increment_usage(job["api_key_hash"], "failed_jobs")

        callback_url = job.get("callback_url", "")
        if callback_url:
            await send_webhook(callback_url, {
                "event": "job.failed",
                "job_id": job_id,
                "error": str(e),
            })


async def main_loop(gpu_id: int):
    logger.info(f" {gpu_id}] Starting GPU worker")

    engine = InferenceEngine(
        gpu_id=gpu_id,
        model_path=settings.BASE_MODEL_PATH,
        lora_dir=settings.LORA_DIR,
        fast_lora_dir=settings.FAST_LORA_DIR,
    )
    engine.startup()

    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    await r.ping()
    logger.info(f" {gpu_id}] Connected to Redis")

    while not SHUTDOWN:
        try:
            job_id = await pop_job(r, timeout=5)
            if job_id is None:
                continue
            await process_job(engine, r, job_id)
        except Exception as e:
            logger.error(f" {gpu_id}] Main loop error: {e}", exc_info=True)
            await asyncio.sleep(2)

    logger.info(f" {gpu_id}] Worker shut down cleanly")
    await r.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    asyncio.run(main_loop(args.gpu_id))
