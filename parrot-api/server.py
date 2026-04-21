"""FastAPI + Ray inference server for LTX 2.3 video generation.

Persistent transformer + int8 Gemma in VRAM — avoids ~28s model reload per inference.
Pipeline parallelism: inference and enhancement overlap across consecutive requests.
  Request A: [inference] -> release sem -> [enhance  ]
  Request B:              -> [inference              ]
Start with:  python server.py
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import ray
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("parrot-api")


class GenerateRequest(BaseModel):
    prompt: str
    position: str = "cowgirl"
    image_path: str = Field(..., description="Path to the conditioning image on the Pod")
    height: int = cfg.DEFAULT_HEIGHT
    width: int = cfg.DEFAULT_WIDTH
    num_frames: int = cfg.DEFAULT_NUM_FRAMES
    frame_rate: int = cfg.DEFAULT_FRAME_RATE
    seed: int = cfg.DEFAULT_SEED
    lora_weights: dict = Field(default_factory=lambda: dict(cfg.DEFAULT_LORA_WEIGHTS))
    enhance: bool = True
    audio_description: str | None = Field(
        None, description="Optional Audio: section for dirty talk"
    )
    enhance_prompt: bool = False
    image_strength: float = 0.9


class GenerateResponse(BaseModel):
    raw_video: str
    enhanced_video: str | None = None
    elapsed_s: float
    inference_s: float
    enhance_s: float | None = None
    position: str
    seed: int


class StatusResponse(BaseModel):
    status: str
    ltx: dict
    gfpgan: dict
    queue_depth: int


ltx_actor = None
gfpgan_actor = None
_pending_tasks = 0
_last_enhance_ran = False  # track whether GFPGAN ran last request

# One inference at a time (single GPU).
# Released BEFORE enhance starts — next request's inference overlaps with current enhance.
_inference_sem: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ltx_actor, gfpgan_actor, _inference_sem
    _inference_sem = asyncio.Semaphore(1)

    logger.info("Starting Ray...")
    ray.init(ignore_reinit_error=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Creating GFPGAN actor first (small, fast)...")
    from actors.gfpgan_actor import GFPGANEnhanceActor
    gfpgan_actor = GFPGANEnhanceActor.remote()
    await asyncio.to_thread(ray.get, gfpgan_actor.get_status.remote())
    logger.info("GFPGAN actor ready")

    logger.info("Creating LTX actor (loads transformer into VRAM ~43GB)...")
    from actors.ltx_actor import LTXInferenceActor
    ltx_actor = LTXInferenceActor.remote()
    await asyncio.to_thread(ray.get, ltx_actor.get_status.remote())
    logger.info("LTX actor ready with persistent transformer")

    logger.info(f"Parrot API ready on {cfg.HOST}:{cfg.PORT}")
    yield

    logger.info("Shutting down...")
    ray.shutdown()


app = FastAPI(title="Parrot API", version="1.2.0", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(cfg.OUTPUT_DIR)), name="outputs")


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    global _pending_tasks

    if ltx_actor is None:
        raise HTTPException(503, "Service not ready")

    full_prompt = req.prompt
    if req.audio_description:
        full_prompt = f"{req.prompt} Audio: {req.audio_description}"

    lw = req.lora_weights
    _pending_tasks += 1
    t0 = time.perf_counter()

    global _last_enhance_ran
    try:
        async with _inference_sem:
            # Always free GFPGAN's PyTorch cached VRAM before inference.
            # GFPGAN idles at ~27GB cached; releasing it prevents OOM during LTX spatial upsampler.
            if gfpgan_actor is not None:
                await asyncio.to_thread(ray.get, gfpgan_actor.free_cache.remote())
                _last_enhance_ran = False

            infer_result = await asyncio.to_thread(
                ray.get,
                ltx_actor.generate.remote(
                    prompt=full_prompt,
                    position=req.position,
                    image_path=req.image_path,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    frame_rate=req.frame_rate,
                    seed=req.seed,
                    nsfw_w=lw.get("nsfw", 1.0),
                    motion_w=lw.get("motion", 0.7),
                    position_w=lw.get("position", 0.8),
                    image_strength=req.image_strength,
                    enhance_prompt=req.enhance_prompt,
                ),
            )
        # Semaphore released here — next request can start inference
        # while this request continues to enhance (pipeline parallelism)
    except Exception as e:
        _pending_tasks -= 1
        raise HTTPException(500, f"Inference failed: {e}")
    enhanced_path = None
    enhance_elapsed = None

    if req.enhance and gfpgan_actor is not None:
        try:
            enh_result = await asyncio.to_thread(
                ray.get,
                gfpgan_actor.enhance.remote(
                    input_path=infer_result["output_path"],
                ),
            )
            enhanced_path = enh_result["output_path"]
            enhance_elapsed = enh_result["elapsed_s"]
            _last_enhance_ran = True
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")

    _pending_tasks -= 1
    total = time.perf_counter() - t0

    return GenerateResponse(
        raw_video=infer_result["output_path"],
        enhanced_video=enhanced_path,
        elapsed_s=round(total, 1),
        inference_s=infer_result["elapsed_s"],
        enhance_s=enhance_elapsed,
        position=infer_result["position"],
        seed=infer_result["seed"],
    )


@app.get("/status", response_model=StatusResponse)
async def status():
    if ltx_actor is None:
        raise HTTPException(503, "Service not ready")
    ltx_status = await asyncio.to_thread(ray.get, ltx_actor.get_status.remote())
    gfpgan_status = await asyncio.to_thread(ray.get, gfpgan_actor.get_status.remote())
    return StatusResponse(
        status="ready",
        ltx=ltx_status,
        gfpgan=gfpgan_status,
        queue_depth=_pending_tasks,
    )


@app.get("/download/{filename}")
async def download(filename: str):
    path = cfg.OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=cfg.HOST,
        port=cfg.PORT,
        workers=1,
        log_level="info",
    )
