"""Ray actor wrapping GFPGAN V5 face enhancement.

Keeps the GFPGAN model loaded, with free_cache() to release CUDA cache
before LTX inference uses the VRAM.
"""

import importlib.util
import logging
import os
import time

import ray
import torch

import config as cfg

logger = logging.getLogger(__name__)

V5_SCRIPT = "/workspace/gfpgan_enhance_stable_v5.py"


def _import_v5():
    spec = importlib.util.spec_from_file_location("gfpgan_v5", V5_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@ray.remote(num_gpus=0.2)
class GFPGANEnhanceActor:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        logger.info("GFPGANEnhanceActor: loading...")
        self._v5 = _import_v5()
        self._enhancer = self._v5.build_enhancer(cfg.GFPGAN_MODEL)
        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - t0
        logger.info(f"GFPGANEnhanceActor ready in {elapsed:.1f}s")

    def free_cache(self) -> None:
        """Release CUDA cache to free VRAM for LTX inference."""
        torch.cuda.empty_cache()

    def enhance(
        self,
        input_path: str,
        output_path: str | None = None,
        *,
        blend: float = 0.85,
        batch_size: int = cfg.ENHANCE_BATCH_SIZE,
        detect_every: int = cfg.ENHANCE_DETECT_EVERY,
        win_r: int = 3,
        temporal_blend: float = cfg.ENHANCE_TEMPORAL_BLEND,
        deflicker: int = cfg.ENHANCE_DEFLICKER,
    ) -> dict:
        t0 = time.perf_counter()

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_enhanced{ext}"

        stats = self._v5.process_video(
            self._enhancer,
            input_path,
            output_path,
            blend=blend,
            batch_size=batch_size,
            detect_every=detect_every,
            win_r=win_r,
            temporal_blend=temporal_blend,
            deflicker=deflicker,
        )

        elapsed = time.perf_counter() - t0
        return {
            "output_path": output_path,
            "elapsed_s": round(elapsed, 1),
            "stats": stats or {},
        }

    def get_status(self) -> dict:
        alloc = torch.cuda.memory_allocated() / 1024**2
        return {"model": "GFPGANv1.4", "ready": True, "vram_mb": round(alloc)}
