"""Ray actor wrapping the LTX 2.3 DistilledPipeline with persistent transformer.

The transformer (~43GB) stays in VRAM between calls.
- nsfw + motion LoRAs are fused into the transformer at startup (constant).
- Position LoRA is hot-swapped in-place (~2s) instead of rebuilding from disk (~28s).
"""

import logging
import os
import time
import uuid

import ray
import torch

# Reduce VRAM fragmentation — spatial upsampler needs a large contiguous block
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

from persistent_pipeline import PersistentDiffusionStage
from persistent_prompt_encoder import PersistentPromptEncoder
import config as cfg

logger = logging.getLogger(__name__)


def _pos_key(position: str, strength: float) -> tuple:
    return (position, round(strength, 3))


def _base_loras(nsfw_w: float, motion_w: float) -> tuple[LoraPathStrengthAndSDOps, ...]:
    """LoRAs that are always fused into the base transformer (constant)."""
    return (
        LoraPathStrengthAndSDOps(cfg.NSFW_LORA, nsfw_w, LTXV_LORA_COMFY_RENAMING_MAP),
        LoraPathStrengthAndSDOps(cfg.MOTION_LORA, motion_w, LTXV_LORA_COMFY_RENAMING_MAP),
    )


@ray.remote(num_gpus=0.8)
class LTXInferenceActor:
    """Singleton GPU actor — transformer + int8 Gemma both persistent in VRAM.

    Position LoRAs are hot-swapped in-place (~2s) on position change.
    """

    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        default_pos = "cowgirl"
        default_pos_w = cfg.DEFAULT_LORA_WEIGHTS["position"]
        default_nsfw_w = cfg.DEFAULT_LORA_WEIGHTS["nsfw"]
        default_motion_w = cfg.DEFAULT_LORA_WEIGHTS["motion"]

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=cfg.DISTILLED_CHECKPOINT,
            gemma_root=cfg.GEMMA_ROOT,
            spatial_upsampler_path=cfg.SPATIAL_UPSAMPLER,
            loras=list(_base_loras(default_nsfw_w, default_motion_w)),
        )

        # Patch prompt_encoder with persistent int8 version
        persistent_encoder = PersistentPromptEncoder(
            cfg.DISTILLED_CHECKPOINT,
            cfg.GEMMA_ROOT,
            self.dtype,
            self.device,
        )
        persistent_encoder.warmup()
        self.pipeline.prompt_encoder = persistent_encoder
        self._persistent_encoder = persistent_encoder

        # Build position_loras dict using per-position weight overrides (must match generate())
        position_loras = {
            _pos_key(pos, cfg.POSITION_LORA_WEIGHTS.get(pos, default_pos_w)):
                (path, cfg.POSITION_LORA_WEIGHTS.get(pos, default_pos_w))
            for pos, path in cfg.POSITION_LORAS.items()
        }

        default_pos_w_actual = cfg.POSITION_LORA_WEIGHTS.get(default_pos, default_pos_w)
        persistent_stage = PersistentDiffusionStage(
            checkpoint_path=cfg.DISTILLED_CHECKPOINT,
            dtype=self.dtype,
            device=self.device,
            loras=_base_loras(default_nsfw_w, default_motion_w),
            position_loras=position_loras,
            initial_position_key=_pos_key(default_pos, default_pos_w_actual),
        )
        persistent_stage.warmup()
        self.pipeline.stage = persistent_stage
        self._persistent_stage = persistent_stage

        self._current_nsfw_w = default_nsfw_w
        self._current_motion_w = default_motion_w

        elapsed = time.perf_counter() - t0
        logger.info(f"LTXInferenceActor ready in {elapsed:.1f}s (hot-swap position LoRAs)")

    def _ensure_position(self, position: str, position_w: float) -> None:
        """Hot-swap position LoRA if changed (~2s). No rebuild needed."""
        key = _pos_key(position, position_w)
        self._persistent_stage.swap_position_lora(key)

    def _ensure_base_loras(self, nsfw_w: float, motion_w: float) -> None:
        """Hot-swap NSFW/Motion base LoRA weights if changed (~1s, GPU layer-by-layer)."""
        logger.info(
            f"_ensure_base_loras: requested nsfw={nsfw_w} motion={motion_w}; "
            f"current nsfw={self._current_nsfw_w} motion={self._current_motion_w}"
        )
        if nsfw_w == self._current_nsfw_w and motion_w == self._current_motion_w:
            return
        self._persistent_stage.swap_base_loras({"nsfw": nsfw_w, "motion": motion_w})
        self._current_nsfw_w = nsfw_w
        self._current_motion_w = motion_w

    def generate(
        self,
        prompt: str,
        position: str,
        image_path: str,
        *,
        height: int = cfg.DEFAULT_HEIGHT,
        width: int = cfg.DEFAULT_WIDTH,
        num_frames: int = cfg.DEFAULT_NUM_FRAMES,
        frame_rate: int = cfg.DEFAULT_FRAME_RATE,
        seed: int = cfg.DEFAULT_SEED,
        nsfw_w: float = cfg.DEFAULT_LORA_WEIGHTS["nsfw"],
        motion_w: float = cfg.DEFAULT_LORA_WEIGHTS["motion"],
        position_w: float = cfg.DEFAULT_LORA_WEIGHTS["position"],
        image_strength: float = 0.9,
        enhance_prompt: bool = False,
        # server.py passes these names; accept both for compatibility
        nsfw_weight: float | None = None,
        motion_weight: float | None = None,
        position_weight: float | None = None,
    ) -> dict:
        t0 = time.perf_counter()

        # Accept _weight aliases from server.py
        if nsfw_weight is not None:
            nsfw_w = nsfw_weight
        if motion_weight is not None:
            motion_w = motion_weight
        if position_weight is not None:
            position_w = position_weight

        # Apply per-position weight override if defined (only if no explicit override)
        if position_weight is None:
            position_w = cfg.POSITION_LORA_WEIGHTS.get(position, position_w)

        self._ensure_base_loras(nsfw_w, motion_w)
        self._ensure_position(position, position_w)

        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=image_strength)]
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        vid_id = uuid.uuid4().hex[:8]
        output_path = str(cfg.OUTPUT_DIR / f"{position}_{vid_id}.mp4")

        with torch.inference_mode():
            video, audio = self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
            )

        with torch.inference_mode():
            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )

        elapsed = time.perf_counter() - t0
        logger.info(f"Generated {position} (w={position_w}) in {elapsed:.1f}s -> {output_path}")
        return {
            "output_path": output_path,
            "elapsed_s": round(elapsed, 1),
            "position": position,
            "seed": seed,
        }

    def free_vram(self) -> None:
        torch.cuda.empty_cache()

    def get_status(self) -> dict:
        stage = self._persistent_stage
        return {
            "current_position": stage._current_position_key,
            "transformer_loaded": stage.is_loaded,
            "gemma_loaded": self._persistent_encoder._cached_text_encoder is not None,
            "device": str(self.device),
            "position_loras_cached": list(stage._position_deltas.keys()),
        }
