"""
Inference engine: wraps Wan 2.2 I2V pipeline with LoRA hot-swapping.

Each GPU worker creates one InferenceEngine instance at startup,
which pre-loads the base model and Fast LoRA into VRAM.
Position LoRAs are loaded/swapped dynamically per job (~1s).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

logger = logging.getLogger(__name__)

LORA_MAP = {
    "blow_job":        0,
    "cowgirl":         1,
    "doggy":           2,
    "handjob":         3,
    "lift_clothes":    4,
    "masturbation":    5,
    "missionary":      6,
    "reverse_cowgirl": 7,
}

FAST_LORA_FILES = {
    "high": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    "low":  "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
}


def _find_lora_paths(position: str, lora_dir: str) -> dict[str, str]:
    gpu_idx = LORA_MAP[position]
    lora_folder = os.path.join(lora_dir, f"lora_{gpu_idx}", f"wan_i2v_lora_{gpu_idx}")

    if not os.path.exists(lora_folder):
        raise FileNotFoundError(f"LoRA folder not found: {lora_folder}")

    high_files = sorted([
        f for f in os.listdir(lora_folder)
        if f.endswith("_high_noise.safetensors")
    ])
    if not high_files:
        raise FileNotFoundError(f"No safetensors found in {lora_folder}")

    base = high_files[-1].replace("_high_noise.safetensors", "")

    return {
        "high": os.path.join(lora_folder, f"{base}_high_noise.safetensors"),
        "low":  os.path.join(lora_folder, f"{base}_low_noise.safetensors"),
    }


@dataclass
class InferenceEngine:
    gpu_id: int
    model_path: str
    lora_dir: str
    fast_lora_dir: str

    pipe: Optional[WanImageToVideoPipeline] = field(default=None, init=False)
    current_position: Optional[str] = field(default=None, init=False)
    _device: str = field(default="", init=False)

    def startup(self):
        self._device = f"cuda:{self.gpu_id}"
        logger.info(f"[GPU {self.gpu_id}] Loading base model from {self.model_path}")
        t0 = time.time()

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to(self._device)

        logger.info(f"[GPU {self.gpu_id}] Base model loaded in {time.time() - t0:.1f}s")

        self._load_fast_loras()
        logger.info(f"[GPU {self.gpu_id}] Engine ready")

    def _load_fast_loras(self):
        high_path = os.path.join(self.fast_lora_dir, FAST_LORA_FILES["high"])
        low_path = os.path.join(self.fast_lora_dir, FAST_LORA_FILES["low"])

        if not os.path.exists(high_path):
            logger.warning(f"[GPU {self.gpu_id}] Fast LoRA not found at {high_path}, skipping")
            return

        logger.info(f"[GPU {self.gpu_id}] Loading Fast LoRA (LightX2V)")
        self.pipe.load_lora_weights(high_path, adapter_name="fast_high")
        self.pipe.load_lora_weights(
            low_path, adapter_name="fast_low",
            load_into_transformer_2=True,
        )

    def switch_position(self, position: str):
        if position == self.current_position:
            logger.info(f"[GPU {self.gpu_id}] Position LoRA '{position}' already loaded")
            return

        t0 = time.time()

        if self.current_position is not None:
            try:
                self.pipe.delete_adapters([f"{self.current_position}_high", f"{self.current_position}_low"])
            except Exception:
                pass

        paths = _find_lora_paths(position, self.lora_dir)
        self.pipe.load_lora_weights(paths["high"], adapter_name=f"{position}_high")
        self.pipe.load_lora_weights(
            paths["low"], adapter_name=f"{position}_low",
            load_into_transformer_2=True,
        )

        self.current_position = position
        logger.info(f"[GPU {self.gpu_id}] Switched to '{position}' in {time.time() - t0:.1f}s")

    def _activate_adapters(self, position: str, lora_scale: float = 1.0,
                           fast: bool = True, fast_high_scale: float = 1.5,
                           fast_low_scale: float = 1.0):
        names = [f"{position}_high", f"{position}_low"]
        weights = [lora_scale, lora_scale]

        if fast:
            names += ["fast_high", "fast_low"]
            weights += [fast_high_scale, fast_low_scale]

        self.pipe.set_adapters(names, adapter_weights=weights)

    def generate(
        self,
        *,
        position: str,
        image_path: str,
        prompt: str = "",
        duration: int = 10,
        seed: int = 42,
        lora_scale: float = 1.0,
        fast: bool = True,
        high_steps: int = 4,
        low_steps: int = 4,
        fast_high_scale: float = 1.5,
        fast_low_scale: float = 1.0,
        width: int = 512,
        height: int = 768,
        fps: int = 16,
    ) -> tuple[str, float]:
        """
        Generate a video. Returns (output_path, generation_time_seconds).
        """
        self.switch_position(position)
        self._activate_adapters(
            position, lora_scale=lora_scale,
            fast=fast, fast_high_scale=fast_high_scale, fast_low_scale=fast_low_scale,
        )

        num_frames = 161 if duration >= 10 else 81
        num_frames = ((num_frames - 1) // 4) * 4 + 1

        image = Image.open(image_path).convert("RGB").resize((width, height))
        generator = torch.Generator(device=self._device).manual_seed(seed)

        total_steps = (high_steps + low_steps) if fast else 30
        guidance = 1.0 if fast else 5.0

        logger.info(
            f"[GPU {self.gpu_id}] Generating: position={position}, "
            f"frames={num_frames}, steps={total_steps}, seed={seed}"
        )

        t0 = time.time()
        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted",
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance,
            num_inference_steps=total_steps,
            generator=generator,
        ).frames[0]
        gen_time = time.time() - t0

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        export_to_video(output, tmp.name, fps=fps)

        logger.info(f"[GPU {self.gpu_id}] Generated in {gen_time:.1f}s → {tmp.name}")
        return tmp.name, gen_time
