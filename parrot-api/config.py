"""Centralized configuration for the LTX 2.3 inference service."""

from pathlib import Path

# -- Model paths -------------------------------------------------------------
MODEL_ROOT = Path("/workspace/models")

DISTILLED_CHECKPOINT    = str(MODEL_ROOT / "ltx23/ltx-2.3-22b-distilled.safetensors")
DEV_CHECKPOINT          = str(MODEL_ROOT / "ltx23/ltx-2.3-22b-distilled.safetensors")
OFFICIAL_DISTILLED_LORA = "/root/models/ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER = str(MODEL_ROOT / "ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
GEMMA_ROOT = "/workspace/gemma_configs"
GFPGAN_MODEL = "/workspace/GFPGANv1.4.pth"

# -- LoRA paths ---------------------------------------------------------------
LORA_DIR = MODEL_ROOT / "loras"

NSFW_LORA = str(LORA_DIR / "LTX2_3_NSFW_furry_concat_v2.safetensors")
MOTION_LORA = str(LORA_DIR / "LTX23_NSFW_Motion.safetensors")

POSITION_LORAS = {
    "blow_job":        str(LORA_DIR / "blow_job_v2.safetensors"),
    "cowgirl":         str(LORA_DIR / "cowgirl.safetensors"),
    "doggy":           str(LORA_DIR / "doggy.safetensors"),
    "handjob":         str(LORA_DIR / "handjob.safetensors"),
    "lift_clothes":    str(LORA_DIR / "lift_clothes_v2.safetensors"),
    "masturbation":    str(LORA_DIR / "masturbation.safetensors"),
    "missionary":      str(LORA_DIR / "missionary.safetensors"),
    "reverse_cowgirl": str(LORA_DIR / "reverse_cowgirl.safetensors"),
    "dildo":           str(LORA_DIR / "dildo.safetensors"),
    "boobs_play":      str(LORA_DIR / "boobs_play.safetensors"),
    "cumshot":         str(LORA_DIR / "cumshot.safetensors"),
}

# Per-position LoRA weight overrides (default: DEFAULT_LORA_WEIGHTS["position"])
POSITION_LORA_WEIGHTS: dict[str, float] = {
    "blow_job":        0.8,
    "cowgirl":         0.8,
    "doggy":           0.8,
    "handjob":         0.8,
    "lift_clothes":    0.6,
    "masturbation":    0.8,
    "missionary":      0.8,
    "reverse_cowgirl": 0.8,
    "dildo":           0.8,
    "boobs_play":      0.8,
    "cumshot":         1.0,
    # dildo/boobs_play: v3 — dildo step1250, boobs_play step1000
    # cumshot: v2 step2000, trained on DEV (distilled inference; start w=1.0, bump to 1.2-1.5 if too weak)
}

# Secondary LoRA stacked on top of position LoRA for certain positions
POSITION_SECONDARY: dict[str, tuple[str, float]] = {
    "handjob": ("blow_job", 0.6),
}

# -- Default inference params (matching run_batch_3lora.sh) --------------------
DEFAULT_HEIGHT = 960   # was 768 — raised 2026-04-22 after 20-case A/B test showed better detail at 640×960
DEFAULT_WIDTH = 640    # was 512
DEFAULT_NUM_FRAMES = 249
DEFAULT_FRAME_RATE = 25
DEFAULT_SEED = 42

DEFAULT_LORA_WEIGHTS = {
    "nsfw": 1.0,
    "motion": 0.7,
    "position": 1.2,
}

# -- Default V5 face enhancement params ---------------------------------------
ENHANCE_BATCH_SIZE = 8  # reduced from 32 to keep activation VRAM peak ~5GB (bf16 Gemma + LTX leaves ~13GB headroom)
ENHANCE_DETECT_EVERY = 2
ENHANCE_TEMPORAL_BLEND = 0.85
ENHANCE_DEFLICKER = 25  # 15→25: stronger luma smoothing (keep DETECT_EVERY=2 to avoid stale bbox blur on motion)
ENHANCE_UPSCALE = 2  # ffmpeg lanczos 2x upscale after GFPGAN (512x768 → 1024x1536)

# -- Output / Server ----------------------------------------------------------
OUTPUT_DIR = Path("/workspace/outputs")
HOST = "0.0.0.0"
PORT = 8000
