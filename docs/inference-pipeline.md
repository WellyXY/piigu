# Inference Pipeline — LTX-2.3-22B (parrot-api)

Last updated: 2026-04-10

---

## Overview

Image-to-video inference using LTX-2.3-22B dev model, running on a single H100 80GB pod.
All components are persistent in VRAM — no model reloading between jobs.

---

## Checkpoint

| File | Size | Format |
|------|------|--------|
| `ltx-2.3-22b-dev-q8-bf16.safetensors` | 43 GB | BF16 safetensors |

**How it was built:**
1. Download `unsloth/LTX-2.3-GGUF` → `ltx-2.3-22b-dev-Q8_0.gguf` (22 GB)
2. Dequantize Q8_0 → BF16 via `/root/gguf_to_safetensors.py` (4,444 transformer tensors)
3. Load GGUF to RAM → delete GGUF → save output (avoids MooseFS ~140 GB per-pod quota)
4. Patch in non-transformer tensors from `ltx-2.3-22b-distilled.safetensors`:
   - `vae.*` (170 tensors)
   - `audio_vae.*` (102 tensors)
   - `vocoder.*` (1,227 tensors)
   - `text_embedding_projection.*` (4 tensors)

Config path: `/root/config.py` → `DEV_CHECKPOINT`

---

## LoRA Stack

### Layer 1 — Base LoRAs (fused permanently into transformer at startup)

| LoRA | Weight | Purpose |
|------|--------|---------|
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 1.0 | Distilled LoRA — enables 7-step inference without ghosting |
| `LTX2_3_NSFW_furry_concat_v2.safetensors` | 1.0 (default) | NSFW content |
| `LTX23_NSFW_Motion.safetensors` | 0.7 (default) | Motion quality |

All three are fused via `fuse_loras.apply_loras()` at actor init. No disk access after startup.

### Layer 2 — Position LoRAs (hot-swapped in VRAM, ~2s per swap)

10 positions total. Delta tensors for all positions are pre-computed and held in VRAM at startup.
Swap = `subtract(old_delta) + add(new_delta)` in-place on transformer weights. No disk I/O.

| Position | Weight | Notes |
|----------|--------|-------|
| blow_job | 0.8 | |
| cowgirl | 0.8 | |
| doggy | 0.8 | |
| handjob | 0.8 | Secondary LoRA: blow_job @ 0.6 stacked on top |
| lift_clothes | 0.6 | |
| masturbation | 0.8 | |
| missionary | 0.8 | |
| reverse_cowgirl | 0.8 | |
| dildo | 0.8 | |
| boobs_play | 0.8 | |

Per-position weights are defined in `cfg.POSITION_LORA_WEIGHTS` and override the default at runtime.

---

## Inference Parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 512 × 768 |
| Frames | 249 |
| Frame rate | 25 fps |
| Image strength | 0.9 |
| Seed | 42 |
| Steps | 7 (distilled) |

---

## Text Encoder

Gemma-3 12B int8 — persistent in VRAM via `PersistentPromptEncoder`.

- User prompt is passed through **as-is** (no template wrapping)
- Prompt is received from the API caller — no hardcoded DEFAULT_PROMPTS (unlike parrot-service/Wan)

**Effective prompt structure:**
```
[Subject description] A [hair]-haired woman, facing the camera, [position_context].
[Motion detail] She [action], [motion_detail], with a steady, fluid rhythm.
[Quality] Please ensure the stability of the face and the object, and present more refined details.
```
Key terms that improve output: `stability`, `steady`, `fluid`, `rhythmic`, `refined details`, `facing the camera`

---

## Post-processing — GFPGAN Face Enhancement

Applied after generation via `enhance_video()`:

| Parameter | Value |
|-----------|-------|
| Batch size | 32 frames |
| Face detect every N frames | 2 |
| Temporal blend | 0.85 |
| Deflicker window | 15 frames |
| Upscale | 2× Lanczos (512×768 → 1024×1536) |

Model: `/workspace/GFPGANv1.4.pth`

---

## VRAM Budget (H100 80 GB)

| Component | VRAM |
|-----------|------|
| Transformer BF16 (43 GB weights) | ~43 GB |
| Gemma-3 12B int8 | ~7 GB |
| GFPGAN | ~0.5 GB |
| Position delta tensors (10×) | ~2–3 GB |
| Activations / overhead | ~5 GB |
| **Total** | **~57–60 GB** |

---

## Actor Architecture

- **`LTXInferenceActor`** — Ray remote actor (0.8 GPU), singleton
  - `PersistentDiffusionStage` — holds transformer + fused LoRAs, owns position hot-swap
  - `PersistentPromptEncoder` — holds Gemma int8, warmed up at init
  - `pipeline.stage` and `pipeline.prompt_encoder` are replaced with persistent versions after `DistilledPipeline` construction

- **`quantization=None`** — BF16 checkpoint, no FP8 path needed. `fuse_loras.apply_loras()` handles weight fusion directly.

---

## Key Files (on pod)

| File | Purpose |
|------|---------|
| `/root/actors/ltx_actor.py` | Ray actor, LoRA logic, generate() |
| `/root/config.py` | All paths and default weights |
| `/root/gguf_to_safetensors.py` | GGUF Q8→BF16 conversion script |
| `/workspace/models/ltx23/ltx-2.3-22b-dev-q8-bf16.safetensors` | Active checkpoint |
| `/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors` | Kept for rollback / tensor source |
| `/workspace/ltx2_repo/packages/ltx-core/src/ltx_core/loader/fuse_loras.py` | LoRA fusion (modified) |
