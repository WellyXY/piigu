# Parrot API — Pod Deployment Guide

## Prerequisites (before running fresh_setup.sh)
1. Accept Gemma 3 terms: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
2. Get a HuggingFace token with read access
3. Have an H100 RunPod (80GB VRAM)

## Step 1: Upload LoRAs to pod
```bash
# Position LoRAs (8 files, ~428MB each)
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  models_backup/ltx23_position_loras/*.safetensors \
  root@<HOST>:/workspace/models/loras/

# Community LoRAs
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  models_backup/ltx23_community_loras/LTX2_3_NSFW_furry_concat_v2.safetensors \
  models_backup/ltx23_community_loras/LTX23_NSFW_Motion.safetensors \
  root@<HOST>:/workspace/models/loras/

# Optional: deepthroat LoRA (1.1GB — currently only in old pod backup)
# scp -P <PORT> -i ~/.ssh/id_ed25519 \
#   models_backup/ltx23_community_loras/ltxdeepthroat_v01.safetensors \
#   root@<HOST>:/workspace/models/loras/
```

## Step 2: Upload parrot-api code
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 -r \
  pod_deploy/parrot-api \
  pod_deploy/gfpgan_enhance_stable_v5.py \
  root@<HOST>:/workspace/
```

## Step 3: Run fresh_setup.sh
```bash
# SSH into pod
ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST>

# Run setup (downloads ~66GB total: LTX 43GB + Gemma 23GB + upscaler + GFPGAN)
bash /path/to/fresh_setup.sh <HF_TOKEN>
```
Setup takes ~30-40 min (mostly download time). On completion server starts automatically.

## Step 4: Verify
```bash
curl http://localhost:8000/status
# Expected: {"status":"ready","ltx":{...},"gfpgan":{...},"queue_depth":0}
```

---

## File Layout on Pod
```
/workspace/
├── parrot-api/
│   ├── server.py                    # FastAPI + Ray server entry point
│   ├── config.py                    # All paths and defaults
│   ├── persistent_pipeline.py       # Hot-swap LoRA logic (keeps transformer in VRAM)
│   ├── persistent_prompt_encoder.py # Keeps Gemma in VRAM (int8 via bitsandbytes)
│   ├── actors/
│   │   ├── ltx_actor.py             # Ray actor: LTX inference
│   │   └── gfpgan_actor.py          # Ray actor: GFPGAN face enhancement
│   └── requirements.txt
├── gfpgan_enhance_stable_v5.py      # GFPGAN face enhancement (loaded by actor)
├── models/
│   ├── ltx23/
│   │   ├── ltx-2.3-22b-distilled.safetensors      # 43GB transformer
│   │   └── ltx-2.3-spatial-upscaler-x2-1.0.safetensors
│   └── loras/
│       ├── LTX2_3_NSFW_furry_concat_v2.safetensors # NSFW LoRA (weight=1.0, constant)
│       ├── LTX23_NSFW_Motion.safetensors            # Motion LoRA (weight=0.7, constant)
│       ├── blow_job.safetensors                     # Position LoRA (weight=0.8, hot-swap)
│       ├── cowgirl.safetensors
│       ├── doggy.safetensors
│       ├── handjob.safetensors
│       ├── lift_clothes.safetensors
│       ├── masturbation.safetensors
│       ├── missionary.safetensors
│       └── reverse_cowgirl.safetensors
├── gemma_configs/                   # Gemma 3 12B model files (~23GB)
├── GFPGANv1.4.pth                   # GFPGAN weights
├── outputs/                         # Generated videos
├── start_server.sh                  # Wrapper: sets LD_LIBRARY_PATH, starts server
└── parrot-api.log                   # Server log
```

---

## Inference Pipeline
```
POST /generate
  {prompt, position, image_path, num_frames=249, frame_rate=25, seed=42,
   lora_weights={nsfw:1.0, motion:0.7, position:0.8}, enhance=true}

1. LTXInferenceActor.generate()
   - Fused base LoRAs: NSFW (1.0) + Motion (0.7) — baked into transformer
   - Hot-swap position LoRA (~0s if same, ~2s if changed)
   - DistilledPipeline(prompt, image, 512x768x249@25fps)
   - Gemma 3 12B int8 persistent in VRAM (~8GB after cache clear)
   - Transformer persistent in VRAM (~43GB)
   - Inference: ~12-14s on H100

2. GFPGANEnhanceActor.enhance()  [overlaps with next request's inference]
   - GFPGAN v1.4 face detection + enhancement
   - Temporal blend (win_r=3, blend=0.85)
   - FFmpeg deflicker (size=15)
   - Audio from raw video, fixed duration (-t flag, not -shortest)
   - Enhancement: ~8-10s
```

---

## VRAM Budget (H100 80GB)
| Component | VRAM |
|-----------|------|
| Transformer (bf16, fused NSFW+Motion LoRAs) | ~43GB |
| Position LoRA deltas × 8 (bf16) | ~6GB |
| Gemma 3 12B int8 | ~8GB |
| GFPGAN (num_gpus=0.2) | ~3GB |
| **Total** | **~60GB** |

---

## Critical Version Pins
```bash
# torch ecosystem MUST match cu130 (pod runs torch 2.11.0+cu130)
torch==2.11.0+cu130
torchvision==0.26.0+cu130
torchaudio==2.11.0+cu130
# Install from: --index-url https://download.pytorch.org/whl/cu130

# transformers MUST stay at 4.52.1 (other versions break LTX-2 Gemma3 loading)
transformers==4.52.1

# bitsandbytes needs libnvJitLink.so.13 — set before starting server:
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
# The start_server.sh wrapper handles this automatically
```

---

## Starting / Restarting Server
```bash
# Always use the wrapper (sets LD_LIBRARY_PATH for bitsandbytes cu13)
pkill -f 'python server.py' 2>/dev/null; sleep 2
nohup /workspace/start_server.sh > /workspace/parrot-api.log 2>&1 &

# Monitor startup (takes ~60s to load all models)
tail -f /workspace/parrot-api.log

# Check ready
curl http://localhost:8000/status
```

---

## Training a New Position LoRA
See `lora_training/` folder. Requirements:
- 20+ mp4 videos of the position (30fps, portrait, no audio needed)
- Stop inference server before training (`pkill -f 'python server.py'`)
- Peak VRAM during training: ~30GB (int8-quanto)
- Training time: ~23min for 2000 steps on H100

```bash
# 1. Upload training videos to pod
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  "training data/<position>/*.mp4" \
  root@<HOST>:/workspace/lora_training/<position>_videos/

# 2. Run preprocessing + training (see lora_training/run_training.sh)

# 3. Deploy
cp /workspace/lora_training/<position>_output/checkpoints/lora_weights_step_02000.safetensors \
   /workspace/models/loras/<position>.safetensors

# 4. Restart server
nohup /workspace/start_server.sh > /workspace/parrot-api.log 2>&1 &
```

---

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate video |
| `/status` | GET | Server health + loaded LoRAs |
| `/download/{filename}` | GET | Download output video |
| `/health` | GET | Quick health check |

## Known Issues / Fixes
- **Audio longer than video**: gfpgan_enhance_stable_v5.py uses `-t {duration}` not `-shortest` (fixed in this deploy package)
- **torchvision ABI mismatch**: `pip install diffusers` pulls cu124 torchvision; fix: `pip install --no-deps torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
- **bitsandbytes libnvJitLink.so.13 missing**: Use `start_server.sh` wrapper, not `python server.py` directly
- **webp images break ltx-trainer process_videos.py**: Convert to PNG first with PIL before preprocessing
