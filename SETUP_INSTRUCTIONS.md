# Parrot API Setup Instructions

## Requirements
- RunPod H100 80GB (or equivalent)
- HuggingFace account with Gemma 3 access accepted at:
  huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized

## One-command setup
```bash
bash scripts/fresh_setup.sh <YOUR_HF_TOKEN>
```

## What fresh_setup.sh does
1. Installs Python packages (ltx_pipelines, gfpgan, ray, etc.)
2. Patches basicsr torchvision compat
3. Downloads LTX-2.3 transformer (43GB) from Lightricks HuggingFace
4. Downloads Gemma 3 12B (23GB) — requires HF token
5. Downloads GFPGAN v1.4
6. Starts parrot-api server

## Before running, upload LoRAs:
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  loras/position/*.safetensors \
  loras/community/LTX23_NSFW_Motion.safetensors \
  LTX2_3_NSFW_furry_concat_v2.safetensors \  # store separately
  root@<HOST>:/workspace/models/loras/
```

## SSH key
~/.ssh/id_ed25519

## Important notes
- transformers must be 4.52.x (NOT 5.x) — fresh_setup.sh handles this
- After migration, IP/port changes — check RunPod console for new SSH command
