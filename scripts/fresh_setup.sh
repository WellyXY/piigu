#!/usr/bin/env bash
# =============================================================
# Parrot API — Fresh Pod Setup Script
# Run this on a new H100 RunPod (or any CUDA 12.x Ubuntu pod)
#
# Usage:
#   bash fresh_setup.sh <HF_TOKEN>
#
# Before running:
#   1. Accept Gemma 3 terms at huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
#   2. Upload LoRAs from local models_backup/:
#      scp -P <PORT> -i ~/.ssh/id_ed25519 \
#        models_backup/ltx23_position_loras/*.safetensors \
#        models_backup/ltx23_community_loras/LTX2_3_NSFW_furry_concat_v2.safetensors \
#        models_backup/ltx23_community_loras/LTX23_NSFW_Motion.safetensors \
#        root@<HOST>:/workspace/models/loras/
# =============================================================

set -euo pipefail

HF_TOKEN="${1:?Usage: bash fresh_setup.sh <HF_TOKEN>}"

echo "============================================"
echo " Parrot API Fresh Setup"
echo " $(date)"
echo "============================================"

mkdir -p /workspace/models/ltx23 /workspace/models/loras \
         /workspace/gemma_configs /workspace/outputs

# ── Step 1: Clone LTX-2 repo & install packages ──────────────
echo ""
echo "[1/6] Installing LTX-2 packages..."
if [ ! -d /workspace/ltx2_repo ]; then
    git clone https://github.com/Lightricks/LTX-2.git /workspace/ltx2_repo
fi
pip install -q --no-deps \
    -e /workspace/ltx2_repo/packages/ltx-core \
    -e /workspace/ltx2_repo/packages/ltx-pipelines

echo "[1/6] Installing other Python deps..."
pip install -q av accelerate gfpgan opencv-python-headless imageio imageio-ffmpeg \
    soundfile bitsandbytes safetensors einops "transformers==4.52.1" sentencepiece protobuf \
    "ray[default]" fastapi "uvicorn[standard]" pydantic

# Pin torch ecosystem versions to match cu130 builds (avoids ABI mismatch)
pip install --no-deps \
    "torchvision==0.26.0+cu130" \
    "torchaudio==2.11.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130

# bitsandbytes on CUDA 13 needs libnvJitLink.so.13 — set LD_LIBRARY_PATH
NVCU13_LIB=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib
if [ -d "$NVCU13_LIB" ]; then
    echo "export LD_LIBRARY_PATH=${NVCU13_LIB}:\${LD_LIBRARY_PATH}" >> /root/.bashrc
    export LD_LIBRARY_PATH="${NVCU13_LIB}:${LD_LIBRARY_PATH}"
    echo "  cu13 lib path set"
fi

apt-get install -y -q ffmpeg

# Patch basicsr torchvision compat
DEG=/usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py
if [ -f "$DEG" ] && grep -q "functional_tensor" "$DEG"; then
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEG"
    echo "  basicsr patched"
fi
echo "[1/6] Done"

# ── Step 2: Download LTX transformer + upscaler ──────────────
echo ""
echo "[2/6] Downloading LTX-2.3 transformer (43GB)..."
if [ ! -f /workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors ] || \
   [ "$(stat -c%s /workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors 2>/dev/null || echo 0)" -lt 40000000000 ]; then
    wget -q --show-progress \
        -O /workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors \
        'https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors'
else
    echo "  Already downloaded, skipping"
fi

echo "[2/6] Downloading spatial upscaler..."
if [ ! -f /workspace/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors ]; then
    wget -q -O /workspace/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
        'https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors'
fi
echo "[2/6] Done"

# ── Step 3: Download Gemma 3 (gated) ─────────────────────────
echo ""
echo "[3/6] Downloading Gemma 3 12B (~23GB, requires HF token)..."
if [ ! -f /workspace/gemma_configs/tokenizer.model ]; then
    python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
    local_dir="/workspace/gemma_configs",
    token="$HF_TOKEN",
    ignore_patterns=["*.bin", "flax_model*", "tf_model*"],
)
print("Gemma done")
PYEOF
else
    echo "  Already downloaded, skipping"
fi
echo "[3/6] Done"

# ── Step 4: Download GFPGAN ──────────────────────────────────
echo ""
echo "[4/6] Downloading GFPGAN..."
if [ ! -f /workspace/GFPGANv1.4.pth ]; then
    wget -q -O /workspace/GFPGANv1.4.pth \
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
fi
echo "[4/6] Done"

# ── Step 5: Verify LoRAs ─────────────────────────────────────
echo ""
echo "[5/6] Checking LoRAs..."
MISSING=0
for lora in LTX2_3_NSFW_furry_concat_v2.safetensors LTX23_NSFW_Motion.safetensors \
            blow_job.safetensors cowgirl.safetensors doggy.safetensors \
            handjob.safetensors lift_clothes.safetensors masturbation.safetensors \
            missionary.safetensors reverse_cowgirl.safetensors; do
    if [ ! -f "/workspace/models/loras/$lora" ]; then
        echo "  MISSING: $lora"
        MISSING=$((MISSING + 1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo ""
    echo "Upload missing LoRAs with:"
    echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 \\"
    echo "    models_backup/ltx23_position_loras/*.safetensors \\"
    echo "    models_backup/ltx23_community_loras/LTX2_3_NSFW_furry_concat_v2.safetensors \\"
    echo "    models_backup/ltx23_community_loras/LTX23_NSFW_Motion.safetensors \\"
    echo "    root@<HOST>:/workspace/models/loras/"
    exit 1
fi
echo "[5/6] All LoRAs present"

# ── Step 6: Start server ─────────────────────────────────────
echo ""
echo "[6/6] Starting Parrot API server..."
pkill -f 'python server.py' 2>/dev/null || true
sleep 2

# Use wrapper to ensure LD_LIBRARY_PATH is set for bitsandbytes cu13
cat > /workspace/start_server.sh << 'WRAPPER'
#!/bin/bash
NVCU13=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib
[ -d "$NVCU13" ] && export LD_LIBRARY_PATH="${NVCU13}:${LD_LIBRARY_PATH}"
cd /workspace/parrot-api
exec python server.py
WRAPPER
chmod +x /workspace/start_server.sh

nohup /workspace/start_server.sh > /workspace/parrot-api.log 2>&1 &
echo "  Server PID: $!"
echo ""
echo "============================================"
echo " Setup complete! Server starting..."
echo " Monitor: tail -f /workspace/parrot-api.log"
echo " Status:  curl http://localhost:8000/status"
echo "============================================"
