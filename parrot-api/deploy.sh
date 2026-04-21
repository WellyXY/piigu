#!/usr/bin/env bash
# Deploy parrot-api to RunPod H100
# Usage: bash deploy.sh <ssh-port> [host-ip]
# Example: bash deploy.sh 16394 87.120.211.209

set -euo pipefail

PORT="${1:?Usage: deploy.sh <ssh-port> [host-ip]}"
HOST_IP="${2:-87.120.211.209}"
HOST="root@$HOST_IP"
SSH="ssh -p $PORT -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"
SCP="scp -P $PORT -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"

REMOTE_DIR="/workspace/parrot-api"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Uploading parrot-api to $HOST:$REMOTE_DIR ==="
$SSH $HOST "mkdir -p $REMOTE_DIR/actors"

$SCP "$LOCAL_DIR/config.py"                    "$HOST:$REMOTE_DIR/config.py"
$SCP "$LOCAL_DIR/server.py"                    "$HOST:$REMOTE_DIR/server.py"
$SCP "$LOCAL_DIR/__init__.py"                  "$HOST:$REMOTE_DIR/__init__.py"
$SCP "$LOCAL_DIR/persistent_pipeline.py"       "$HOST:$REMOTE_DIR/persistent_pipeline.py"
$SCP "$LOCAL_DIR/persistent_prompt_encoder.py" "$HOST:$REMOTE_DIR/persistent_prompt_encoder.py"
$SCP "$LOCAL_DIR/actors/__init__.py"           "$HOST:$REMOTE_DIR/actors/__init__.py"
$SCP "$LOCAL_DIR/actors/ltx_actor.py"          "$HOST:$REMOTE_DIR/actors/ltx_actor.py"
$SCP "$LOCAL_DIR/actors/gfpgan_actor.py"       "$HOST:$REMOTE_DIR/actors/gfpgan_actor.py"
$SCP "$LOCAL_DIR/requirements.txt"             "$HOST:$REMOTE_DIR/requirements.txt"

echo "=== Setting up venv + dependencies ==="
$SSH $HOST "
set -e
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:\$LD_LIBRARY_PATH

# System deps
apt-get install -y ffmpeg -q 2>/dev/null || (apt-get update -q && apt-get install -y ffmpeg -q)

# Create persistent venv in /workspace if not exists
if [ ! -d /workspace/venv ]; then
  python -m venv /workspace/venv --system-site-packages
  echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:\$LD_LIBRARY_PATH' >> /workspace/venv/bin/activate
fi
source /workspace/venv/bin/activate

pip install -q -r $REMOTE_DIR/requirements.txt
pip install -q opencv-python-headless gfpgan facexlib basicsr bitsandbytes --no-deps
pip install -q 'transformers==4.52.0' || true
pip install -q -e /workspace/ltx2_repo/packages/ltx-core \
               -e /workspace/ltx2_repo/packages/ltx-pipelines
# Remove torch from venv — fall back to system torch 2.4.1+cu124 via --system-site-packages
# (ltx packages may have upgraded torch; pip uninstall leaves stale dirs; use shutil)
python3 -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [
  '/workspace/venv/lib/python3.11/site-packages/torch',
  '/workspace/venv/lib/python3.11/site-packages/torchvision',
  '/workspace/venv/lib/python3.11/site-packages/torchaudio',
]]" && echo 'torch removed from venv (using system 2.4.1)'

# Patches
TV='/usr/local/lib/python3.11/dist-packages/torchvision/__init__.py'
grep -q '_meta_registrations' \"\$TV\" && sed -i 's/from torchvision import _meta_registrations, /from torchvision import /' \"\$TV\" && echo 'torchvision patched'

BS='/usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py'
grep -q 'functional_tensor' \"\$BS\" && sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \"\$BS\" && echo 'basicsr patched'

python $REMOTE_DIR/patch_nms.py

GEMMA_PY=\$(find /workspace/venv /usr/local/lib -name 'modeling_gemma3.py' 2>/dev/null | head -1)
grep -q 'torch.autocast(device_type=device_type, enabled=False)' \"\$GEMMA_PY\" && \
  sed -i 's/with torch.autocast(device_type=device_type, enabled=False):/with torch.amp.autocast(device_type=device_type, dtype=torch.float32, enabled=False):/' \"\$GEMMA_PY\" && echo 'gemma3 autocast patched'

# Fix transformers 4.52.0 + torch<2.5: ALL_PARALLEL_STYLES is None → TypeError in post_init
MODELING_UTILS=/workspace/venv/lib/python3.11/site-packages/transformers/modeling_utils.py
grep -q 'ALL_PARALLEL_STYLES is not None' \"\$MODELING_UTILS\" || \
  sed -i 's/if self._tp_plan is not None and is_torch_greater_or_equal(\"2.3\"):/if self._tp_plan is not None and is_torch_greater_or_equal(\"2.3\") and ALL_PARALLEL_STYLES is not None:/' \"\$MODELING_UTILS\" && echo 'transformers ALL_PARALLEL_STYLES patched'

mkdir -p /workspace/outputs
"

echo ""
echo "=== Deploy complete ==="
echo "Start server:"
echo "  $SSH $HOST 'export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:\$LD_LIBRARY_PATH && cd $REMOTE_DIR && nohup /workspace/venv/bin/python server.py > /workspace/parrot-api.log 2>&1 &'"
echo "Check logs:"
echo "  $SSH $HOST 'tail -f /workspace/parrot-api.log'"
