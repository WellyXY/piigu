#!/usr/bin/env bash
# =============================================================================
# 本機執行：SCP 上傳影片 + 腳本，啟動 pod 訓練
# 用法：
#   nohup bash "/Users/welly/Parrot API/lora_training/run_all_training.sh" > ~/lora_upload.log 2>&1 &
#   tail -f ~/lora_upload.log
# =============================================================================
set -euo pipefail

SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="91.199.227.82"
POD_PORT="11710"
POD="root@${POD_HOST}"
SSH="ssh -i $SSH_KEY $POD -p $POD_PORT"
SCP="scp -i $SSH_KEY -P $POD_PORT"
WORK="/Users/welly/Parrot API"
LORA="$WORK/lora_training"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== LoRA Retrain v2 — Upload & Launch ==="
log "Positions: lift_clothes → blow_job → handjob"
log ""

# ── Step 1: 確認 pod 可連 ─────────────────────────────────────────────────
log "Checking pod connectivity..."
$SSH "echo 'Pod OK'" || { log "ERROR: Cannot connect to pod. Check SSH key and host."; exit 1; }
$SSH "mkdir -p /workspace/lora_training /workspace/logs"

# ── Step 2: 上傳腳本 + YAML ────────────────────────────────────────────────
log "Uploading YAML configs and scripts..."
$SCP \
    "$LORA/lift_clothes_lora_v2.yaml" \
    "$LORA/blow_job_lora_v2.yaml" \
    "$LORA/handjob_lora.yaml" \
    "$LORA/prepare_dataset_v2.py" \
    "$LORA/train_all_pod.sh" \
    "${POD}:/workspace/lora_training/"
log "Scripts uploaded."

# ── Step 3: 上傳影片 ────────────────────────────────────────────────────────
for pos in lift_clothes blow_job handjob; do
    VIDEO_DIR="$WORK/training data/$pos"
    if [ ! -d "$VIDEO_DIR" ]; then
        log "ERROR: Video directory not found: $VIDEO_DIR"
        exit 1
    fi
    MP4_COUNT=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
    log "Uploading $pos videos ($MP4_COUNT files)..."
    $SSH "mkdir -p /workspace/lora_training/${pos}_videos"
    $SCP "$VIDEO_DIR/"*.mp4 "${POD}:/workspace/lora_training/${pos}_videos/"
    log "  $pos upload done."
done

# ── Step 4: 啟動 pod 訓練（背景 nohup）────────────────────────────────────
log "Starting training on pod (nohup background)..."
REMOTE_PID=$($SSH "nohup bash /workspace/lora_training/train_all_pod.sh >> /workspace/logs/train_all.log 2>&1 & echo \$!")
log ""
log "============================================"
log "  Training started on pod. PID: $REMOTE_PID"
log "  Expected duration: 4-6 hours"
log "============================================"
log ""
log "Monitor progress:"
log "  ssh -i $SSH_KEY $POD -p $POD_PORT 'tail -f /workspace/logs/train_all.log'"
log ""
log "Check checkpoints:"
log "  ssh -i $SSH_KEY $POD -p $POD_PORT 'find /workspace/lora_training -name \"*.safetensors\" | sort'"
log ""
log "Download checkpoints when done:"
log "  bash \"$LORA/download_checkpoints.sh\""
log ""
log "Upload complete. You can close this terminal — training runs on pod independently."
