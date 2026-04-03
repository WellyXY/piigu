#!/usr/bin/env bash
# 下載三個 position 的所有 checkpoints 到本機
# 用法：bash "/Users/welly/Parrot API/lora_training/download_checkpoints.sh"

SSH_KEY="$HOME/.ssh/id_ed25519"
POD_HOST="91.199.227.82"
POD_PORT="11710"
POD="root@${POD_HOST}"
SCP="scp -i $SSH_KEY -P $POD_PORT"
LORA="/Users/welly/Parrot API/lora_training"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for pos in lift_clothes blow_job handjob; do
    DEST="$LORA/${pos}_checkpoints_v2"
    mkdir -p "$DEST"
    log "Downloading $pos checkpoints..."
    $SCP "${POD}:/workspace/lora_training/${pos}_output_v2/checkpoints/*.safetensors" "$DEST/" 2>/dev/null \
        && log "  $pos: $(ls "$DEST"/*.safetensors 2>/dev/null | wc -l | tr -d ' ') files downloaded to $DEST" \
        || log "  $pos: no checkpoints yet (training may still be running)"
done

log "Done."
log "Steps available: 250, 500, 750, 1000, 1250, 1500"
log "Compare with I2V testing to pick the best checkpoint per position."
