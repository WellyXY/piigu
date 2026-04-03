#!/usr/bin/env bash
# =============================================================================
# Master LoRA Trainer — lift_clothes → blow_job → handjob
# 在 pod 上執行：
#   nohup bash /workspace/lora_training/train_all_pod.sh >> /workspace/logs/train_all.log 2>&1 &
# =============================================================================
set -uo pipefail

TRAINER=/workspace/ltx2_repo/packages/ltx-trainer
WORK=/workspace/lora_training
MODEL=/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors
GEMMA=/workspace/gemma_configs
LOG=/workspace/logs/train_all.log

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

mkdir -p /workspace/logs

# ── 安裝依賴（只跑一次）────────────────────────────────────────────────────
log "Installing ltx-trainer dependencies..."
pip install -q --no-deps -e "$TRAINER" 2>&1 | tail -2
pip install -q typer rich pandas pillow-heif scenedetect peft optimum-quanto wandb 'setuptools>=79' 2>&1 | tail -2
pip install -q torchcodec 2>&1 | tail -2 || log "torchcodec not available, continuing"
log "Dependencies installed."

# ── 停推理 server（釋放 VRAM）──────────────────────────────────────────────
log "Stopping inference server to free VRAM..."
pkill -f 'python3 server.py' 2>/dev/null || true
pkill -f 'gpu_worker' 2>/dev/null || true
sleep 5
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' MiB')
log "VRAM free: ${VRAM_FREE} MiB"
if [ "${VRAM_FREE:-0}" -lt 50000 ]; then
    log "WARNING: Less than 50GB VRAM free. Training may OOM."
fi

# ── 通用 preprocess 函數 ────────────────────────────────────────────────────
preprocess_position() {
    local pos=$1
    local preproc_dir="$WORK/${pos}_preprocessed_v2"
    local jsonl="$WORK/${pos}_dataset.jsonl"

    log "[$pos] Creating dataset manifest..."
    python3 "$WORK/prepare_dataset_v2.py" --position "$pos" || die "[$pos] prepare_dataset_v2.py failed"

    local n_rows
    n_rows=$(wc -l < "$jsonl")
    log "[$pos] Dataset: $n_rows videos"

    mkdir -p "$preproc_dir"

    log "[$pos] Preprocessing (encoding latents + text embeddings)..."
    cd "$TRAINER"
    python3 scripts/process_dataset.py \
        "$jsonl" \
        --resolution-buckets "512x768x249;512x768x121" \
        --model-path "$MODEL" \
        --text-encoder-path "$GEMMA" \
        --output-dir "$preproc_dir" \
        --batch-size 1 \
        --device cuda \
        --load-text-encoder-in-8bit \
        2>&1 | tee -a "$LOG" \
        || die "[$pos] process_dataset.py failed"

    # ── 修復 conditions path bug ──────────────────────────────────────────
    local video_dir="$WORK/${pos}_videos"
    local pt_count
    pt_count=$(ls "$video_dir"/*.pt 2>/dev/null | wc -l || echo 0)
    if [ "$pt_count" -gt 0 ]; then
        log "[$pos] Conditions path bug detected ($pt_count .pt files in video dir). Auto-fixing..."
        mkdir -p "$preproc_dir/conditions/${pos}_videos"
        cp "$video_dir"/*.pt "$preproc_dir/conditions/${pos}_videos/"
        rm -f "$video_dir"/*.pt
        log "[$pos] Conditions path bug fixed."
    fi

    # 確認 latent + conditions 數量
    local latent_count
    latent_count=$(find "$preproc_dir/latents" -name "*.pt" 2>/dev/null | wc -l || echo 0)
    local cond_count
    cond_count=$(find "$preproc_dir/conditions" -name "*.pt" 2>/dev/null | wc -l || echo 0)
    log "[$pos] Preprocessed: $latent_count latents, $cond_count conditions"

    if [ "$latent_count" -eq 0 ]; then
        die "[$pos] No latents generated. Check video format or resolution-buckets."
    fi
}

# ── 通用 train 函數（含 retry + 自動錯誤修復）────────────────────────────────
train_position() {
    local pos=$1
    local yaml_name=$2
    local output_dir="$WORK/${pos}_output_v2"
    local yaml_src="$WORK/$yaml_name"
    local yaml_dst="$TRAINER/configs/$yaml_name"
    local max_retries=2
    local attempt=0

    mkdir -p "$output_dir"
    cp "$yaml_src" "$yaml_dst"

    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        log "[$pos] Training attempt $attempt/$max_retries..."

        local train_log="/tmp/${pos}_train_attempt${attempt}.log"
        cd "$TRAINER"
        python3 scripts/train.py "$yaml_dst" 2>&1 | tee "$train_log" | tee -a "$LOG"
        local exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            log "[$pos] Training completed successfully."
            return 0
        fi

        log "[$pos] Training failed (exit $exit_code). Diagnosing..."

        # num_samples=0 → conditions path bug
        if grep -q "num_samples=0\|No samples found\|Empty dataset" "$train_log" 2>/dev/null; then
            log "[$pos] DIAGNOSIS: num_samples=0 — conditions path bug. Auto-fixing..."
            local preproc_dir="$WORK/${pos}_preprocessed_v2"
            local video_dir="$WORK/${pos}_videos"
            mkdir -p "$preproc_dir/conditions/${pos}_videos"
            find "$video_dir" -name "*.pt" -exec cp {} "$preproc_dir/conditions/${pos}_videos/" \; 2>/dev/null || true
            log "[$pos] Fix applied. Retrying..."
            continue
        fi

        # CUDA OOM → 降低 dataloader workers
        if grep -q "out of memory\|CUDA out of memory\|OOM" "$train_log" 2>/dev/null; then
            log "[$pos] DIAGNOSIS: CUDA OOM — reducing num_dataloader_workers to 1..."
            sed -i 's/num_dataloader_workers: 2/num_dataloader_workers: 1/' "$yaml_dst"
            python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            sleep 10
            log "[$pos] Fix applied. Retrying..."
            continue
        fi

        # Disk full → 清上一個 position 的 preprocessed
        if grep -q "No space left on device" "$train_log" 2>/dev/null; then
            log "[$pos] DIAGNOSIS: Disk full — cleaning previous preprocessed dirs..."
            case "$pos" in
                blow_job) rm -rf "$WORK/lift_clothes_preprocessed_v2" && log "Deleted lift_clothes_preprocessed_v2" ;;
                handjob)  rm -rf "$WORK/blow_job_preprocessed_v2" && log "Deleted blow_job_preprocessed_v2" ;;
            esac
            log "[$pos] Fix applied. Retrying..."
            continue
        fi

        # initial_step >= target_steps → yaml steps 設定錯誤
        if grep -q "initial_step.*>=.*target_steps\|already reached\|already completed" "$train_log" 2>/dev/null; then
            log "[$pos] DIAGNOSIS: initial_step >= target_steps — resetting load_checkpoint to null..."
            sed -i 's/load_checkpoint:.*/load_checkpoint: null/' "$yaml_dst"
            log "[$pos] Fix applied. Retrying..."
            continue
        fi

        log "[$pos] Unknown error. No auto-fix available."
        log "[$pos] Last 30 lines of training log:"
        tail -30 "$train_log" | tee -a "$LOG"
        die "[$pos] Training failed with unrecognized error after $attempt attempt(s)."
    done

    die "[$pos] Training failed after $max_retries attempts."
}

# ── 驗證 checkpoints ────────────────────────────────────────────────────────
verify_checkpoints() {
    local pos=$1
    local output_dir="$WORK/${pos}_output_v2"
    local ckpt_count
    ckpt_count=$(find "$output_dir" -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
    if [ "$ckpt_count" -eq 0 ]; then
        log "[$pos] WARNING: No checkpoints found in $output_dir"
    else
        log "[$pos] Checkpoints ($ckpt_count files):"
        find "$output_dir" -name "*.safetensors" | sort | tee -a "$LOG"
    fi
}

# ── Main：依序訓練三個 position ──────────────────────────────────────────────
declare -A YAML_MAP=(
    ["lift_clothes"]="lift_clothes_lora_v2.yaml"
    ["blow_job"]="blow_job_lora_v2.yaml"
    ["handjob"]="handjob_lora.yaml"
)
POSITIONS=("lift_clothes" "blow_job" "handjob")

for pos in "${POSITIONS[@]}"; do
    yaml="${YAML_MAP[$pos]}"

    log "=============================="
    log "  Starting: $pos"
    log "  YAML: $yaml"
    log "=============================="

    preprocess_position "$pos"
    train_position "$pos" "$yaml"
    verify_checkpoints "$pos"

    log "[$pos] Done at $(date)"
    log ""
done

log "============================================"
log "  ALL TRAINING COMPLETE"
log "  lift_clothes, blow_job, handjob"
log "  Checkpoints ready for download."
log "============================================"
