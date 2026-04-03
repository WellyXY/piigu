# LoRA Retrain — lift_clothes / blow_job / handjob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重訓三個 position LoRA，一鍵 fire-and-forget，自動依序完成 lift_clothes → blow_job → handjob，含自動錯誤診斷修復。

**Architecture:** 本機建好 3 個 yaml + 2 個腳本，commit 後 SCP 上傳到 pod，本機起一個 `nohup bash run_all_training.sh`。`run_all_training.sh` 在本機完成上傳後，SSH 到 pod 啟動 `train_all_pod.sh` 作為背景 nohup 進程。`train_all_pod.sh` 在 pod 上依序跑三個 position，每個 position 完成後自動下載 checkpoints 回本機。

**Tech Stack:** bash, Python 3, ltx-trainer (`python3 scripts/process_dataset.py`, `python3 scripts/train.py`), SSH/SCP

---

## File Structure

```
lora_training/
├── lift_clothes_lora_v2.yaml     # 新建：first_frame_conditioning_p=0.5, steps=1500
├── blow_job_lora_v2.yaml         # 新建：同上
├── handjob_lora.yaml             # 新建：同上
├── prepare_dataset_v2.py         # 新建：通用 JSONL 生成，接受 --position --caption
├── train_all_pod.sh              # 新建：pod 上執行的 master trainer（含錯誤修復）
└── run_all_training.sh           # 新建：本機執行，SCP 上傳 + 啟動 pod 訓練
```

---

## Task 1：建立三個 YAML 配置

**Files:**
- Create: `lora_training/lift_clothes_lora_v2.yaml`
- Create: `lora_training/blow_job_lora_v2.yaml`
- Create: `lora_training/handjob_lora.yaml`

- [ ] **Step 1: 建立 lift_clothes_lora_v2.yaml**

```yaml
# lora_training/lift_clothes_lora_v2.yaml
model:
  model_path: "/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors"
  text_encoder_path: "/workspace/gemma_configs"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.5
  with_audio: false

optimization:
  learning_rate: 1e-4
  steps: 1500
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "linear"
  scheduler_params: {}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"
  load_text_encoder_in_8bit: true

data:
  preprocessed_data_root: "/workspace/lora_training/lift_clothes_preprocessed_v2"
  num_dataloader_workers: 2

validation:
  prompts:
    - "A female lifts her shirt to reveal her breasts. She cups and jiggles them with both hands. Her facial expression is neutral, and her lips are slightly parted. The pose is front view, and the motion level is moderate. The camera is static with a medium shot. The performance is suggestive and moderately paced. --lift_clothes"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 768, 249]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: 250
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: 250
  keep_last_n: 6
  precision: "bfloat16"

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {}

hub:
  push_to_hub: false
  hub_model_id: null

wandb:
  enabled: false

seed: 42
output_dir: "/workspace/lora_training/lift_clothes_output_v2"
```

- [ ] **Step 2: 建立 blow_job_lora_v2.yaml**

```yaml
# lora_training/blow_job_lora_v2.yaml
model:
  model_path: "/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors"
  text_encoder_path: "/workspace/gemma_configs"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.5
  with_audio: false

optimization:
  learning_rate: 1e-4
  steps: 1500
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "linear"
  scheduler_params: {}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"
  load_text_encoder_in_8bit: true

data:
  preprocessed_data_root: "/workspace/lora_training/blow_job_preprocessed_v2"
  num_dataloader_workers: 2

validation:
  prompts:
    - "A long-haired woman, facing the camera, holding a man's penis, performing a blow job. She slowly takes the entire penis completely into her mouth, fully submerging it until her lips press against the base of the penis and lightly touch the testicles, with the penis fully accommodated in her throat, and repeatedly moves it in and out with a steady, fluid rhythm multiple times. Please ensure the stability of the face and the object, and present more refined details."
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 768, 249]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: 250
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: 250
  keep_last_n: 6
  precision: "bfloat16"

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {}

hub:
  push_to_hub: false
  hub_model_id: null

wandb:
  enabled: false

seed: 42
output_dir: "/workspace/lora_training/blow_job_output_v2"
```

- [ ] **Step 3: 建立 handjob_lora.yaml**

```yaml
# lora_training/handjob_lora.yaml
model:
  model_path: "/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors"
  text_encoder_path: "/workspace/gemma_configs"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.5
  with_audio: false

optimization:
  learning_rate: 1e-4
  steps: 1500
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "linear"
  scheduler_params: {}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"
  load_text_encoder_in_8bit: true

data:
  preprocessed_data_root: "/workspace/lora_training/handjob_preprocessed"
  num_dataloader_workers: 2

validation:
  prompts:
    - "A female is seen in a close-up shot, sitting and stroking a man's penis with her hands. The woman is tilting her head while looking at the camera. She uses her hand to grip and move up and down the shaft. The motion is slow and rhythmic. --handjob"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 768, 249]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: 250
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: 250
  keep_last_n: 6
  precision: "bfloat16"

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {}

hub:
  push_to_hub: false
  hub_model_id: null

wandb:
  enabled: false

seed: 42
output_dir: "/workspace/lora_training/handjob_output"
```

- [ ] **Step 4: Commit**

```bash
cd "/Users/welly/Parrot API"
git add lora_training/lift_clothes_lora_v2.yaml lora_training/blow_job_lora_v2.yaml lora_training/handjob_lora.yaml
git commit -m "feat: add v2 LoRA configs — first_frame_conditioning_p=0.5, steps=1500, interval=250"
```

---

## Task 2：建立 prepare_dataset_v2.py（通用版）

**Files:**
- Create: `lora_training/prepare_dataset_v2.py`

- [ ] **Step 1: 建立 prepare_dataset_v2.py**

```python
#!/usr/bin/env python3
"""
通用 JSONL dataset manifest 生成器。
用法（在 pod 上執行）：
  python3 prepare_dataset_v2.py --position lift_clothes
  python3 prepare_dataset_v2.py --position blow_job
  python3 prepare_dataset_v2.py --position handjob
"""
import argparse
import json
import os

CAPTIONS = {
    "lift_clothes": (
        "A female lifts her shirt to reveal her breasts. "
        "She cups and jiggles them with both hands. "
        "Her facial expression is neutral, and her lips are slightly parted. "
        "The pose is front view, and the motion level is moderate. "
        "The camera is static with a medium shot. "
        "The performance is suggestive and moderately paced. --lift_clothes"
    ),
    "blow_job": (
        "A long-haired woman, facing the camera, holding a man's penis, performing a blow job. "
        "She slowly takes the entire penis completely into her mouth, fully submerging it until her lips "
        "press against the base of the penis and lightly touch the testicles, with the penis fully "
        "accommodated in her throat, and repeatedly moves it in and out with a steady, fluid rhythm "
        "multiple times. Please ensure the stability of the face and the object, and present more "
        "refined details."
    ),
    "handjob": (
        "A female is seen in a close-up shot, sitting and stroking a man's penis with her hands. "
        "The woman is tilting her head while looking at the camera. "
        "She uses her hand to grip and move up and down the shaft. "
        "The motion is slow and rhythmic. --handjob"
    ),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", required=True, choices=list(CAPTIONS.keys()))
    args = parser.parse_args()

    pos = args.position
    video_dir = f"/workspace/lora_training/{pos}_videos"
    out_file = f"/workspace/lora_training/{pos}_dataset.jsonl"
    caption = CAPTIONS[pos]

    if not os.path.isdir(video_dir):
        raise SystemExit(f"ERROR: Video dir not found: {video_dir}")

    rows = []
    for fname in sorted(os.listdir(video_dir)):
        if fname.lower().endswith(".mp4"):
            rows.append({"media_path": os.path.join(video_dir, fname), "caption": caption})

    if not rows:
        raise SystemExit(f"ERROR: No .mp4 files found in {video_dir}")

    with open(out_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"[prepare_dataset_v2] {pos}: wrote {len(rows)} rows to {out_file}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add lora_training/prepare_dataset_v2.py
git commit -m "feat: add generalized prepare_dataset_v2.py"
```

---

## Task 3：建立 train_all_pod.sh（pod 上執行的 master trainer）

**Files:**
- Create: `lora_training/train_all_pod.sh`

這個腳本在 pod 上執行，負責依序訓練三個 position，含自動錯誤診斷修復。

- [ ] **Step 1: 建立 train_all_pod.sh**

```bash
#!/usr/bin/env bash
# =============================================================================
# Master LoRA Trainer — lift_clothes → blow_job → handjob
# 在 pod 上執行：nohup bash /workspace/lora_training/train_all_pod.sh > /workspace/logs/train_all.log 2>&1 &
# =============================================================================
set -uo pipefail

TRAINER=/workspace/ltx2_repo/packages/ltx-trainer
WORK=/workspace/lora_training
MODEL=/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors
GEMMA=/workspace/gemma_configs
LOG=/workspace/logs/train_all.log
LOCAL_HOST="91.199.227.82"
LOCAL_PORT="11710"

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
        die "[$pos] No latents generated. Check video format."
    fi
}

# ── 通用 train 函數（含 retry）──────────────────────────────────────────────
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

        local train_log="$WORK/${pos}_train_attempt${attempt}.log"
        cd "$TRAINER"
        python3 scripts/train.py "$yaml_dst" 2>&1 | tee "$train_log" | tee -a "$LOG"
        local exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            log "[$pos] Training completed successfully."
            return 0
        fi

        log "[$pos] Training failed (exit $exit_code). Diagnosing..."

        # ── 診斷 + 修復 ──────────────────────────────────────────────────
        if grep -q "num_samples=0\|No samples found\|Empty dataset" "$train_log" 2>/dev/null; then
            log "[$pos] ERROR: num_samples=0. Re-running conditions fix and retrying..."
            local preproc_dir="$WORK/${pos}_preprocessed_v2"
            local video_dir="$WORK/${pos}_videos"
            mkdir -p "$preproc_dir/conditions/${pos}_videos"
            find "$video_dir" -name "*.pt" -exec cp {} "$preproc_dir/conditions/${pos}_videos/" \; 2>/dev/null || true
            continue
        fi

        if grep -q "out of memory\|CUDA out of memory\|OOM" "$train_log" 2>/dev/null; then
            log "[$pos] ERROR: CUDA OOM. Reducing num_dataloader_workers to 1 and retrying..."
            sed -i 's/num_dataloader_workers: 2/num_dataloader_workers: 1/' "$yaml_dst"
            python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            sleep 10
            continue
        fi

        if grep -q "No space left on device" "$train_log" 2>/dev/null; then
            log "[$pos] ERROR: Disk full. Cleaning up previous preprocessed dirs..."
            # 刪除上一個 position 的 preprocessed（最大的目錄）
            case "$pos" in
                blow_job)   rm -rf "$WORK/lift_clothes_preprocessed_v2" && log "Deleted lift_clothes_preprocessed_v2" ;;
                handjob)    rm -rf "$WORK/blow_job_preprocessed_v2" && log "Deleted blow_job_preprocessed_v2" ;;
            esac
            continue
        fi

        if grep -q "initial_step.*>=.*target_steps\|already reached" "$train_log" 2>/dev/null; then
            log "[$pos] ERROR: initial_step >= target_steps. Resetting load_checkpoint to null..."
            sed -i 's/load_checkpoint:.*/load_checkpoint: null/' "$yaml_dst"
            continue
        fi

        log "[$pos] Unknown error. No auto-fix available. Giving up after $attempt attempts."
        log "[$pos] Last 20 lines of training log:"
        tail -20 "$train_log" | tee -a "$LOG"
        die "[$pos] Training failed with unrecognized error."
    done

    die "[$pos] Training failed after $max_retries attempts."
}

# ── 驗證 checkpoints ────────────────────────────────────────────────────────
verify_checkpoints() {
    local pos=$1
    local output_dir="$WORK/${pos}_output_v2"
    local ckpt_count
    ckpt_count=$(find "$output_dir" -name "*.safetensors" 2>/dev/null | wc -l)
    if [ "$ckpt_count" -eq 0 ]; then
        log "[$pos] WARNING: No checkpoints found in $output_dir"
    else
        log "[$pos] Checkpoints: $ckpt_count files"
        find "$output_dir" -name "*.safetensors" | sort | tee -a "$LOG"
    fi
}

# ── Main：依序訓練三個 position ──────────────────────────────────────────────
POSITIONS=("lift_clothes:lift_clothes_lora_v2.yaml" "blow_job:blow_job_lora_v2.yaml" "handjob:handjob_lora.yaml")

for entry in "${POSITIONS[@]}"; do
    pos="${entry%%:*}"
    yaml="${entry##*:}"

    log "=============================="
    log "  Starting: $pos"
    log "=============================="

    preprocess_position "$pos"
    train_position "$pos" "$yaml"
    verify_checkpoints "$pos"

    log "[$pos] Done. $(date)"
    log ""
done

log "=============================="
log "  ALL TRAINING COMPLETE"
log "  lift_clothes, blow_job, handjob"
log "=============================="
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add lora_training/train_all_pod.sh
git commit -m "feat: add train_all_pod.sh — sequential training with auto error recovery"
```

---

## Task 4：建立 run_all_training.sh（本機執行腳本）

**Files:**
- Create: `lora_training/run_all_training.sh`

- [ ] **Step 1: 建立 run_all_training.sh**

```bash
#!/usr/bin/env bash
# =============================================================================
# 本機執行：上傳 + 啟動 pod 訓練
# 用法：nohup bash "/Users/welly/Parrot API/lora_training/run_all_training.sh" > ~/train_upload.log 2>&1 &
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

log "=== LoRA Retrain — Upload & Launch ==="

# ── Step 1: 上傳腳本 + YAML ────────────────────────────────────────────────
log "Uploading scripts and YAML configs..."
$SSH "mkdir -p /workspace/lora_training /workspace/logs"
$SCP \
    "$LORA/lift_clothes_lora_v2.yaml" \
    "$LORA/blow_job_lora_v2.yaml" \
    "$LORA/handjob_lora.yaml" \
    "$LORA/prepare_dataset_v2.py" \
    "$LORA/train_all_pod.sh" \
    "${POD}:/workspace/lora_training/"
log "Scripts uploaded."

# ── Step 2: 上傳影片 ────────────────────────────────────────────────────────
for pos in lift_clothes blow_job handjob; do
    VIDEO_DIR="$WORK/training data/$pos"
    MP4_COUNT=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
    log "Uploading $pos videos ($MP4_COUNT files)..."
    $SSH "mkdir -p /workspace/lora_training/${pos}_videos"
    $SCP "$VIDEO_DIR/"*.mp4 "${POD}:/workspace/lora_training/${pos}_videos/"
    log "  $pos upload done."
done

# ── Step 3: 啟動 pod 訓練（背景 nohup）────────────────────────────────────
log "Starting training on pod (nohup background)..."
REMOTE_PID=$($SSH "nohup bash /workspace/lora_training/train_all_pod.sh >> /workspace/logs/train_all.log 2>&1 & echo \$!")
log "Training started on pod. PID: $REMOTE_PID"
log ""
log "Monitor with:"
log "  ssh -i $SSH_KEY $POD -p $POD_PORT 'tail -f /workspace/logs/train_all.log'"
log ""
log "Check progress:"
log "  ssh -i $SSH_KEY $POD -p $POD_PORT 'ls /workspace/lora_training/lift_clothes_output_v2/checkpoints/ 2>/dev/null | tail -5'"
log ""
log "Upload complete. You can close this terminal."
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/welly/Parrot API"
git add lora_training/run_all_training.sh
git commit -m "feat: add run_all_training.sh — local upload + launch pod training"
```

---

## Task 5：執行

- [ ] **Step 1: 確認 pod 在線**

```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 'nvidia-smi --query-gpu=memory.free --format=csv,noheader'
# Expected: ~80000 MiB（推理 server 需先停止，train_all_pod.sh 會自動停）
```

- [ ] **Step 2: 本機起 nohup 執行**

```bash
cd "/Users/welly/Parrot API"
nohup bash "lora_training/run_all_training.sh" > ~/lora_upload.log 2>&1 &
echo "Upload PID: $!"
```

- [ ] **Step 3: 確認上傳完成 + training 已啟動**

```bash
tail -f ~/lora_upload.log
# Expected 最後幾行:
#   [HH:MM:SS] handjob upload done.
#   [HH:MM:SS] Training started on pod. PID: XXXXXX
#   [HH:MM:SS] Upload complete. You can close this terminal.
```

- [ ] **Step 4: 監控訓練進度**

```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 'tail -30 /workspace/logs/train_all.log'
```

看到 `ALL TRAINING COMPLETE` 表示三個都完成。

- [ ] **Step 5: 下載所有 checkpoints**

```bash
for pos in lift_clothes blow_job handjob; do
    mkdir -p "/Users/welly/Parrot API/lora_training/${pos}_checkpoints_v2"
    scp -i ~/.ssh/id_ed25519 -P 11710 \
        "root@91.199.227.82:/workspace/lora_training/${pos}_output_v2/checkpoints/*.safetensors" \
        "/Users/welly/Parrot API/lora_training/${pos}_checkpoints_v2/" \
        2>/dev/null || echo "$pos: no checkpoints yet"
done
ls -lh "/Users/welly/Parrot API/lora_training/"*_checkpoints_v2/
```

---

## Task 6：部署最佳 checkpoint（訓完後手動）

看完各 step 的 checkpoint，用 I2V 測試選最好的再部署。

- [ ] **Step 1: 部署選定的 checkpoint（以 lift_clothes step 1000 為例）**

```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 '
cp /workspace/lora_training/lift_clothes_output_v2/checkpoints/lora_weights_step_01000.safetensors \
   /workspace/models/loras/lift_clothes.safetensors
echo "Deployed: $(ls -lh /workspace/models/loras/lift_clothes.safetensors)"
'
```

- [ ] **Step 2: 重啟 inference server**

```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
    'nohup bash /workspace/start_server.sh >> /workspace/logs/server.log 2>&1 &'
```

- [ ] **Step 3: 等 server 就緒（6-7 分鐘）後確認 health**

```bash
sleep 420 && ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
    'curl -s http://localhost:8000/health'
# Expected: {"status":"ok"}
```
