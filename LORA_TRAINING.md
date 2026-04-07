# LTX-2.3 Position LoRA 訓練完整指南

> 最後更新：2026-04-07（基於 handjob_v2 + lift_clothes_v2 實戰確認）

---

## 最重要的教訓（血淚）

### 1. 必須用 DEV 模型訓練，不能用 Distilled

- **訓練用**：`ltx-2.3-22b-dev.safetensors`（46GB）
- **推理用**：`ltx-2.3-22b-distilled.safetensors`（46GB）
- Distilled 模型加了蒸餾 noise schedule，用它訓出來的 LoRA 效果差

### 2. 唯一可用工具：官方 ltx-trainer

- **musubi-tuner 不支援 LTX 2.3**（只支援 HunyuanVideo、Wan、FramePack、FLUX）
- `ltx2_train_network.py` 是 AI 幻想出來的假腳本，不存在
- 唯一可用：`/workspace/ltx2_repo/packages/ltx-trainer`

### 3. resolution-buckets frame 數必須覆蓋完整動作

- ❌ `512x768x25`（1 秒）→ 只學到「人站著」
- ✅ `512x768x121`（5 秒）→ 短影片用
- ✅ `512x768x249`（10 秒）→ 長影片用，效果最好

**短於 resolution-buckets 的影片會被默默跳過。** 混合長度的解法：

```bash
--resolution-buckets "512x768x249;512x768x121"
```

### 4. Conditions Path Bug（必踩）

`process_dataset.py` pathlib bug：當 `media_path` 為絕對路徑時，conditions (.pt) 存到影片目錄而非 preprocessed 目錄。

**症狀**：`ValueError: num_samples=0`

**修復**：
```bash
mkdir -p /workspace/lora_training/<pos>_preprocessed/conditions/<pos>_videos
cp /workspace/lora_training/<pos>_videos/*.pt \
   /workspace/lora_training/<pos>_preprocessed/conditions/<pos>_videos/
```

### 5. first_frame_conditioning_p — 動作 LoRA 用 0.35

- ✅ **0.35** → handjob_v2 實測有效值
- ✅ 0.3（偏向 T2V 泛化）
- ❌ 0.9 → 過度依賴第一幀，泛化差
- ❌ 0.5 → 踩過的坑

### 6. batch_size 與影片長度的關係（H100 80GB）

| 影片長度 | frames | batch_size | VRAM |
|---------|--------|-----------|------|
| 5-6 秒  | 121    | **2** 可行 | ~35 GB |
| 10 秒   | 249    | **1** 只能 | ~62 GB |

- batch=2 需要先修好 squeeze bug（見下方）
- **249 frames 必加 env var**：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，否則 OOM

### 7. 影片長度 vs 品質

- 80 支 5-6 秒影片（121 frames）訓出來比 29 支 10 秒影片（249 frames）**更差**
- 原因：**較長影片讓 model 學到更完整的動作時序結構**，即使 inference 只跑 121 frames
- 建議：盡量用 **8 秒以上**的訓練影片

### 8. batch=2 的 squeeze bug（embeddings_connector.py）

```python
# /workspace/ltx2_repo/packages/ltx-core/src/ltx_core/text_encoders/gemma/embeddings_connector.py line 147
# 錯誤（batch>1 時 IndexError）：
non_zero_hidden_states = hidden_states[:, attention_mask_binary.squeeze().bool(), :]
# 修復：
non_zero_hidden_states = hidden_states[:, attention_mask_binary[0].bool(), :]
```

---

## 確認有效的訓練參數（2026-04-06）

### YAML 配置模板

```yaml
model:
  model_path: "/workspace/models/ltx23/ltx-2.3-22b-dev.safetensors"  # 必須是 DEV
  text_encoder_path: "/workspace/gemma_configs"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: 64
  alpha: 64
  dropout: 0.0
  target_modules: [to_k, to_q, to_v, to_out.0]

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.35
  with_audio: false

optimization:
  learning_rate: 1e-4
  steps: 1000
  batch_size: 1        # 121 frames 可用 2；249 frames 只能 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw"
  scheduler_type: "linear"
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: null
  load_text_encoder_in_8bit: true

data:
  preprocessed_data_root: "/workspace/lora_training/<position>_v2_preprocessed"
  num_dataloader_workers: 2

validation:
  prompts:
    - "<caption> --<position>"
  images: null
  video_dims: [512, 768, 121]    # 121 或 249，對應訓練數據長度
  frame_rate: 25.0
  inference_steps: 30
  interval: 250
  seed: 42
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: 250
  keep_last_n: 4
  precision: "bfloat16"

output_dir: "/workspace/lora_training/<position>_v2_output"
seed: 42
```

### Checkpoint 大小說明

- 中間 checkpoint（250/500/750 步）：~817MB（含 optimizer state）
- 最終 checkpoint（1000 步）：~192MB（純 LoRA weights）
- 1000 步 checkpoint 若損壞，用 750 步的（格式相同，可正常推理）

---

## 訓練流程（Step by Step）

### 前置條件

- 訓練影片放 local：`/Users/welly/Parrot API/training data/<position>/`
- Pod SSH：`ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170`
- 推理 server 必須先關（VRAM 衝突）

### Step 1：準備 jsonl + yaml（local）

```
/Users/welly/Parrot API/lora_training/
├── <position>_v2_lora.yaml
└── <position>_v2_dataset.jsonl
```

### Step 2：上傳影片和腳本

```bash
# 上傳 yaml
scp -i ~/.ssh/id_ed25519 -P 11170 \
  "/Users/welly/Parrot API/lora_training/<position>_v2_lora.yaml" \
  root@91.199.227.82:/workspace/ltx2_repo/packages/ltx-trainer/configs/

# 建目錄 + 上傳影片
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 \
  'mkdir -p /workspace/lora_training/<position>_v2_videos'

scp -i ~/.ssh/id_ed25519 -P 11170 \
  "/Users/welly/Parrot API/training data/<position>/*.mp4" \
  root@91.199.227.82:/workspace/lora_training/<position>_v2_videos/
```

### Step 3：停推理 server + Preprocess

```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 '
pkill -f server.py; sleep 3
python3 /workspace/ltx2_repo/packages/ltx-trainer/scripts/process_dataset.py \
  /workspace/lora_training/<position>_v2_dataset.jsonl \
  --resolution-buckets "512x768x249" \
  --model-path /workspace/models/ltx23/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /workspace/gemma_configs \
  --output-dir /workspace/lora_training/<position>_v2_preprocessed \
  --batch-size 1 --device cuda --load-text-encoder-in-8bit \
  > /workspace/logs/preprocess_<position>_v2.log 2>&1
echo "Done: $?"
'
```

### Step 4：執行訓練

```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 '
cd /workspace/ltx2_repo/packages/ltx-trainer
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 scripts/train.py configs/<position>_v2_lora.yaml \
  > /workspace/logs/train_<position>_v2.log 2>&1 &
echo "PID: $!"
'
```

### Step 5：監控

```bash
# log
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 \
  'tail -20 /workspace/logs/train_<position>_v2.log'

# GPU
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 \
  'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'

# checkpoints
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 \
  'ls /workspace/lora_training/<position>_v2_output/checkpoints/'
```

### Step 6：部署

```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 '
cp /workspace/lora_training/<position>_v2_output/checkpoints/lora_weights_step_01000.safetensors \
   /workspace/models/loras/<position>.safetensors
'

# 更新 config
scp -P 11170 -i ~/.ssh/id_ed25519 \
  "/Users/welly/Parrot API/parrot-api/config.py" \
  root@91.199.227.82:/workspace/parrot-api/config.py

# 重啟 server（LoRA delta 在啟動時預計算，必須重啟）
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11170 '
pkill -f server.py; sleep 3
setsid bash /workspace/start_server.sh > /workspace/logs/server.log 2>&1 &
'
# 等 6-7 分鐘
```

---

## 訓練速度參考（H100 80GB，rank=64，adamw，gradient_checkpointing）

| Frame 數 | batch | VRAM | 每步時間 | 1000 steps 總時間 |
|---------|-------|------|---------|-----------------|
| 121 frames | 2 | ~35 GB | ~6-8s | ~1.5-2 hrs |
| 249 frames | 1 | ~62 GB | ~10-15s | ~3-4 hrs |

---

## per-position LoRA 權重（config.py）

```python
POSITION_LORA_WEIGHTS = {
    "blow_job":        1.2,
    "cowgirl":         0.8,
    "doggy":           0.8,
    "handjob":         0.6,   # 自訓練；同時 stack blow_job w=0.6（POSITION_SECONDARY）
    "lift_clothes":    1.2,   # 自訓練；前端 nsfw/motion 預設 0.0
    "masturbation":    0.8,
    "missionary":      0.8,
    "reverse_cowgirl": 0.8,
    # tit_job: 無 position LoRA（不在此 dict），pos_w=0.0，用 handjob prompt
}

POSITION_SECONDARY: dict[str, tuple[str, float]] = {
    "handjob": ("blow_job", 0.6),
}
```

調整原則：動作太弱 → 往上（1.2, 1.5）；artifact / 臉變形 → 往下降。

NSFW LoRA = 1.0，Motion LoRA = 0.7（全 position 統一，lift_clothes 前端預設 0.0）。

---

## Caption 規範

1. 主要動作描述
2. 面部/身體細節
3. 攝影機角度
4. 動作節奏
5. 觸發詞 `--<position>` 放最後

Audio dirty talk 透過 API `audio_description` 欄位傳入，不寫在 caption 裡。

---

## 訓練完成後測試計劃

1. **Validation video** — 下載各 checkpoint sample，比較哪步最好
2. **隔離測試** — `nsfw=0, motion=0`，只掛 position LoRA，確認動作有沒有學到
3. **真實圖測試** — 用 `/Users/welly/Parrot API/testing data/<position>/` 的測試圖
4. **調整 weight** — 從 0.8 開始，視效果調整
