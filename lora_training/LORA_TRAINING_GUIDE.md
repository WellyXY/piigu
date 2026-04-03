---
name: LoRA Training — Complete Guide
description: LTX-2.3 position LoRA 完整訓練流程、已知坑、測試計劃、per-position 權重設定
type: project
---

# LTX-2.3 Position LoRA 訓練完整指南

## Community 研究總結（Action LoRA 專項）

### first_frame_conditioning_p 是最大坑

官方推薦 **0.1~0.5**，不是 0.9。

- `0.9` = 90% 的 training steps 在學「從這張 frame 繼續生成」，不是在學動作 pattern
- 結果是 LoRA 在背訓練影片的 frame，不是學「掀衣服」這個 action 本身
- **Action LoRA 用 `0.5`**（官方 AV config 的值），兼顧 I2V 可用性和 action generalization

### 影片數量與多樣性

- Motion LoRA 最少 **40-60 部**，20 部在底線邊緣
- 最重要的是**多樣性**：不同的人、不同衣服、不同角度執行同一個動作
- 如果 20 部都是相似的人/背景，LoRA 學的是「那個外觀」不是「那個動作」

### 不要跑太多 steps

- 20 部影片跑 2000 steps 容易 overfit
- 建議 **1000-1500 steps**，每 250 steps 一個 checkpoint，選最好的那個
- 2000 steps 的 checkpoint 不一定比 1000 steps 的好

### AI 生成影片當訓練數據的問題

- AI 生成影片的手部、布料物理通常不準確，LoRA 會學到這些 artifact
- 訓練完的 LoRA 會複製出 AI 影片本身的 artifact（手指變形等）
- AI 影片底層風格相似（同個 base model），多樣性是假的
- **最好用真實影片**，或至少確保 AI 影片手部清晰、動作流暢

### 手指/胸部 artifact 的成因

1. Base model 本身的弱點（官方 negative prompt 就有 "fused fingers"）
2. 訓練數據本身有 artifact 被 LoRA 學進去
3. `first_frame_conditioning_p` 太高，overfit 到特定 starting frame
4. 訓練影片 reference frame 與第一幀對不齊

### 評估 checkpoint 要用 I2V，不是 T2V validation video

- training 期間生成的 validation video 是 T2V（`images: null`）
- 但 inference 是 I2V，兩者效果不一樣
- 要比較 checkpoint 品質必須用 I2V 跑（見下方測試計劃 Phase 2）

---

## 最重要的教訓（血淚）

### 1. resolution-buckets frame 數必須覆蓋完整動作
**這是最常見的錯誤。** `--resolution-buckets` 的 frame 數如果太短，LoRA 只會學到動作開始前的靜態畫面，不會學到動作本身。

- ❌ `512x768x25`（1 秒）→ 只學到「人站著」
- ✅ `512x768x249`（10 秒）→ 學到完整動作

訓練數據通常是 151 frames（~6s）或 301 frames（~12s）。用 249 frames 基本覆蓋。

### 2. 短於 resolution-buckets 的影片會被跳過
如果某些影片短於設定的 frame 數，process_dataset.py 會默默跳過。解法：

**Option A：用多個 bucket（分號分隔）**
```bash
--resolution-buckets "512x768x249;512x768x121"
```
→ 長影片用 249 frames，短影片（≥121 frames）用 121 frames

**Option B：分開 preprocess 再合并**
```bash
# 短影片單獨跑 121 frames preprocess
python3 scripts/process_dataset.py short.jsonl \
    --resolution-buckets "512x768x121" \
    --output-dir /workspace/lora_training/<pos>_short_preprocessed

# 手動把 latents/conditions 合并到主目錄
cp -r /workspace/lora_training/<pos>_short_preprocessed/latents/. \
      /workspace/lora_training/<pos>_preprocessed/latents/
cp -r /workspace/lora_training/<pos>_short_preprocessed/conditions/. \
      /workspace/lora_training/<pos>_preprocessed/conditions/
```

### 3. Conditions Path Bug（必踩）
`process_dataset.py` pathlib bug：當 `media_path` 為絕對路徑時，conditions (.pt) 存到影片目錄而非 preprocessed 目錄。

**症狀**：`ValueError: num_samples=0`

**檢查**：
```bash
ls /workspace/lora_training/<pos>_videos/*.pt    # 如果有 .pt 就是 bug 觸發
ls /workspace/lora_training/<pos>_preprocessed/conditions/  # 應該有 .pt 才對
```

**修復**：
```bash
mkdir -p /workspace/lora_training/<pos>_preprocessed/conditions/<pos>_videos
cp /workspace/lora_training/<pos>_videos/*.pt \
   /workspace/lora_training/<pos>_preprocessed/conditions/<pos>_videos/

# 確認 latents 和 conditions 數量一致
ls /workspace/lora_training/<pos>_preprocessed/latents/<pos>_videos/ | wc -l
ls /workspace/lora_training/<pos>_preprocessed/conditions/<pos>_videos/ | wc -l
# 若不一致，刪除多餘的 conditions .pt（沒有對應 latent 的）
```

---

## 訓練流程（Step by Step）

### 前置條件
- 訓練影片放在 local：`/Users/welly/Parrot API/training data/<position>/`
- Pod SSH：`ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710`
- 推理 server 必須先關閉（`pkill -f "python3 server.py"`）才能訓練（VRAM 衝突）

### Step 1：準備腳本（local）
```
/Users/welly/Parrot API/lora_training/
├── run_training_<position>.sh
├── prepare_dataset_<position>.py
└── <position>_lora.yaml
```

### Step 2：SCP 上傳到 pod
```bash
# 腳本
scp -i ~/.ssh/id_ed25519 -P 11710 \
  "/Users/welly/Parrot API/lora_training/<position>_lora.yaml" \
  "/Users/welly/Parrot API/lora_training/run_training_<position>.sh" \
  "/Users/welly/Parrot API/lora_training/prepare_dataset_<position>.py" \
  root@91.199.227.82:/workspace/lora_training/

# 訓練影片
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'mkdir -p /workspace/lora_training/<position>_videos'
scp -i ~/.ssh/id_ed25519 -P 11710 \
  "/Users/welly/Parrot API/training data/<position>/*.mp4" \
  root@91.199.227.82:/workspace/lora_training/<position>_videos/
```

### Step 3：停推理 server
```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'pkill -f "python3 server.py"; sleep 3; nvidia-smi --query-gpu=memory.free --format=csv,noheader'
# 應顯示 ~80GB free
```

### Step 4：執行訓練
```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'nohup bash /workspace/lora_training/run_training_<position>.sh \
   > /workspace/logs/lora_train_<position>.log 2>&1 & echo $!'
```

### Step 5：監控
```bash
# 看 log
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'tail -20 /workspace/logs/lora_train_<position>.log'

# 看 VRAM（249 frames 預計 ~40GB）
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'nvidia-smi --query-gpu=memory.used --format=csv,noheader'

# 看 checkpoints（每 500 steps 一個）
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'ls /workspace/lora_training/<position>_output/checkpoints/'
```

### Step 5.5：下載所有 checkpoints 到 local（⚠️ 必做）
> **重要：pod 隨時可能被終止/更換，checkpoints 只在 network volume 上。**
> **每次換 pod 前，或訓練完成後，立刻下載到本機備份。**

```bash
mkdir -p "/Users/welly/Parrot API/lora_training/<position>_checkpoints"

scp -i ~/.ssh/id_ed25519 -P 11710 \
  "root@91.199.227.82:/workspace/lora_training/<position>_output/checkpoints/*.safetensors" \
  "/Users/welly/Parrot API/lora_training/<position>_checkpoints/"

# 確認下載
ls -lh "/Users/welly/Parrot API/lora_training/<position>_checkpoints/"
```

同時下載 validation samples（看哪個 step 最好）：
```bash
mkdir -p "/Users/welly/Parrot API/lora_training/<position>_validation"

scp -i ~/.ssh/id_ed25519 -P 11710 \
  "root@91.199.227.82:/workspace/lora_training/<position>_output/samples/*.mp4" \
  "/Users/welly/Parrot API/lora_training/<position>_validation/" 2>/dev/null || true
```

### Step 6：修復 conditions path bug（如果遇到 num_samples=0）
（見上方 Bug 修復步驟）

### Step 7：部署最終 checkpoint
```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 '
cp /workspace/lora_training/<position>_output/checkpoints/lora_weights_step_02000.safetensors \
   /workspace/models/loras/<position>.safetensors
echo "Deployed: $(ls -lh /workspace/models/loras/<position>.safetensors)"
'
```

### Step 8：重啟推理 server
```bash
ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'nohup bash /workspace/start_server.sh >> /workspace/logs/server.log 2>&1 &'
# 等 2-3 分鐘，確認 health
sleep 180 && ssh -i ~/.ssh/id_ed25519 root@91.199.227.82 -p 11710 \
  'curl -s http://localhost:8000/health'
```

---

## YAML 配置模板

```yaml
model:
  model_path: "/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors"
  text_encoder_path: "/workspace/gemma_configs"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  target_modules: [to_k, to_q, to_v, to_out.0]

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.9   # I2V 訓練
  with_audio: false

optimization:
  learning_rate: 1e-4
  steps: 2000
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw8bit"
  scheduler_type: "linear"
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"
  load_text_encoder_in_8bit: true

data:
  preprocessed_data_root: "/workspace/lora_training/<position>_preprocessed"
  num_dataloader_workers: 2

validation:
  prompts:
    - "<full caption> --<position>"
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [512, 768, 249]        # 必須和 resolution-buckets 一致
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: 500
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: true

checkpoints:
  interval: 500
  keep_last_n: 4
  precision: "bfloat16"

seed: 42
output_dir: "/workspace/lora_training/<position>_output"
```

---

## 訓練速度參考（H100 80GB）

| Frame 數 | VRAM | 每步時間 | 2000 steps 總時間 |
|---------|------|---------|-----------------|
| 25 frames | ~32 GB | ~1s | ~34 min |
| 249 frames | ~40 GB | ~3-5s | ~2-3 hrs |

---

## 訓練完成後的測試計劃

### Phase 1：Validation video 確認（訓練後立即）
下載各 checkpoint 的 validation video：
```bash
# 下載所有 validation samples
scp -i ~/.ssh/id_ed25519 -P 11710 \
  "root@91.199.227.82:/workspace/lora_training/<position>_output/samples/*.mp4" \
  ~/Desktop/<position>_validation/
```
看 step_000500, step_001000, step_001500, step_002000 哪個效果最佳。

> ⚠️ **重要：Validation video 是 T2V（無 image conditioning）**
> yaml 裡 `images: null` → 純文字生成，沒有 starting frame 拉住原始狀態。
> Validation 看起來好 ≠ I2V inference 好。
>
> **正確的 checkpoint 比較方式是用 I2V 跑**，見 Phase 2。

### Phase 2：I2V checkpoint 比較測試（⚠️ 正確的評估方式）

**必須用 I2V 測試才能反映真實 inference 效果。** T2V validation 效果好不代表 I2V 好。

部署各 checkpoint 到 `/workspace/models/loras/<position>.safetensors`，重啟 server，各跑一次 I2V 比較：

```bash
# 部署某個 checkpoint
ssh -i ~/.ssh/id_ed25519 root@<HOST> -p <PORT> '
cp /workspace/lora_training/<position>_output/checkpoints/lora_weights_step_0XXXX.safetensors \
   /workspace/models/loras/<position>.safetensors
pkill -f server.py; sleep 3
nohup bash /workspace/start_server.sh > /workspace/logs/server.log 2>&1 &
'

# 等 server ready（~6-7 分鐘）後跑 I2V 測試
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "<full production caption>",
    "position": "<position>",
    "image_path": "/workspace/test_images/<position>.webp",
    "seed": 42,
    "enhance": false,
    "image_strength": 0.7,
    "nsfw_weight": 0.0,
    "motion_weight": 0.0
  }'
```

**測試參數建議（lift_clothes 類型的動作 LoRA）：**
- `image_strength`: 0.7（降低 image conditioning 對衣服的束縛）
- `nsfw_weight`: 0.0、`motion_weight`: 0.0（排除性行為 LoRA 的干擾）
- 用真實 production prompt，不加 `--<trigger>` 觸發詞

下載結果到本機比較各 step：
```bash
scp -i ~/.ssh/id_ed25519 -P <PORT> \
  "root@<HOST>:/workspace/outputs/<position>_*.mp4" \
  "/Users/welly/Parrot API/results/<position>_step_comparison/"
```

### Phase 3：真實測試圖
使用 `/Users/welly/Parrot API/testing data/<position>/` 的 _input.webp 圖片：
```bash
# SCP 測試圖到 pod
scp -i ~/.ssh/id_ed25519 -P 11710 \
  "/Users/welly/Parrot API/testing data/<position>/<uuid>_input.webp" \
  root@91.199.227.82:/tmp/test_<position>.webp

# 生成並下載
curl -X POST http://localhost:8000/generate \
  -d '{"image_path": "/tmp/test_<position>.webp", ...}'
```

### Phase 4：調整 position weight
從 1.0 開始，視效果調整：
- 動作太弱 → 往上加（1.2, 1.5, 1.8）
- 有 artifact / 臉變形 → 往下降

---

## per-position LoRA 權重設定（config.py）

```python
# /workspace/parrot-api/config.py

POSITION_LORA_WEIGHTS = {
    "blow_job":        1.2,
    "cowgirl":         0.8,
    "doggy":           0.8,
    "lift_clothes":    0.8,   # 1.5 太強有 artifact，0.8 效果較自然
    "masturbation":    0.8,
    "missionary":      0.8,
    "reverse_cowgirl": 0.8,
}

POSITION_NSFW_WEIGHTS = {
    # 性行為 positions 用 1.0
    # lift_clothes 設 0.0 — NSFW LoRA 對衣服動作無益
    "lift_clothes":    0.0,
    # 其他: 1.0
}

POSITION_MOTION_WEIGHTS = {
    # 性行為 positions 用 0.7
    # lift_clothes 設 0.0 — Motion LoRA 是性行為動作，會干擾衣服動作
    "lift_clothes":    0.0,
    # 其他: 0.7
}
```

修改 config.py 後必須重啟 server 才生效（config 在 server startup 時讀入）。

---

## Caption 規範

每個 position 的 caption 包含：
1. 主要動作描述
2. 面部/身體細節
3. 攝影機角度
4. 動作節奏
5. 觸發詞 `--<position>` 放在最後

範例（lift_clothes）：
```
A female lifts her shirt to reveal her breasts. She cups and jiggles them
with both hands. Her facial expression is neutral, and her lips are slightly
parted. The pose is front view, and the motion level is moderate. The camera
is static with a medium shot. The performance is suggestive and moderately
paced. --lift_clothes
```

Audio（dirty talk）透過 API `audio_description` 欄位傳入，不寫在 caption 裡。
