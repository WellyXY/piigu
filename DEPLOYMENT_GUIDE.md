# Parrot API — 部署、配置與運行策略

> 最後更新：2026-04-22（default 解析度 512×768 → 640×960 A/B 通過後採用；bf16 Gemma + eager attention patch；卸載 bitsandbytes；ENHANCE_BATCH_SIZE 32→8 避 OOM；ENHANCE_DEFLICKER 15→25 抑制臉部 flicker；base LoRA rescale 跳過 audio 層）

---

## 快速查閱表

| 操作 | 指令 |
|------|------|
| **啟動 Server** | 見 Step 5（`cd /workspace/parrot-api && nohup /workspace/venv/bin/python server.py > /workspace/parrot-api.log 2>&1 &`） |
| **確認就緒** | `curl http://localhost:8000/status` |
| **監看日誌** | `tail -f /workspace/parrot-api.log` |
| **停止 Server** | ⚠️ **不要用 `pkill -f server.py`**（會匹配到自己的 ssh shell）。用 `ps -eo pid,cmd \| awk '$0 ~ /venv\/bin\/python server\.py/ && $0 !~ /awk/ {print $1}' \| xargs -r kill -9` |
| **完整重啟** | 先 kill server（上一格）→ `/workspace/venv/bin/ray stop --force` → 重啟 server |

---

## 一、RunPod 新節點從零配置

### 前置要求

- **硬體**：H100 80GB（或同等 VRAM）
- **Network Volume**：`elwop1sm62`（lonely_amethyst_wombat），掛載路徑設為 `/workspace`
- **SSH Key**：本機 `~/.ssh/id_ed25519`
- **GitHub SSH Key**：在 pod 上產生並加到 GitHub（見 Step 2）
- **Worker .env**：本機 `/Users/welly/Parrot API/parrot-service.env`（已含所有 credentials）

---

### Step 1 — 建立 Pod

RunPod console → **New Pod** → 選 H100 → **Attach Network Volume** → 選 `elwop1sm62`，Mount Path 填 `/workspace` → Deploy。

---

### Step 2 — GitHub SSH Key

```bash
# 在 pod 上產生新 key
ssh-keygen -t ed25519 -C 'runpod-parrot' -f ~/.ssh/id_ed25519 -N ''
cat ~/.ssh/id_ed25519.pub  # 複製後貼到 GitHub → Settings → SSH keys
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Clone 兩個 repo
git clone git@github.com:WellyXY/piigu.git /workspace/parrot-api
git clone git@github.com:WellyXY/piigun.git /workspace/parrot-service
```

---

### Step 3 — 部署程式碼 + 建立 venv

從**本機**執行 `deploy.sh`（`parrot-api/deploy.sh`）：

```bash
# 在本機 parrot-api/ 目錄下
bash deploy.sh <PORT> <HOST_IP>
# 例：bash deploy.sh 10574 91.199.227.82
```

`deploy.sh` 會自動：
1. SCP `config.py`, `server.py`, `actors/`, `persistent_pipeline.py` 等到 pod
2. 在 pod 建立 `/workspace/venv`（`--system-site-packages`，保留 torch 2.4.1）
3. 安裝 pip 依賴（requirements.txt + gfpgan/facexlib/basicsr + transformers 4.52.0 + ltx-core/ltx-pipelines）
   - ⚠️ **不要安裝 bitsandbytes**：裝了會讓 Gemma 自動走 int8 path，破壞 speech 精度。bf16 Gemma 是目前唯一能穩定產生清晰 dirty talk 的路徑。
   - ⚠️ **新 pod 的 venv 可能預裝 bnb**（base image 帶入）：啟動前 `/ workspace/venv/bin/pip uninstall -y bitsandbytes` 檢查。log 看到 `bitsandbytes available — Gemma will be quantized to int8` 就是中招，必須 uninstall 後重啟。
   - **系統層：裝 ffmpeg（給 GFPGAN enhance 用）**：`apt-get update -qq && apt-get install -y ffmpeg`。沒裝會讓 `enhance=True` 失敗，log 看到 `FileNotFoundError: 'ffmpeg'`。
   - **worker 依賴（system python3，非 venv）**：`start_worker.sh` 用系統 `python3` 跑。新 pod 要裝：`pip install 'redis[asyncio]>=5.0' pydantic asyncpg boto3 httpx pillow tenacity`。沒裝 worker 啟動即 `ModuleNotFoundError`，job 卡 queue `started_at=null`。
4. 套用以下 patches：
   - **torchvision** `_meta_registrations` import 移除
   - **basicsr** `functional_tensor` → `functional` 改名
   - **patch_nms.py** torchvision NMS 修正
   - **Gemma3 autocast** `torch.autocast` → `torch.amp.autocast`（torch 2.4.1 相容性）
   - **transformers ALL_PARALLEL_STYLES**（4.52.0 + torch < 2.5 的 `None` bug 修正）
   - **Gemma3 eager attention**（必做）：在 `/workspace/ltx2_repo/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py` 的 `Gemma3Config` 建立後加入 `gemma_config._attn_implementation = "eager"`。否則 bf16 + SDPA attention 會產生 NaN hidden states → 全黑影片。

---

### Step 4 — 驗證模型

```bash
# 確認 network volume 上的模型都在
ls /workspace/models/ltx23/        # 應有 2 個 .safetensors（distilled + spatial-upscaler）
ls /workspace/models/loras/        # 應有 10 個 position .safetensors 和 2 個 base LoRA
ls /workspace/gemma_configs/       # 應有 Gemma 分片
ls /workspace/GFPGANv1.4.pth      # 應存在
```

---

### Step 4.5 — 部署 Worker .env

```bash
# 從本機 scp credentials 過去
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  "/Users/welly/Parrot API/parrot-service.env" \
  root@<HOST>:/workspace/parrot-service/.env
```

---

### Step 5 — 啟動

```bash
# 啟動 parrot-api server（log → /workspace/parrot-api.log）
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
cd /workspace/parrot-api
nohup /workspace/venv/bin/python server.py > /workspace/parrot-api.log 2>&1 &

# 啟動 gpu_worker（等 server ready 後）
set -a; source /workspace/parrot-service/.env; set +a
nohup bash /workspace/parrot-service/start_worker.sh >> /workspace/worker.log 2>&1 &
```

---

### Step 6 — 驗證就緒

```bash
# 監看啟動日誌
tail -f /workspace/parrot-api.log

# 確認狀態（server ready 後）
curl http://localhost:8000/status
```

**預期回應：**
```json
{
  "status": "ready",
  "ltx": {
    "current_position": ["cowgirl", 0.8],
    "transformer_loaded": true,
    "gemma_loaded": true,
    "device": "cuda",
    "position_loras_cached": ["blow_job", "cowgirl", "doggy", "handjob", "lift_clothes", "masturbation", "missionary", "reverse_cowgirl", "dildo", "boobs_play"]
  },
  "gfpgan": {"model": "GFPGANv1.4", "ready": true, "vram_mb": 525},
  "queue_depth": 0
}
```

**啟動時間分解：**

| 階段 | 時間 |
|------|------|
| Python + Ray 初始化 | 5-8s |
| GFPGAN actor 載入 | 10-12s |
| LTX transformer 載入（43GB from disk） | 10-30s |
| bf16 Gemma 載入 | 8-10s |
| Position LoRA delta 預計算（10 個，CPU） | 3-4min |
| **合計** | **~4-5min** |

> **注意：** Base LoRA（nsfw/motion）的 delta **不在** warmup 預計算（B@A 展開後約 71GB CPU RAM，會 OOM）。改為 on-demand GPU layer-by-layer 計算，weight 變更時才觸發（~1s/per LoRA，H100 實測）。大部分請求使用預設值→直接跳過，**零額外耗時**。

---

## 二、啟動說明

### parrot-api server（手動啟動）

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
cd /workspace/parrot-api
nohup /workspace/venv/bin/python server.py > /workspace/parrot-api.log 2>&1 &
```

> **注意：** 使用 `/workspace/venv/bin/python`（venv 有 transformers 4.52.0 + patches），不要用系統 `python3`（系統版本缺少 gfpgan 等依賴）。

### `parrot-service/start_worker.sh`（在 piigun repo）

```bash
#!/bin/bash
set -a
source /workspace/parrot-service/.env
set +a
NVCU12=/usr/local/lib/python3.11/dist-packages/nvidia
for d in nvjitlink cusparse cusparselt cublas cudnn cufft curand cusolver cuda_runtime; do
    [ -d "${NVCU12}/${d}/lib" ] && export LD_LIBRARY_PATH="${NVCU12}/${d}/lib:${LD_LIBRARY_PATH:-}"
done
cd /workspace/parrot-service
exec python3 -m workers.gpu_worker --gpu_id 0
```

> **注意：** 必須 source .env 才能讀到 REDIS_URL 等 credentials，否則 worker 會嘗試連 localhost:6379 失敗。

**必須透過包裝器啟動**，原因：需要設定 `LD_LIBRARY_PATH` 指向 `nvidia/cu13/lib`，否則 Ray actor 初始化時 torch 動態庫載入會 segfault。（目前 Gemma 跑 bf16，不經過 int8 path，但 torch 本身仍需要此設定。）

---

## 三、節點上的目錄結構

```
/workspace/
├── parrot-api/                    ← 程式碼
│   ├── server.py                  ← 主入口
│   ├── config.py                  ← 路徑 + 預設值
│   ├── actors/
│   │   ├── ltx_actor.py
│   │   └── gfpgan_actor.py
│   ├── persistent_pipeline.py
│   ├── persistent_prompt_encoder.py
│   └── scripts/
│
├── parrot-service/                ← git clone WellyXY/piigun
│   ├── workers/gpu_worker.py
│   └── start_worker.sh            ← 啟動包裝器
│
├── gfpgan_enhance_stable_v5.py    ← GFPGAN V5 主要邏輯
├── logs/
│   ├── server.log                 ← Server 日誌
│   └── worker.log                 ← GPU Worker 日誌
│
├── models/
│   ├── ltx23/
│   │   ├── ltx-2.3-22b-distilled.safetensors          ← 43GB
│   │   └── ltx-2.3-spatial-upscaler-x2-1.0.safetensors
│   └── loras/
│       ├── LTX2_3_NSFW_furry_concat_v2.safetensors    ← 常駐 w=1.0
│       ├── LTX23_NSFW_Motion.safetensors               ← 常駐 w=0.7
│       ├── blow_job_v2.safetensors                     ← 熱切換 w=0.8（自訓練 v2；⚠️ 待 retrain）
│       ├── boobs_play.safetensors                      ← 熱切換 w=0.8（自訓練 v3 step1000）
│       ├── cowgirl.safetensors                         ← 熱切換 w=0.8  CivitAI
│       ├── dildo.safetensors                           ← 熱切換 w=0.8（自訓練 v3 step1250）
│       ├── doggy.safetensors                           ← 熱切換 w=0.8  CivitAI
│       ├── handjob.safetensors                         ← 熱切換 w=0.8（自訓練；stack blow_job w=0.6）
│       ├── lift_clothes_v2.safetensors                 ← 熱切換 w=0.6（自訓練 production v2；來源 models_backup/ltx23_production/positions/）
│       ├── masturbation.safetensors                    ← 熱切換 w=0.8  CivitAI
│       ├── missionary.safetensors                      ← 熱切換 w=0.8  CivitAI
│       └── reverse_cowgirl.safetensors                 ← 熱切換 w=0.8  CivitAI
│
├── gemma_configs/                 ← Gemma-3 12B 模型檔案
├── GFPGANv1.4.pth
└── outputs/                       ← 生成的影片（Nginx 靜態服務）
```

---

## 四、配置檔詳解

### `parrot-api/config.py`

```python
# 模型路徑
DISTILLED_CHECKPOINT = "/workspace/models/ltx23/ltx-2.3-22b-distilled.safetensors"
SPATIAL_UPSAMPLER    = "/workspace/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT           = "/workspace/gemma_configs"
GFPGAN_MODEL         = "/workspace/GFPGANv1.4.pth"
LORA_DIR             = "/workspace/models/loras"

# 推理預設值
DEFAULT_HEIGHT     = 960    # 2026-04-22 raised from 768 after A/B test
DEFAULT_WIDTH      = 640    # 2026-04-22 raised from 512
DEFAULT_NUM_FRAMES = 249    # ~10s @ 25fps
DEFAULT_FRAME_RATE = 25
DEFAULT_SEED       = 42

# 常駐 LoRA 預設權重
DEFAULT_LORA_WEIGHTS = {
    "nsfw":     1.0,
    "motion":   0.7,
    "position": 1.2,
}

# Position LoRA 個別權重覆蓋
POSITION_LORA_WEIGHTS = {
    "blow_job":        0.8,   # ⚠️ 待 retrain（training params mismatch）
    "cowgirl":         0.8,
    "doggy":           0.8,
    "handjob":         0.8,
    "lift_clothes":    0.6,
    "masturbation":    0.8,
    "missionary":      0.8,
    "reverse_cowgirl": 0.8,
    "dildo":           0.8,   # 自訓練 v3 step1250
    "boobs_play":      0.8,   # 自訓練 v3 step1000
}

# 疊加 Position LoRA（handjob 同時 stack blow_job）
POSITION_SECONDARY = {
    "handjob": ("blow_job", 0.6),
}

# GFPGAN V5 增強參數
ENHANCE_BATCH_SIZE      = 8     # 降自 32，避 bf16 Gemma 環境下 forward pass activation OOM
ENHANCE_DETECT_EVERY    = 2     # 快速動作時 DETECT_EVERY≥3 會因 bbox 過時導致人臉變形，保留 2
ENHANCE_TEMPORAL_BLEND  = 0.85
ENHANCE_DEFLICKER       = 25    # 升自 15，更強時域亮度平滑（deflicker 不動人臉幾何，只穩亮度）
ENHANCE_UPSCALE         = 2     # FFmpeg 2x Lanczos 放大

# Server
OUTPUT_DIR = Path("/workspace/outputs")
HOST       = "0.0.0.0"
PORT       = 8000
```

---

## 四-B、Inference Engine — Prompt 與 Dirty Talk 策略

### `parrot-service/workers/inference_engine.py`

#### Prompt 策略
- **用戶有傳 prompt** → 直接使用用戶 prompt
- **用戶沒傳 / 空字串** → 自動 fallback 到 `DEFAULT_PROMPTS[position]`

```python
effective_prompt = prompt.strip() if prompt and prompt.strip() \
    else DEFAULT_PROMPTS.get(position, position.replace("_", " "))
```

> ⚠️ 舊版本（2026-04-10 之前）是 **永遠忽略用戶 prompt**，強制使用 DEFAULT_PROMPTS。已修正。

#### Dirty Talk（Audio）策略
- 前端開關 `include_audio=True` 時啟用
- 用戶有填自訂文字 → 包在標準場景格式裡送出
- 用戶沒填 → fallback 到 `DEFAULT_AUDIO[position]`

```python
if include_audio:
    if audio_description.strip():
        effective_audio = f'rhythmic bouncing sounds, ... saying : "{audio_description}" ...'
    else:
        effective_audio = DEFAULT_AUDIO.get(position, "")
```

**DEFAULT_AUDIO 涵蓋的 positions（10 個）：**
`blow_job`, `cowgirl`, `doggy`, `handjob`, `lift_clothes`, `masturbation`, `missionary`, `reverse_cowgirl`, `boobs_play`, `dildo`

> ⚠️ 舊版本缺少 `boobs_play` 和 `dildo`，導致這兩個 position 開啟 dirty talk 卻無音效。已修正。

#### 當前 Checkpoint（2026-04-22）
- **使用**：`ltx-2.3-22b-distilled.safetensors`（43GB）
- **Text encoder**：Gemma-3 12B **bf16**（~24GB VRAM；bitsandbytes 未安裝）
  - 關鍵 patch：`Gemma3Config._attn_implementation = "eager"`（避 bf16 + SDPA 的 NaN bug）
- **Base LoRA**：NSFW (w=1.0) + Motion (w=0.7)，**distill LoRA 已移除**
- **Position LoRA**：10 個 position 熱切換（各自權重見 config.py）；音訊層（audio_attn1/2、audio_ff、audio_to_video_attn、video_to_audio_attn）在 position LoRA 和 base LoRA swap 時都會跳過，避免破壞 speech 生成

---

## 五、Ray Actor 架構

```
FastAPI (server.py)
  │
  ├── _inference_sem (asyncio.Semaphore(1))   ← 同時只跑一個推理
  │
  ├── LTXInferenceActor  [num_gpus=0.8]
  │    ├── DistilledPipeline (Transformer 常駐 ~43GB VRAM)
  │    ├── PersistentPromptEncoder (Gemma-3 12B bf16 常駐 ~24GB VRAM)
  │    └── PersistentDiffusionStage (10 個 position delta 預載於 CPU RAM)
  │
  └── GFPGANEnhanceActor [num_gpus=0.2]
       ├── GFPGANv1.4 (常駐 ~525MB VRAM)
       └── free_cache() — 推理前清除 CUDA cache（釋放 enhance 完的 cached memory 給 LTX）
```

**總 idle VRAM ≈ 67-69GB / 80GB。**

**Pipeline 平行化策略：**

```
Request A:  [Inference 25-27s (bf16, 640×960)] ──── [Enhancement 14-20s (batch=8)]
Request B:                     [Inference 25-27s] ──── [Enhancement]
                               ^
                               semaphore 在 inference 完成後即釋放
                               enhancement 與下一個 inference 同時進行
```

單一 10s 影片端到端 ~45-55s（無 queue，default 640×960），連續流量下每 job 間隔 ≈ inference 時間（~25-27s）因 enhance 被 overlap 掉。

---

## 六、環境變數

### RunPod 節點（parrot-api）
**無需額外環境變數**，所有路徑硬編碼在 `config.py`。
唯一需手動設的是 `LD_LIBRARY_PATH`，由 `start_server.sh` 自動處理。

| 變數 | 說明 |
|------|------|
| `LD_LIBRARY_PATH` | 由 `start_server.sh` 自動設定（bitsandbytes cu13 依賴） |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True`（在 ltx_actor.py 內設定） |
| `CUDA_VISIBLE_DEVICES` | 單 GPU 節點不需設定 |

### Railway 閘道（parrot-service）
完整 `.env`（參考 `parrot-service/.env.example`）：

```bash
# Redis
REDIS_URL=redis://localhost:6379/0

# PostgreSQL（Railway plugin 自動注入）
DATABASE_URL=postgresql://user:pass@host:5432/railway

# Cloudflare R2
R2_ACCOUNT_ID=<cloudflare_account_id>
R2_ACCESS_KEY_ID=<access_key>
R2_SECRET_ACCESS_KEY=<secret_key>
R2_BUCKET_NAME=parrot-videos
R2_PUBLIC_URL=https://pub-xxxx.r2.dev

# API
VIDEO_BASE_URL=https://your-app.railway.app/v1/jobs
WEBHOOK_SECRET=<random_string>
NUM_GPUS=1

# RunPod 模型路徑（Worker 用）
BASE_MODEL_PATH=/raid/.../Wan2.2-I2V-A14B-Diffusers
LORA_DIR=/raid/.../output_10s_conv
FAST_LORA_DIR=/raid/.../fast_loras
STORAGE_DIR=/tmp/videos
UPLOAD_DIR=/tmp/uploads

# 後處理
POSTPROCESS_ENABLED=true
RIFE_DIR=/raid/.../Practical-RIFE

# 佇列
JOB_TTL_HOURS=168          # Redis 保留 7 天
QUEUE_EXPIRE_SECONDS=300   # 排隊超過 5 分鐘自動過期
```

---

## 七、裸機部署腳本（Multi-GPU 伺服器）

### 全自動部署

```bash
bash parrot-api/scripts/deploy.sh
```

依序執行：[1] 安裝 Redis → [2] 安裝 Nginx → [3] 安裝 Python 依賴 → [4] 建立儲存目錄 → [5] 設定 `.env` → [6] 安裝 systemd services

---

### 手動啟動 API

```bash
bash parrot-api/scripts/start_api.sh
```

對應 systemd service：`parrot-api.service`

```bash
sudo systemctl enable parrot-api
sudo systemctl restart parrot-api
sudo journalctl -u parrot-api -f
```

---

### 手動啟動 GPU Workers（8 GPU）

```bash
NUM_GPUS=8 bash parrot-api/scripts/start_workers.sh
```

對應 systemd template：`parrot-worker@.service`

```bash
# 啟動全部 8 個 worker
for i in {0..7}; do
    sudo systemctl enable parrot-worker@$i
    sudo systemctl restart parrot-worker@$i
done

# 查看 worker 日誌
sudo journalctl -u parrot-worker@0 -f

# 確認所有 worker 狀態
for i in {0..7}; do
    echo "=== Worker $i ===" && sudo systemctl status parrot-worker@$i --no-pager -n 3
done
```

Worker 啟動方式：
```bash
# 每個 worker 自動對應 GPU（由 systemd template 設定）
CUDA_VISIBLE_DEVICES=N python -m workers.gpu_worker --gpu_id N
```

---

## 八、Nginx 配置

**`parrot-api/scripts/nginx.conf`** — 裸機部署用（RunPod 不需要）

重點：
- 封鎖非 Cloudflare IP 的直連（`return 444`）
- 僅允許 Cloudflare edge IP 訪問 443
- 代理 `/v1/` → `127.0.0.1:8000`
- `/metrics`、`/admin` 僅 localhost 可訪問
- 靜態影片 `/static/videos/` 直接由 Nginx 服務（不過 API）

```bash
# 安裝
sudo cp parrot-api/scripts/nginx.conf /etc/nginx/sites-available/parrot-api
sudo ln -sf /etc/nginx/sites-available/parrot-api /etc/nginx/sites-enabled/parrot-api
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

---

## 九、Cloudflare SSL 設定

```bash
bash parrot-api/scripts/setup_cloudflare.sh api.racoonn.me
```

**依序設定：**
1. Cloudflare Origin Certificate → `/etc/nginx/ssl/`
2. 更新 nginx.conf domain
3. UFW 防火牆：僅允許 Cloudflare CIDR + SSH port 44222
4. 更新 `.env` 的 `VIDEO_BASE_URL`
5. 印出 Cloudflare Dashboard 操作提示（DNS A record + SSL Full Strict + WAF）

---

## 十、API Key 管理（裸機 Multi-Client 用）

```bash
cd /raid/parrot-api

# 建立 key
python scripts/manage_keys.py create --name "client1" --rate_limit 30
# 輸出: pk_xxxxxxxx...（只顯示一次，請立即保存）

# 列出所有 key
python scripts/manage_keys.py list

# 停用 key
python scripts/manage_keys.py disable --key_hash <hash>
```

Redis 資料結構：
- `apikeys` — Set，存所有 key hash
- `apikey:{sha256}` — Hash，存 name / rate_limit / total_jobs / enabled

---

## 十一、Redis 初始化

```bash
bash parrot-api/scripts/setup_redis.sh
```

- 安裝 + 啟動 Redis
- 建立初始 API key

---

## 十二、影片清理（Cron Job）

```bash
# 加入 crontab（每小時清除超過 24h 的影片）
0 * * * * /raid/parrot-api/scripts/cleanup_videos.sh
```

`cleanup_videos.sh` 刪除 `$STORAGE_DIR` 下超過 24h 的 `.mp4` 檔案及空目錄。

---

## 十三、Railway 部署（parrot-service）

**`parrot-service/railway.json`：**
```json
{
  "build": {"builder": "DOCKERFILE", "dockerfilePath": "Dockerfile"},
  "deploy": {
    "startCommand": "/start.sh",
    "healthcheckPath": "/health",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

**`parrot-service/start.sh`：**
```bash
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Railway 自動注入：`PORT`、`REDIS_URL`、`DATABASE_URL`。

---

## 十四、版本鎖定（關鍵）

當前 pod 實測工作版本（2026-04-22）：

```bash
# PyTorch 2.4.1 + CUDA 12.4
torch==2.4.1+cu124
torchvision==0.19.1+cu124
torchaudio==2.4.1+cu124
--index-url https://download.pytorch.org/whl/cu124

# transformers — 不能升到 5.x，會破壞 LTX-2 + Gemma3 管道
transformers==4.52.0
# 需配合 ALL_PARALLEL_STYLES patch（見 Step 3）

# bitsandbytes — 不要安裝！
# 裝了之後 transformers 會自動把 Gemma 跑 int8 path，
# 破壞 speech 精度（dirty talk 變成模糊音效或消失）
# Gemma 必須跑 bf16 才能產生清晰角色語音

# LD_LIBRARY_PATH — 仍需設定（torch 動態庫載入）
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

**關鍵 patch 清單：**
1. `/workspace/ltx2_repo/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py` — `Gemma3Config._attn_implementation = "eager"`（修 bf16 + SDPA 的 NaN bug，見 Step 3）
2. `persistent_pipeline.py` — position 和 base LoRA swap 都跳過 audio 層（`audio_attn1/2`、`audio_ff`、`audio_to_video_attn`、`video_to_audio_attn`）

---

## 十五、故障排查

| 症狀 | 原因 | 解法 |
|------|------|------|
| `ModuleNotFoundError: ltx_pipelines` | LTX-2 repo 未安裝 | 重跑 fresh_setup.sh 步驟 [1/6] |
| `CUDA out of memory` 在 GFPGAN `enhance` | bf16 Gemma 24GB + LTX 43GB = 67GB 佔用；batch=32 時 GFPGAN forward activation ~19GB 峰值會 OOM | 已修：`ENHANCE_BATCH_SIZE=8`（activation 峰值 ~5GB，有 ~13GB 餘裕） |
| `FileNotFoundError: 'ffmpeg'` 在 enhance | 新 pod 沒有裝 ffmpeg（apt 不繼承 network volume） | `apt-get update -qq && apt-get install -y ffmpeg` |
| Worker 不拉 job（job `started_at=null` 永遠 queue） | `start_worker.sh` 用 system python3，缺 redis/pydantic 等套件 | `pip install 'redis[asyncio]>=5.0' pydantic asyncpg boto3 httpx pillow tenacity`，然後重啟 worker |
| 影片全黑（inference 正常但畫面全黑） | Gemma bf16 + SDPA attention 產生 NaN hidden states | 已修：`Gemma3Config._attn_implementation = "eager"`（encoder_configurator.py） |
| Dirty talk 變模糊音效或消失 | bitsandbytes 被安裝 → Gemma 自動走 int8 path → speech 精度破壞 | `pip uninstall -y bitsandbytes` → 重啟 server |
| `ray::OutOfMemoryError` 在 LTX actor init（144GB+ RAM） | 舊版預計算 nsfw+motion B@A delta，展開後 ~71GB CPU RAM | 已修復：base LoRA delta 改為 on-demand，不在 warmup 持久存儲 |
| Position LoRA 完全沒效果（所有 position 輸出一樣） | `_compute_lora_deltas` key matching bug：param 端 key 與 LoRA 端 key prefix 不符 → 空 dict | 已修復（suffix index 方式匹配）；確認 log 有 `matched N/M LoRA layers` |
| Base LoRA（nsfw/motion）swap 後視覺無變化 | `swap_base_loras` 對合成 `.weight` key 呼叫 `op.apply_to_key`，rename map 不認識此格式，suffix_index 永遠 miss → delta=0（matched=1344 但實際 magnitude=0） | 已修復（2026-04-07）：先 rename 實際 `lora_A.weight` key，strip suffix 得 renamed_prefix，再查 suffix_index；magnitude 現為 ~11M |
| `got an unexpected keyword argument 'position_weight'` | `server.py` 傳 `position_weight`，但 `ltx_actor.generate()` 參數名是 `position_w` | 已修復：actor 同時接受兩種名字 |
| `FileNotFoundError: tokenizer.model` | Gemma 下載失敗或 HF token 無效 | 以有效 token 重跑 fresh_setup.sh |
| `cannot import name 'rgb_to_grayscale'` | basicsr / torchvision 不相容 | fresh_setup.sh 已自動修補 |
| Port 8000 被佔用 | 舊 server 未停 | `lsof -i :8000` → `kill -9 <PID>` |
| `libnvJitLink.so.13 not found` | LD_LIBRARY_PATH 未設 | 使用 `/workspace/start_server.sh` 啟動，不要直接 `python3 parrot-api/server.py` |
| Worker 不拉 job | REDIS_URL 錯誤或 Redis 掛掉 | `redis-cli -h <host> ping` 確認回 PONG |
| `pkill -f server.py` 後 VRAM 不降 | Ray worker zombie 進程仍佔 VRAM（`nvidia-smi` 無進程但顯示使用中）| `ps aux \| grep -E 'python\|ray'` → `kill -9 <所有 PID>` → 等 VRAM 降至 &lt;100MB 再重啟 |
| Server 停在 VRAM=~57GB 不繼續載入 | 上次推理 OOM 的殘留分配，Ray actor 卡住 | 同上：完整 kill 所有 Python/Ray 進程，VRAM 歸零後重啟 |
| LoRA 視覺無效果（blow_job/dildo/boobs_play） | 訓練時 `first_frame_conditioning_p=0.35`，但推理永遠用 I2V（image_strength=0.9），參數不匹配 | 以正確參數重新訓練：`first_frame_conditioning_p=0.5`、`adamw8bit`、`1500 steps` |
| `OSError: [Errno 122] Disk quota exceeded` 寫入 /workspace | MooseFS inode/chunk quota（與磁碟空間無關，即使有 87TB 可用）| 所有寫入改到 `/root/`；`rm` 可用；`cp`/`dd`/`python open('w')` 均失敗 |
| `FileNotFoundError: /workspace/models/loras/dildo.safetensors` | 換新 pod 後，symlink 指向舊 pod 的 `/root/dildo_v2_output/` 路徑，目標消失 | 從 `/root/config.py` 移除 dildo/boobs_play entry；等重訓後補回 |
| `ModuleNotFoundError: No module named 'config'` | `PYTHONPATH` 未設定 + `/workspace/parrot-api/config.py` 被刪除 | 確認 `/root/config.py` 存在且 `PYTHONPATH=/root` 有 export，再重啟 |
| `torchaudio` ABI 不相容（`undefined symbol: torch_library_impl`） | 換 pod/環境後 torchaudio 版本與 torch 版本不符（如 torchaudio 2.11 + torch 2.7）| `pip install torchaudio==X.Y.Z+cuXXX --index-url ...`（版本必須與 torch 完全一致） |
| fresh_setup.sh 中途失敗（`E: Unable to locate package ffmpeg`） | 部分 pod 的 apt 不含 ffmpeg，但 apt-get update 後通常能裝 | `apt-get update -qq && apt-get install -y ffmpeg` |
