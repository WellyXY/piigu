# Parrot API — 部署、配置與運行策略

> 最後更新：2026-04-07

---

## 快速查閱表

| 操作 | 指令 |
|------|------|
| **啟動 Server** | `nohup /workspace/parrot-api/start_server.sh > /workspace/logs/server.log 2>&1 &` |
| **啟動 Worker** | `setsid bash /workspace/parrot-service/start_worker.sh > /workspace/worker.log 2>&1 &` |
| **確認就緒** | `curl http://localhost:8000/status` |
| **監看日誌** | `tail -f /workspace/logs/server.log` |
| **停止 Server** | `pkill -f server.py` |
| **完整重啟** | `pkill -f server.py; sleep 2; nohup /workspace/parrot-api/start_server.sh > /workspace/logs/server.log 2>&1 &` |

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

### Step 3 — 安裝 Python 依賴（CUDA 12.x 節點）

```bash
# 一鍵安裝（模型已在 Network Volume，跳過下載）
bash /workspace/fresh_setup.sh

# 若 fresh_setup.sh 不在 workspace，先從本機 scp 過來：
# scp -P <PORT> -i ~/.ssh/id_ed25519 "scripts/fresh_setup.sh" root@<HOST>:/workspace/fresh_setup.sh
```

**注意（CUDA 12 特有）：**
1. fresh_setup.sh 安裝完後需手動升級 torch：
   ```bash
   pip install torch==2.7.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
2. patch ltx2_repo __init__.py（刪掉 a2vid import）：
   ```bash
   sed -i 's/^from ltx_pipelines.a2vid_two_stage/# &/' \
     /workspace/ltx2_repo/packages/ltx-pipelines/src/ltx_pipelines/__init__.py
   find /workspace/ltx2_repo -name '*.pyc' -path '*/ltx_pipelines/*' -delete
   ```

---

### Step 4 — 建立目錄並驗證模型

```bash
mkdir -p /workspace/logs /workspace/outputs

# 確認 network volume 上的模型都在
ls /workspace/models/ltx23/        # 應有 2 個 .safetensors
ls /workspace/models/loras/        # 應有 10 個 .safetensors（含 lift_clothes、handjob）
ls /workspace/gemma_configs/       # 應有 Gemma 分片
ls /workspace/GFPGANv1.4.pth      # 應存在
```

---

### Step 4.5 — 部署 Worker .env

```bash
# 從本機 scp credentials 過去（不需重新查 Railway）
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  "/Users/welly/Parrot API/parrot-service.env" \
  root@<HOST>:/workspace/parrot-service/.env

# 安裝 worker Python 依賴
pip install -q -r /workspace/parrot-service/requirements.txt
```

---

### Step 5 — 啟動

```bash
nohup bash /workspace/start_server.sh > /workspace/logs/server.log 2>&1 &
nohup bash /workspace/parrot-service/start_worker.sh > /workspace/logs/worker.log 2>&1 &
```

---

### Step 6 — 驗證就緒

```bash
# 監看啟動日誌（約 60-90 秒載入模型）
tail -f /workspace/logs/server.log

# 確認狀態
curl http://localhost:8000/status
```

**預期回應：**
```json
{
  "status": "ready",
  "ltx": {"model": "LTX-2.3", "ready": true, "vram_mb": 43000},
  "gfpgan": {"model": "GFPGANv1.4", "ready": true, "vram_mb": 3200},
  "queue_depth": 0
}
```

**啟動時間分解：**

| 階段 | 時間 |
|------|------|
| Python + Ray 初始化 | 5-8s |
| GFPGAN actor 載入 | 5-8s |
| LTX transformer 載入（43GB from disk） | 4-5min |
| Position LoRA delta 預計算（8 個，CPU） | 1-2min |
| **合計** | **~6-7min** |

> **注意：** Base LoRA（nsfw/motion）的 delta **不在** warmup 預計算（B@A 展開後約 71GB CPU RAM，會 OOM）。改為 on-demand 載入，首次 weight 變更時才觸發（~3-5s）。大部分請求使用預設值→直接跳過，無額外耗時。

---

## 二、啟動腳本說明

### `parrot-api/start_server.sh`（在 piigu repo）

```bash
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
cd /workspace
exec python3 parrot-api/server.py
```

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

**必須透過包裝器啟動**，原因：bitsandbytes 的 CUDA 13 動態庫（`libnvJitLink.so.13`）需要設定 `LD_LIBRARY_PATH`，否則 Gemma int8 量化會報錯。

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
│       ├── blow_job.safetensors                        ← 熱切換 w=1.2
│       ├── cowgirl.safetensors                         ← 熱切換 w=0.8
│       ├── doggy.safetensors                           ← 熱切換 w=0.8
│       ├── handjob.safetensors                         ← 熱切換 w=0.8（自訓練）
│       ├── lift_clothes.safetensors                    ← 熱切換 w=0.6（自訓練）
│       ├── masturbation.safetensors                    ← 熱切換 w=0.8
│       ├── missionary.safetensors                      ← 熱切換 w=0.8
│       └── reverse_cowgirl.safetensors                 ← 熱切換 w=0.8
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
DEFAULT_HEIGHT     = 768
DEFAULT_WIDTH      = 512
DEFAULT_NUM_FRAMES = 249    # ~10s @ 25fps
DEFAULT_FRAME_RATE = 25
DEFAULT_SEED       = 42

# 常駐 LoRA 權重
NSFW_LORA_WEIGHT   = 1.0
MOTION_LORA_WEIGHT = 0.7

# Position LoRA 個別權重覆蓋
POSITION_LORA_WEIGHTS = {
    "blow_job":        1.2,
    "cowgirl":         0.8,
    "doggy":           0.8,
    "lift_clothes":    1.5,
    "masturbation":    0.8,
    "missionary":      0.8,
    "reverse_cowgirl": 0.8,
}

# GFPGAN V5 增強參數
ENHANCE_BATCH_SIZE      = 32
ENHANCE_DETECT_EVERY    = 2
ENHANCE_TEMPORAL_BLEND  = 0.85
ENHANCE_DEFLICKER       = 15
ENHANCE_UPSCALE         = 2     # FFmpeg 2x Lanczos 放大

# Server
OUTPUT_DIR = Path("/workspace/outputs")
HOST       = "0.0.0.0"
PORT       = 8000
```

---

## 五、Ray Actor 架構

```
FastAPI (server.py)
  │
  ├── _inference_sem (asyncio.Semaphore(1))   ← 同時只跑一個推理
  │
  ├── LTXInferenceActor  [num_gpus=0.8]
  │    ├── DistilledPipeline (Transformer 常駐)
  │    ├── PersistentPromptEncoder (Gemma int8 常駐)
  │    └── PersistentDiffusionStage (8 個 position delta 預載)
  │
  └── GFPGANEnhanceActor [num_gpus=0.2]
       ├── GFPGANv1.4 (常駐)
       └── free_cache() — 推理前清除 CUDA cache（防 OOM）
```

**Pipeline 平行化策略：**

```
Request A:  [Inference 12-14s] ────────── [Enhancement 8-10s]
Request B:               [Inference 12-14s] ────────── [Enhancement]
                         ^
                         semaphore 在 inference 完成後即釋放
                         enhancement 與下一個 inference 同時進行
```

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

```bash
# PyTorch — 必須與 CUDA 版本匹配
torch==2.11.0+cu130
torchvision==0.26.0+cu130
torchaudio==2.11.0+cu130
--index-url https://download.pytorch.org/whl/cu130

# transformers — 絕對不能升到 5.x，會破壞 LTX-2 + Gemma3 管道
transformers==4.52.1

# bitsandbytes — 需要 libnvJitLink.so.13
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

---

## 十五、故障排查

| 症狀 | 原因 | 解法 |
|------|------|------|
| `ModuleNotFoundError: ltx_pipelines` | LTX-2 repo 未安裝 | 重跑 fresh_setup.sh 步驟 [1/6] |
| `CUDA out of memory` 在 GFPGAN | GFPGAN cache 未清 | server.py 已自動處理；手動：`gfpgan_actor.free_cache.remote()` |
| `ray::OutOfMemoryError` 在 LTX actor init（144GB+ RAM） | 舊版預計算 nsfw+motion B@A delta，展開後 ~71GB CPU RAM | 已修復：base LoRA delta 改為 on-demand，不在 warmup 持久存儲 |
| Position LoRA 完全沒效果（所有 position 輸出一樣） | `_compute_lora_deltas` key matching bug：param 端 key 與 LoRA 端 key prefix 不符 → 空 dict | 已修復（suffix index 方式匹配）；確認 log 有 `matched N/M LoRA layers` |
| `got an unexpected keyword argument 'position_weight'` | `server.py` 傳 `position_weight`，但 `ltx_actor.generate()` 參數名是 `position_w` | 已修復：actor 同時接受兩種名字 |
| `FileNotFoundError: tokenizer.model` | Gemma 下載失敗或 HF token 無效 | 以有效 token 重跑 fresh_setup.sh |
| `cannot import name 'rgb_to_grayscale'` | basicsr / torchvision 不相容 | fresh_setup.sh 已自動修補 |
| Port 8000 被佔用 | 舊 server 未停 | `lsof -i :8000` → `kill -9 <PID>` |
| `libnvJitLink.so.13 not found` | LD_LIBRARY_PATH 未設 | 使用 `/workspace/start_server.sh` 啟動，不要直接 `python3 parrot-api/server.py` |
| Worker 不拉 job | REDIS_URL 錯誤或 Redis 掛掉 | `redis-cli -h <host> ping` 確認回 PONG |
