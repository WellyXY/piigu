# Piigu — 推理 Pipeline、LoRA 策略與 Prompt 設計

> 最後更新：2026-04-23（**新增 cumshot position LoRA（11 個）**；**真正修好 base LoRA (nsfw/motion) 熱切換** — 三層 bug 連環：server Pydantic 未宣告 top-level weight 欄位 + actor 從未呼叫 `_ensure_base_loras` + `PersistentDiffusionStage` 沒收到 `scalable_loras` kwarg 導致 swap_base_loras 永遠 no-op；default 解析度 640×960；Gemma bf16 + eager attention；audio 層在 position/base LoRA swap 時都跳過；ENHANCE_BATCH_SIZE 32→8；ENHANCE_DEFLICKER 15→25）

---

## 一、LoRA 三層結構

```
Transformer（43GB，常駐 VRAM）
    │
    ├── 第一層：Base LoRA（啟動時 fuse 進 transformer，可 per-request 調權重）
    │     ├── NSFW LoRA：LTX2_3_NSFW_furry_concat_v2.safetensors  預設 w=1.0
    │     └── Motion LoRA：LTX23_NSFW_Motion.safetensors           預設 w=0.7
    │
    └── 第二層：Position LoRA（熱切換，in-place add/subtract）
          ├── blow_job_v2.safetensors      w=0.8  自訓練 v2（⚠️ 視覺無效，待 retrain）
          ├── boobs_play.safetensors       w=0.8  自訓練 v3 step1000
          ├── cowgirl.safetensors          w=0.8  CivitAI（有效）
          ├── dildo.safetensors            w=0.8  自訓練 v3 step1250（有效；本地來源見下方）
          ├── doggy.safetensors            w=0.8  CivitAI（有效）
          ├── handjob.safetensors          w=0.8  自訓練 v2（有效；+ stacked blow_job w=0.6）
          ├── lift_clothes_v2.safetensors  w=0.6  自訓練 production v2（有效；nsfw/motion 前端預設 0.0/0.0）
          ├── masturbation.safetensors     w=0.8  CivitAI（有效）
          ├── missionary.safetensors       w=0.8  CivitAI（有效）
          ├── reverse_cowgirl.safetensors  w=0.8  CivitAI（有效）
          └── cumshot.safetensors          w=1.0  自訓練 v2 step2000（DEV→distilled，trigger: ltxmove_cumshot）

    第三層：audio 層 skip（在 position 和 base LoRA swap 時都不套用）
      - audio_attn1 / audio_attn2 / audio_ff / audio_to_video_attn / video_to_audio_attn
      - 原因：position LoRA 未訓練 audio 分支，若套用會破壞 speech 生成
```

**各 LoRA 作用：**
- **NSFW LoRA（預設 1.0x）** — 增強明確的解剖學細節與顯式內容
- **Motion LoRA（預設 0.7x）** — 增加動態流暢度、推力強度、動作模糊
- **Position LoRA（0.6-1.2x）** — 特定體位的姿態、視角、解剖定位

---

## 二、LoRA 熱切換機制

**關鍵檔案：** `parrot-api/persistent_pipeline.py`

### 2a. Position LoRA 切換（預計算，~2s）

**啟動時預計算（warmup，只做一次）：**
```python
# 所有 position LoRA 的 delta tensor 預先計算並存在 CPU（不佔 VRAM）
# strength=1.0 儲存，apply 時再 scale（支援任意 weight）
for pos_key, (lora_path, strength) in self._position_loras.items():
    self._position_deltas[pos_name] = _compute_lora_deltas(
        lora_path, 1.0, param_names, device=cpu, dtype=bfloat16
    )
    # 每個 LoRA 約 300-500MB on CPU，magnitude ~1M+
```

**切換時（~2秒）：**
```python
# 減去舊 delta → 加上新 delta（CPU→GPU transfer + in-place 加減）
for param_name, delta in old_deltas.items():
    self._param_dict[param_name].data.sub_(delta.to(device) * old_strength)
for param_name, delta in new_deltas.items():
    self._param_dict[param_name].data.add_(delta.to(device) * new_strength)
```

### 2b. Base LoRA 權重調整（GPU layer-by-layer，~1s/per swap）

Base LoRA（nsfw、motion）**不在 warmup 預計算**（B@A 展開後 ~71GB CPU RAM OOM）。改為 on-demand、**GPU layer-by-layer** 計算：

```python
def swap_base_loras(self, weights: dict[str, float]) -> None:
    # GPU layer-by-layer: ~1s 以內（H100 實測）
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        # 先建 renamed_prefix → orig_prefix map（用實際 lora_A key 重命名）
        prefix_map = {}
        for k in all_keys:
            if k.endswith(".lora_A.weight"):
                orig_pfx = k[:-len(".lora_A.weight")]
                renamed_key = op.apply_to_key(k)
                renamed_pfx = renamed_key[:-len(".lora_A.weight")]
                prefix_map[renamed_pfx] = orig_pfx
        # 每一層：load A+B → GPU → 計算 B@A*net_strength → apply → 立刻釋放
        for renamed_prefix, orig_prefix in prefix_map.items():
            lora_weight_key = f"{renamed_prefix}.weight"
            param_name = suffix_index.get(lora_weight_key)
            if param_name is None:
                continue
            A = f.get_tensor(f"{orig_prefix}.lora_A.weight").to(device=GPU, dtype=bfloat16)
            B = f.get_tensor(f"{orig_prefix}.lora_B.weight").to(device=GPU, dtype=bfloat16)
            delta = (B @ A) * net_strength   # GPU 計算，幾毫秒/層
            self._param_dict[param_name].data.add_(delta)
            del A, B, delta                  # 峰值 VRAM 僅幾 MB
```

**關鍵：Key matching 必須從 renamed lora_A key 推導 prefix**
- ❌ 舊方法：`op.apply_to_key(f"{orig_prefix}.weight")` — rename map 只認含 `.lora_A.weight` 的完整 key，對合成 `.weight` key 無效，導致 delta=0（silent bug）
- ✅ 正確方法：先 rename 實際 `lora_A.weight` key，strip suffix 得 renamed_prefix，再查 suffix_index

**H100 實測：**
- nsfw swap（1344 layers）：magnitude=11,159,614，~1s
- motion swap（1248 layers）：magnitude=2,844,674，~1s
- 兩個同時 swap：total overhead ~1s 以內
- 大部分請求使用預設權重 → 直接跳過，**零額外耗時**

---

## 三、Stacked Position LoRA（handjob 特殊）

handjob 同時疊加兩個 position LoRA：

```python
# config.py
POSITION_SECONDARY: dict[str, tuple[str, float]] = {
    "handjob": ("blow_job", 0.6),
}
```

```python
# ltx_actor.py
def _ensure_position(self, position: str, position_w: float) -> None:
    key = _pos_key(position, position_w)          # ("handjob", 0.6)
    secondary = cfg.POSITION_SECONDARY.get(position)
    if secondary:
        sec_name, sec_w = secondary
        sec_key = _pos_key(sec_name, sec_w)        # ("blow_job", 0.6)
        self._persistent_stage.ensure_loras(key, sec_key)
    else:
        self._persistent_stage.ensure_loras(key, None)
```

`ensure_loras` 管理 primary + secondary 的正確 add/subtract 順序，避免重複疊加。

---

## 四、Per-Request LoRA 權重控制

所有三個 LoRA 的權重都可在每次請求中個別指定：

```json
POST /v1/generate
{
  "position": "lift_clothes",
  "nsfw_weight": 0.8,       // 選填，0.0~2.0，預設依 position
  "motion_weight": 0.5,     // 選填，0.0~2.0，預設依 position
  "position_weight": 0.6    // 選填，0.0~2.0，預設依 position
}
```

**資料流：**
```
API Request (nsfw_weight, motion_weight, position_weight)
  → GenerateRequest model 驗證（ge=0.0, le=2.0）
  → create_job() 存入 Redis job hash
  → gpu_worker._parse_weight() 解析（None/"" → None）
  → engine.generate() → POST /generate payload
  → ltx_actor.generate()
      → _ensure_base_loras(nsfw_w, motion_w)  ← swap if changed（~1s overhead）
      → _ensure_position(position, position_w) ← swap if changed（~2s overhead）
```

**前端 UI：** 提供三個滑桿，選中 position 時自動顯示該 position 的預設值。

---

## 五、Prompt 策略

**預設行為：** 使用 `DEFAULT_PROMPTS[position]` 硬編碼 prompt。

**自訂 Prompt（2026-04-07 新增）：**
```json
POST /v1/generate
{
  "prompt": "your custom scene description",
  ...
}
```
- `prompt` 非空 → 使用自訂 prompt
- `prompt` 為空字串或未傳 → fallback 到 `DEFAULT_PROMPTS[position]`

```python
# inference_engine.py
effective_prompt = prompt.strip() if prompt and prompt.strip() \
    else DEFAULT_PROMPTS.get(position, position.replace("_", " "))
```

前端有「Custom Prompt」toggle（預設關閉），開啟後出現 textarea。

Audio Description 邏輯（不受 prompt 影響）：
- `include_audio=false` → 不附加任何音效描述
- `include_audio=true` + 用戶有寫 → 包裝成標準格式後附加
- `include_audio=true` + 用戶沒寫 → 使用 `DEFAULT_AUDIO[position]`

---

## 六、推理流程

```
Client Request（image + position [+ nsfw_weight, motion_weight, position_weight, prompt]）
      │
      ▼
parrot-service（Railway）
  image 下載 → 存為 .webp
  prompt = custom 或 DEFAULT_PROMPTS[position]
  audio = custom wrapped 或 DEFAULT_AUDIO[position]
      │
      ▼ Redis queue（含 weight 欄位）
pod（RunPod H100）gpu_worker
      │
      ▼ POST /generate（含 nsfw_weight, motion_weight, position_weight）
pod 推理 server（parrot-api）
  swap_base_loras(nsfw_w, motion_w)   ← on-demand，相同 weight 則跳過（~1s）；跳過 audio 層
  ensure_loras(pos_key, sec_key)      ← 預計算 delta，~2s；跳過 audio 層
  Gemma-3 12B bf16（常駐，eager attention）→ text_embeddings
  LTX-2.3 22B Distilled 推理（25-27s @ 640×960）
      │
      ▼
  640×960 × 249f @ 25fps（raw，default；可透過 API 的 height/width override）
      │
      ▼ （enhance=true，~14-20s）
  GFPGAN V1.4 人臉增強（batch=8，避 activation OOM）
  + Temporal blending（blend=0.85）
  + FFmpeg deflicker（size=25）
  + FFmpeg 2x Lanczos 放大
      │
      ▼
  1280×1920 MP4 → 上傳 R2 → callback
```

**單 job 時間：** 10s 影片 ~45-55s（inference 25-27s + enhance 14-20s）。15s 影片 ~70s（inference ~34s + enhance ~33s）。持續流量下每 job 間隔 ≈ inference 時間，因 enhance 被 semaphore pipeline 平行化掉。

---

## 七、各 Position 預設值

| Position | LoRA 檔案 | 預設 pos_w | nsfw 預設 | motion 預設 | 備注 |
|----------|----------|----------|---------|-----------|------|
| blow_job | blow_job_v2.safetensors | 0.8 | 1.0 | 0.7 | 自訓練 v2；⚠️ 視覺無效（training params mismatch，待 retrain） |
| boobs_play | boobs_play.safetensors | 0.8 | 1.0 | 0.7 | 自訓練 v3 step1000 |
| cowgirl | cowgirl.safetensors | 0.8 | 1.0 | 0.7 | CivitAI；有效 |
| dildo | dildo.safetensors | 0.8 | 1.0 | 0.7 | 自訓練 v3 step1250；有效 |
| doggy | doggy.safetensors | 0.8 | 1.0 | 0.7 | CivitAI；有效 |
| handjob | handjob.safetensors | 0.8 | 1.0 | 0.7 | 自訓練 v2；有效；同時 stack blow_job w=0.6 |
| lift_clothes | lift_clothes_v2.safetensors | **0.6** | **0.0** | **0.0** | 自訓練 production v2；有效 |
| masturbation | masturbation.safetensors | 0.8 | 1.0 | 0.7 | CivitAI；有效 |
| missionary | missionary.safetensors | 0.8 | 1.0 | 0.7 | CivitAI；有效 |
| reverse_cowgirl | reverse_cowgirl.safetensors | 0.8 | 1.0 | 0.7 | CivitAI；有效 |
| cumshot | cumshot.safetensors | **1.0** | **0.0** | **0.0** | 自訓練 v2 step2000（DEV base 訓練，distilled 推理）；trigger word `ltxmove_cumshot`；預設關 nsfw/motion base 避免過度動作 |

> 所有預設值可在每次請求時透過對應 weight 欄位覆蓋（0.0~2.0）。
> 共 **11 個 position**，啟動時全部 pre-compute delta 存於 CPU RAM（每個 ~300-500MB）。

---

## 八、推理參數

```python
# ltx_actor.py：實際傳入 pipeline 的參數
pipeline(
    prompt=full_prompt,          # Gemma-3 編碼後的文字
    seed=42,                     # 可由請求覆蓋
    height=768,                  # 直式 portrait
    width=512,
    num_frames=249,              # 5s=121 / 10s=249 / 15s=377（8n+1）
    frame_rate=25,
    images=[
        ImageConditioningInput(
            path=image_path,
            frame_idx=0,
            strength=0.9,        # 90% 圖片條件化強度
        )
    ],
    tiling_config=TilingConfig.default(),
)
```

Duration 支援範圍：**5s – 15s**（前端滑桿 + API 均驗證）

### 解析度階段拆解（當前 default 640×960）

```
Stage 1 (LTX 低解析度生成，8 steps)    →  320×480   基底
Stage 2 (spatial upsampler x2, 3 steps)→  640×960   raw_video 輸出
GFPGAN 臉部強化（per-frame）            →  640×960   同解析度修臉
ffmpeg lanczos 2x upscale              →  1280×1920 最終 enhanced_video
```

- **Stage 1** = 目標解析度的 1/2（DistilledPipeline 自動）
- **Stage 2** 是**有學習過**的 spatial upscaler（加細節）
- **ffmpeg lanczos** 是**純插值**，不加細節，只放大像素

### 解析度 A/B 測試紀錄（2026-04-22）

比較 512×768 vs 640×960，每個位置跑 10+ cases（cowgirl / missionary 各 10，doggy/lift_clothes/masturbation 各 1）：

| 解析度 | Stage 1 | 最終輸出 | inference_s | enhance_s | wall |
|--------|---------|----------|-------------|-----------|------|
| 512×768（原 default） | 256×384 | 1024×1536 | 16-20s | 13-15s | ~33s |
| **640×960（新 default）** | **320×480** | **1280×1920** | **25-27s** | **14-20s** | **~50s** |

**決定：採用 640×960 為 default**（視覺對照後確認細節明顯提升，值得 +50% 推理時間）。

per-request 仍可以用 `height/width` 欄位 override（API 層接受）。

---

## 九、GFPGAN 後處理參數

```python
# config.py
ENHANCE_BATCH_SIZE     = 8      # 降自 32，避 bf16 Gemma 環境下 GFPGAN forward activation OOM
ENHANCE_DETECT_EVERY   = 2      # 每 2 幀做人臉偵測（實測 4 會讓快速動作臉框過時 → 臉變形）
ENHANCE_TEMPORAL_BLEND = 0.85   # 85% 時間平滑（防閃爍）
ENHANCE_DEFLICKER      = 25     # 升自 15，加強時域亮度平滑抑 flicker（只壓亮度不動臉部幾何）
ENHANCE_UPSCALE        = 2      # 2x Lanczos 放大（default 640×960 → 1280×1920）
```

**Flicker 來源 × 緩解方式：**
- **亮度跳動**（光影 frame 間變化）→ `ENHANCE_DEFLICKER=25`（ffmpeg deflicker 濾波）
- **臉部細節跳動**（GFPGAN per-frame 修出的細節略不同）→ `ENHANCE_TEMPORAL_BLEND=0.85`（時域平滑）
- **bbox 抖動**（detection 每次偵測結果略不同）→ `ENHANCE_DETECT_EVERY=2`（每 2 frame 重偵測）

> **為何不把 `DETECT_EVERY` 調高到 4？** 25fps 下 DETECT_EVERY=4 = 160ms 才更新一次 bbox；快速動作（cowgirl/doggy 的人臉位移 50-100px）會讓 GFPGAN 用過時 bbox 去裁錯位置，貼回後臉部變形且看起來模糊。實測保持 2 才穩。

> **為何不把 `TEMPORAL_BLEND` 調高過 0.9？** 會把自然的眨眼、嘴部動作等細節平均掉 → 臉變糊、失去表情。

---

## 十、已知問題與 LoRA Retrain 計畫

### blow_job_v2 視覺無效根因（dildo / boobs_play 已修復為 v3）

`blow_job_v2` LoRA **數學上有被 apply**（delta magnitude ~680k–890k，matched=1152/1152），但生成影片無對應動作。`dildo` 和 `boobs_play` 在 2026-04-08 至 2026-04-10 已用正確參數重訓為 **v3**（目前 pod 上是 v3 step1250 / step1000），有效。

**blow_job 殘留問題的根因：**

| 參數 | blow_job_v2（無效） | v3 配方（cowgirl/handjob/lift_clothes_v2/dildo_v3/boobs_play_v3） |
|------|----------|--------------------------------------|
| `first_frame_conditioning_p` | **0.35**（65% T2V, 35% I2V）| **0.5**（50/50 混合）|
| `optimizer_type` | `adamw` | `adamw8bit` |
| `steps` | `1000` | `1500` |
| 推理模式 | I2V @ image_strength=0.9 | I2V @ image_strength=0.9 |

訓練時以 T2V 為主（conditioning_p=0.35），但推理永遠以 I2V 條件化輸入圖片；LoRA 學到的 motion 特徵在 I2V 模式下不激活 → 無明顯動作。

**注意：** blow_job 還有 trigger word 問題（已修復）：
- 訓練 caption 結尾都是 `-- blow_job`
- 舊 DEFAULT_PROMPTS 完全缺少此 trigger word（已於 2026-04-08 修復，commit 23143b4）

### Base LoRA 熱切換之前完全失效（2026-04-23 才真正修好）

**症狀：** 前端設 `nsfw=0, motion=0` 完全沒效果——畫面依舊有明確 NSFW/Motion enhancement。

**根因是三層連環 bug（都得修才生效）：**

1. **Server Pydantic：** `GenerateRequest` 只宣告 `lora_weights: dict`，worker 送的 top-level `nsfw_weight=0.0, motion_weight=0.0` 被 Pydantic **靜默 drop** → server 用 `DEFAULT_LORA_WEIGHTS = {nsfw: 1.0, motion: 0.7}`。
   修：加 `nsfw_weight / motion_weight / position_weight: float | None = None` + merge 進 `lw` (`commit 47c7c77`)

2. **Actor 不呼叫 base swap：** `LTXInferenceActor.generate()` 接到 `nsfw_w / motion_w` 參數後，**只呼叫 `_ensure_position`，從未呼叫 `_ensure_base_loras`**。`_current_nsfw_w / _current_motion_w` 只在 `__init__` 設過，之後永遠是 startup 預設。
   修：新增 `_ensure_base_loras(nsfw_w, motion_w)` method + `generate()` 呼叫 (`commit 3e68e1a`)

3. **`scalable_loras` kwarg 遺漏：** Actor 建 `PersistentDiffusionStage` 時**沒傳 `scalable_loras`** → `self._scalable_loras_config = {}`。`swap_base_loras` 的 loop 每次都 `continue`（`"nsfw" not in {}`）→ 靜默 no-op 無 error 無 log。
   修：`PersistentDiffusionStage(..., scalable_loras={"nsfw": (cfg.NSFW_LORA, default_nsfw_w), "motion": (cfg.MOTION_LORA, default_motion_w)})` (`commit c644b6a`)

**驗證（curl 實測）：** 前端 nsfw=0/motion=0 → log 出現 `[PIPELINE] swap_base_loras: rescaling 'nsfw' 1.000 -> 0.000 (net=-1.000, ... magnitude=6.9M)`；反向切回 1.0/0.7 也 OK；連續同樣 weight 是 no-op。

### Retrain 計畫

需以正確參數重新訓練：

```yaml
first_frame_conditioning_p: 0.5   # 原來 0.35 → 改 0.5
optimizer_type: adamw8bit          # 原來 adamw → 改 adamw8bit
max_train_steps: 1500              # 原來 1000 → 改 1500
```

訓練規則：
- 必須用 **DEV** 模型訓練（`ltx-2.3-22b-dev.safetensors`）
- 訓練資料結構：`{name}/latents/*.pt` + `{name}/conditions/*.pt`
- 產出 `.safetensors` 後部署到 `/workspace/models/loras/` 並重啟 server

### 自訓練 LoRA 部署來源（新 pod 必讀）

**目前狀態（2026-04-22）：**
- `dildo.safetensors` → v3 step1250（有效）
- `boobs_play.safetensors` → v3 step1000（有效）
- `lift_clothes_v2.safetensors` → production v2（有效）
- `handjob.safetensors` → v2（有效；stack blow_job w=0.6）
- `blow_job_v2.safetensors` → v2（⚠️ 視覺無效，待 retrain）

**Network volume 沒有這兩個檔案時，從本地上傳：**
```bash
# dildo v3 step1250
scp -P <port> -i ~/.ssh/id_ed25519 \
  '/Users/welly/Parrot API/models_backup/dildo_v3/lora_weights_step_01250.safetensors' \
  root@<ip>:/workspace/models/loras/dildo.safetensors

# lift_clothes production v2
scp -P <port> -i ~/.ssh/id_ed25519 \
  '/Users/welly/Parrot API/models_backup/ltx23_production/positions/lift_clothes_v2.safetensors' \
  root@<ip>:/workspace/models/loras/lift_clothes_v2.safetensors
```

**舊 symlink 問題（已解決）：**
舊版 dildo/boobs_play 存在 pod `/root/`，換 pod 後 symlink 斷掉。現在改用 v3 實際檔案直接上傳，不用 symlink。
