# Piigu — 推理 Pipeline、LoRA 策略與 Prompt 設計

> 最後更新：2026-04-07

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
          ├── blow_job.safetensors         w=1.2（較強）
          ├── cowgirl.safetensors          w=0.8
          ├── doggy.safetensors            w=0.8
          ├── handjob.safetensors          w=0.8（自訓練；nsfw/motion 前端預設 0.6）
          ├── lift_clothes.safetensors     w=1.2（自訓練；nsfw/motion 前端預設 0.0）
          ├── masturbation.safetensors     w=0.8
          ├── missionary.safetensors       w=0.8
          └── reverse_cowgirl.safetensors  w=0.8
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
for pos_key, (lora_path, strength) in self._position_loras.items():
    self._position_deltas[pos_key] = _compute_lora_deltas(
        lora_path, strength, param_names, device=cpu, dtype=bfloat16
    )
    # delta = (B @ A) * strength，每個 LoRA 約 300-500MB on CPU
```

**切換時（~2秒）：**
```python
# 減去舊 delta → 加上新 delta（CPU→GPU transfer + in-place 加減）
for param_name, delta in old_deltas.items():
    self._param_dict[param_name].data.sub_(delta.to(device))
for param_name, delta in new_deltas.items():
    self._param_dict[param_name].data.add_(delta.to(device))
```

### 2b. Base LoRA 權重調整（GPU layer-by-layer，~2-5s/per LoRA）

Base LoRA（nsfw、motion）**不在 warmup 預計算**（B@A 展開後 ~71GB CPU RAM OOM）。改為 on-demand、**GPU layer-by-layer** 計算：

```python
def swap_base_loras(self, weights: dict[str, float]) -> None:
    for name, new_w in weights.items():
        current_w = self._current_base_weights.get(name, 0.0)
        if abs(new_w - current_w) < 1e-4:
            continue  # 相同則跳過，零額外耗時
        # 每一層：load A+B → GPU → 計算 B@A*net_strength → apply → 立刻釋放
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for orig_prefix in lora_prefixes:
                A = f.get_tensor(...).to(device=GPU, dtype=bfloat16)
                B = f.get_tensor(...).to(device=GPU, dtype=bfloat16)
                delta = (B @ A) * net_strength   # GPU 計算，幾毫秒/層
                self._param_dict[param_name].data.add_(delta)
                del A, B, delta                  # 峰值 VRAM 僅幾 MB
```

**為什麼改成 GPU layer-by-layer：**
- 舊方案（CPU B@A）：1152 層全部在 CPU 計算 → ~40-80s（實測 nsfw swap 增加 78s）
- 新方案（GPU layer-by-layer）：每層 A、B 各幾 MB，B@A 在 GPU 完成 → 預估 ~2-5s
- Peak VRAM 開銷：單層 A+B（~幾 MB），不影響 43GB transformer
- 大部分請求使用預設權重 → 直接跳過，零額外耗時

**Key matching 機制：**
- `_compute_lora_deltas` 使用 suffix index 匹配，不受 X0Model prefix 影響
- 從 LoRA 端出發查詢 param，而非從 param 端查詢 LoRA

---

## 三、Per-Request LoRA 權重控制（2026-04-07）

所有三個 LoRA 的權重都可在每次請求中個別指定：

```json
POST /v1/generate
{
  "position": "lift_clothes",
  "nsfw_weight": 0.8,       // 選填，0.0~2.0，預設 1.0
  "motion_weight": 0.5,     // 選填，0.0~2.0，預設 0.7
  "position_weight": 0.6    // 選填，0.0~2.0，預設依 position 而定
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
      → _ensure_base_loras(nsfw_w, motion_w)  ← swap if changed
      → _ensure_position(position, position_w) ← swap if changed
```

**前端 UI：** 提供三個滑桿，選中 position 時自動顯示該 position 的預設值。

---

## 四、Prompt 策略

**使用者的 prompt 欄位無效，永遠使用硬編碼的 DEFAULT_PROMPTS。**

```python
# inference_engine.py
effective_prompt = DEFAULT_PROMPTS.get(position, position.replace("_", " "))
if include_audio and effective_audio:
    payload["audio_description"] = effective_audio
```

Audio Description 邏輯：
- `include_audio=false` → 不附加任何音效描述
- `include_audio=true` + 用戶有寫 → 包裝成標準格式後附加
- `include_audio=true` + 用戶沒寫 → 使用 `DEFAULT_AUDIO[position]`

---

## 五、推理流程

```
Client Request（image + position [+ nsfw_weight, motion_weight, position_weight]）
      │
      ▼
parrot-service（Railway）
  image 下載 → 存為 .webp
  prompt = DEFAULT_PROMPTS[position]（用戶傳的 prompt 欄位被忽略）
  audio = custom wrapped 或 DEFAULT_AUDIO[position]
      │
      ▼ Redis queue（含 weight 欄位）
pod（RunPod H100）gpu_worker
      │
      ▼ POST /generate（含 nsfw_weight, motion_weight, position_weight）
pod 推理 server（parrot-api）
  swap_base_loras(nsfw_w, motion_w)   ← on-demand，相同 weight 則跳過
  swap_position_lora(pos_key)         ← 預計算 delta，~2s
  Gemma-3 12B int8（常駐）→ text_embeddings
  LTX-2.3 22B Distilled 推理
      │
      ▼
  512×768 × 249f @ 25fps（raw）
      │
      ▼ （enhance=true）
  GFPGAN V1.4 人臉增強
  + Temporal blending（blend=0.85）
  + FFmpeg deflicker（size=15）
  + FFmpeg 2x Lanczos 放大
      │
      ▼
  1024×1536 MP4 → 上傳 R2 → callback
```

---

## 六、各 Position LoRA 預設權重

| Position | LoRA 檔案 | 預設權重 | 備注 |
|----------|----------|------|------|
| blow_job | blow_job.safetensors | **1.2** | nsfw/motion 預設 0.6 |
| cowgirl | cowgirl.safetensors | 0.8 | 女性主導視角 |
| doggy | doggy.safetensors | 0.8 | 後入視角 |
| handjob | handjob.safetensors | 0.8 | 自訓練 LoRA；nsfw/motion 預設 0.6 |
| lift_clothes | lift_clothes.safetensors | **1.2** | 自訓練 LoRA；nsfw/motion 預設 0.0 |
| masturbation | masturbation.safetensors | 0.8 | 自慰動作 |
| missionary | missionary.safetensors | 0.8 | 正面傳教士視角 |
| reverse_cowgirl | reverse_cowgirl.safetensors | 0.8 | 反騎視角 |

> 所有預設值可在每次請求時透過 `position_weight` 欄位覆蓋（0.0~2.0）。

---

## 六、推理參數

```python
# ltx_actor.py：實際傳入 pipeline 的參數
pipeline(
    prompt=full_prompt,          # Gemma-3 編碼後的文字
    seed=42,                     # 可由請求覆蓋
    height=768,                  # 直式 portrait
    width=512,
    num_frames=249,              # 10s @ 25fps
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

---

## 七、GFPGAN 後處理參數

```python
# config.py
ENHANCE_BATCH_SIZE     = 32     # 每批處理 32 幀
ENHANCE_DETECT_EVERY   = 2      # 每 2 幀做人臉偵測
ENHANCE_TEMPORAL_BLEND = 0.85   # 85% 時間平滑（防閃爍）
ENHANCE_DEFLICKER      = 15     # FFmpeg deflicker 視窗大小
ENHANCE_UPSCALE        = 2      # 2x Lanczos 放大（512×768 → 1024×1536）
```
