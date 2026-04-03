# Piigu — 推理 Pipeline、LoRA 策略與 Prompt 設計

> 最後更新：2026-04-03

---

## 一、LoRA 三層結構

```
Transformer（43GB，常駐 VRAM）
    │
    ├── 第一層：Base LoRA（啟動時永久 fuse 進 transformer）
    │     ├── NSFW LoRA：LTX2_3_NSFW_furry_concat_v2.safetensors  weight=1.0
    │     └── Motion LoRA：LTX23_NSFW_Motion.safetensors           weight=0.7
    │
    └── 第二層：Position LoRA（熱切換，in-place add/subtract）
          ├── blow_job.safetensors         weight=1.2（較強）
          ├── cowgirl.safetensors          weight=0.8
          ├── doggy.safetensors            weight=0.8
          ├── lift_clothes.safetensors     weight=1.5（自訓練，step 1500）
          ├── masturbation.safetensors     weight=0.8
          ├── missionary.safetensors       weight=0.8
          └── reverse_cowgirl.safetensors  weight=0.8
```

**各 LoRA 作用：**
- **NSFW LoRA（1.0x）** — 增強明確的解剖學細節與顯式內容
- **Motion LoRA（0.7x）** — 增加動態流暢度、推力強度、動作模糊
- **Position LoRA（0.8-1.2x）** — 特定體位的姿態、視角、解剖定位

---

## 二、Position LoRA 熱切換機制

**關鍵檔案：** `parrot-api/persistent_pipeline.py`

**啟動時預計算（warmup，只做一次）：**
```python
# 對所有 position LoRA，預先計算 delta tensor 並存在 CPU（不佔 VRAM）
# Transformer 已佔用 ~56GB VRAM，delta 必須存 CPU 避免 OOM
for pos_key, (lora_path, strength) in self._position_loras.items():
    self._position_deltas[pos_key] = _compute_lora_deltas(
        lora_path, strength, param_names, device=cpu, dtype=bfloat16
    )
    # delta = (B @ A) * strength，每個 LoRA 約 300-500MB
```

**切換時（~2秒）：**
```python
# 減去舊的 delta（CPU tensor 移到 GPU 做 sub_）
for param_name, delta in old_deltas.items():
    self._param_dict[param_name].data.sub_(delta.to(device))

# 加上新的 delta（CPU tensor 移到 GPU 做 add_）
for param_name, delta in new_deltas.items():
    self._param_dict[param_name].data.add_(delta.to(device))
```

**為什麼這樣快：** 不從磁碟讀 LoRA，不重建 transformer，只做 CPU→GPU transfer + in-place 加減法（約 100-400ms overhead）。

**Key matching 機制（2026-04-03 修復）：**
- LoRA 檔案的 key 格式（After LTXV_LORA_COMFY_RENAMING_MAP）：`transformer_blocks.0.attn1.to_k.lora_A.weight`
- `_compute_lora_deltas` 使用 suffix index 匹配，不受 X0Model `named_parameters()` 的 wrapper prefix 影響
- 從 LoRA 端出發查詢 param，而非從 param 端查詢 LoRA

---

## 三、Prompt 策略

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

## 四、推理流程

```
Client Request（image + position）
      │
      ▼
parrot-service（Railway）
  image 下載 / base64 解碼 → 存為 .webp
  prompt = DEFAULT_PROMPTS[position]
  audio = custom wrapped 或 DEFAULT_AUDIO[position]
      │
      ▼ Redis queue
pod（RunPod H100）
  Gemma-3 12B int8（常駐 VRAM）→ text_embeddings
  LTX-2.3 22B Distilled
  + NSFW LoRA（fused, 1.0）
  + Motion LoRA（fused, 0.7）
  + Position LoRA（hot-swap, 0.8-1.2）
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

## 五、各 Position LoRA 權重

| Position | LoRA 檔案 | 權重 | 備注 |
|----------|----------|------|------|
| blow_job | blow_job.safetensors | **1.2** | 強調喉嚨深度與嘴部細節 |
| cowgirl | cowgirl.safetensors | 0.8 | 女性主導視角 |
| doggy | doggy.safetensors | 0.8 | 後入視角 |
| lift_clothes | lift_clothes.safetensors | **1.5** | 自訓練 LoRA，掀衣露胸動作；step 1500 checkpoint |
| masturbation | masturbation.safetensors | 0.8 | 自慰動作 |
| missionary | missionary.safetensors | 0.8 | 正面傳教士視角 |
| reverse_cowgirl | reverse_cowgirl.safetensors | 0.8 | 反騎視角 |

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
