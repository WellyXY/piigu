# LoRA Retrain — lift_clothes / blow_job / handjob Design Spec

## Goal

重訓三個 position LoRA，應用 community best practices（正確 first_frame_conditioning_p、足夠影片數量、混合 resolution buckets），解決之前 lift_clothes artifact 和 action generalization 問題。

## Context

- 單一 H100 node，依序訓練：lift_clothes → blow_job → handjob
- 影片混合長度：5s（~125 frames）+ 10s（~249 frames）
- 之前 lift_clothes LoRA 問題根源：`first_frame_conditioning_p: 0.9`（應為 0.5）、只有 20 部影片、AI 生成數據 artifact

## Training Data

| Position | 影片數 | 路徑 |
|---|---|---|
| lift_clothes | 52 | `/Users/welly/Parrot API/training data/lift_clothes/` |
| blow_job | 58 | `/Users/welly/Parrot API/training data/blow_job/` |
| handjob | 49 | `/Users/welly/Parrot API/training data/handjob/` |

## Captions（固定，每部影片相同）

**lift_clothes:**
```
A female lifts her shirt to reveal her breasts. She cups and jiggles them with both hands. Her facial expression is neutral, and her lips are slightly parted. The pose is front view, and the motion level is moderate. The camera is static with a medium shot. The performance is suggestive and moderately paced. --lift_clothes
```

**blow_job:**
```
A long-haired woman, facing the camera, holding a man's penis, performing a blow job. She slowly takes the entire penis completely into her mouth, fully submerging it until her lips press against the base of the penis and lightly touch the testicles, with the penis fully accommodated in her throat, and repeatedly moves it in and out with a steady, fluid rhythm multiple times. Please ensure the stability of the face and the object, and present more refined details.
```

**handjob:**
```
A female is seen in a close-up shot, sitting and stroking a man's penis with her hands. The woman is tilting her head while looking at the camera. She uses her hand to grip and move up and down the shaft. The motion is slow and rhythmic. --handjob
```

## Hyperparameters（三個 position 統一）

| 參數 | 值 | 原因 |
|---|---|---|
| `first_frame_conditioning_p` | 0.5 | Community 推薦 action LoRA 最佳值（之前用 0.9 是最大問題） |
| `steps` | 1500 | 避免小數據集 overfit（之前 2000 steps 太多） |
| `checkpoint interval` | 250 | 保留全部 6 個 checkpoint，事後選最好的 |
| `keep_last_n` | 6 | 保留全部 6 個 checkpoint |
| `resolution-buckets` | `512x768x249;512x768x121` | 覆蓋 10s（249 frames）和 5s（121 frames）影片 |
| `rank` | 32 | 不變 |
| `alpha` | 32 | 不變 |
| `learning_rate` | 1e-4 | 不變 |
| `optimizer` | adamw8bit | 不變 |
| `quantization` | int8-quanto | 不變 |

## 執行架構（Option B：Master Sequencer）

### 本機檔案結構

```
lora_training/
├── run_all_training.sh          # 一鍵執行：SCP + 依序訓練 + 下載
├── prepare_dataset_all.py       # 通用 dataset prep（接受 position 參數）
├── lift_clothes_lora_v2.yaml    # 更新版 yaml
├── blow_job_lora_v2.yaml        # 更新版 yaml
└── handjob_lora.yaml            # 新建
```

### run_all_training.sh 流程

```
[Step 1] SCP 上傳三個 position 影片到 pod
[Step 2] SCP 上傳腳本 + yaml 到 pod
[Step 3] Pod 上依序執行：
         - prepare_dataset lift_clothes → train lift_clothes
         - prepare_dataset blow_job     → train blow_job
         - prepare_dataset handjob      → train handjob
[Step 4] 下載三個 position 所有 checkpoints 到本機
         lora_training/<position>_checkpoints_v2/
[Step 5] log 完成通知
```

失敗處理：任一 step 失敗，腳本自動診斷常見錯誤並嘗試修復，修復後重試，修復失敗才 exit。

**自動診斷 + 修復邏輯：**

| 錯誤特徵 | 自動修復 |
|---|---|
| `num_samples=0` | conditions path bug → 自動 cp `.pt` 到正確目錄，重跑 preprocess |
| `No space left on device` | 清理舊 preprocessed 目錄（上一個 position 的），重試 |
| `CUDA out of memory` | 清 VRAM cache（`torch.cuda.empty_cache`），降低 `num_dataloader_workers: 1`，重試 |
| `initial_step >= target_steps` | yaml 的 steps 設定錯誤，自動修正後重試 |
| `No delta found for position key` | position weight key 不符，自動修正 config.py，重啟 server |
| server crash / process not found | 重啟 inference server，等 ready 後繼續 |

重試最多 2 次，第 2 次還失敗才停止並輸出診斷報告到 log。

### Pod 上目錄結構

```
/workspace/lora_training/
├── lift_clothes_videos/
├── blow_job_videos/
├── handjob_videos/
├── lift_clothes_preprocessed_v2/
├── blow_job_preprocessed_v2/
├── handjob_preprocessed/
├── lift_clothes_output_v2/       # checkpoints: step 250~1500
├── blow_job_output_v2/
└── handjob_output/
```

## 部署策略（訓完後手動）

訓完後不自動部署——先下載 checkpoints 到本機，用 I2V 比較各 checkpoint（Phase 2 測試），選最好的再手動部署到 `/workspace/models/loras/<position>.safetensors`。

## 預估時間（H100 80GB）

每個 position：249 frames ~3-5s/step × 1500 steps ≈ **75-125 分鐘**
三個合計：**4-6 小時**
