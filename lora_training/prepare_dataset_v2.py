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
