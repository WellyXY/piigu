#!/usr/bin/env python3
"""Test hot-swap timing: consecutive inferences with different positions."""

import json
import time
import requests

API = "http://localhost:8000"
IMG = "/workspace/test_imgs/ref_0.webp"

PROMPT = "A beautiful woman in the scene, her body moving with natural rhythm."
AUDIO = "soft moans and breathing"

SEQUENCE = ["cowgirl", "doggy", "cowgirl", "missionary"]

def generate(position, seed=42):
    t0 = time.time()
    resp = requests.post(f"{API}/generate", json={
        "prompt": PROMPT,
        "position": position,
        "image_path": IMG,
        "seed": seed,
        "enhance": False,
        "audio_description": AUDIO,
    }, timeout=300)
    elapsed = time.time() - t0
    data = resp.json()
    return elapsed, data

def main():
    # Check status first
    status = requests.get(f"{API}/status").json()
    print(f"Current position: {status['ltx']['current_position']}")
    print(f"Cached LoRAs: {len(status['ltx']['position_loras_cached'])} positions\n")

    print(f"{'#':<4} {'Position':<20} {'Wall(s)':<10} {'Infer(s)':<10} {'Note'}")
    print("-" * 60)

    prev_pos = status['ltx']['current_position'][0]

    for i, pos in enumerate(SEQUENCE):
        swap_needed = pos != prev_pos
        note = "SWAP" if swap_needed else "same"
        elapsed, data = generate(pos, seed=42 + i)
        infer_s = data.get("inference_s", "?")
        print(f"{i:<4} {pos:<20} {elapsed:<10.1f} {str(infer_s):<10} {note}")
        prev_pos = pos

    print("\nDone.")

if __name__ == "__main__":
    main()
