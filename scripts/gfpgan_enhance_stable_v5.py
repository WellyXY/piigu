#!/usr/bin/env python3
"""
Face enhancement v5 — GFPGAN + bidirectional Gaussian temporal blend.

Same quality as v2 stable, clean codebase (no dead CodeFormer/BiSeNet/xseg).

Pipeline:
  1. Detect faces (keyframe every 2) + bidirectional EMA smooth affines
  2. Batch GFPGAN (batch=32, weight=0.3)
  3. Bidirectional Gaussian temporal blend
  4. CPU paste-back (numpy warpAffine + alpha blend)
  5. FFmpeg deflicker
"""

import os
import time
import glob
import subprocess
import argparse
import numpy as np
import cv2
import torch
from gfpgan import GFPGANer


FFHQ_512 = np.float32([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
])


def build_enhancer(model_path, device='cuda'):
    enhancer = GFPGANer(
        model_path=model_path, upscale=1,
        arch='clean', channel_multiplier=2,
        bg_upsampler=None, device=device,
    )
    with torch.inference_mode():
        enhancer.enhance(np.zeros((512, 512, 3), dtype=np.uint8), paste_back=True)
    return enhancer


def smooth_affines_ema(affines, alpha=0.45):
    """Bidirectional EMA smoothing over affine matrices."""
    if not affines:
        return affines
    keys = sorted(affines)
    fwd, bwd = {}, {}
    prev = None
    for k in keys:
        fwd[k] = affines[k] if prev is None else alpha * affines[k] + (1 - alpha) * prev
        prev = fwd[k]
    prev = None
    for k in reversed(keys):
        bwd[k] = affines[k] if prev is None else alpha * affines[k] + (1 - alpha) * prev
        prev = bwd[k]
    return {k: (fwd[k] + bwd[k]) / 2.0 for k in keys}


def interp_affine(a1, a2, t):
    return a1 * (1 - t) + a2 * t


def box_mask(h, w, blur=0.4):
    blur_px = int(w * 0.5 * blur)
    blur_area = max(blur_px // 2, 1)
    mask = np.ones((h, w), dtype=np.float32)
    mask[:blur_area, :] = 0
    mask[-blur_area:, :] = 0
    mask[:, :blur_area] = 0
    mask[:, -blur_area:] = 0
    if blur_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_px * 0.25)
    return mask


def gaussian_temporal_blend(enhanced, win_r=3, temporal_blend=0.85):
    """
    Bidirectional Gaussian-weighted temporal blend over enhanced face crops.
    Reduces GAN frame-to-frame jitter without one-directional drift.
    """
    n = len(enhanced)
    if n < 2:
        return enhanced

    # Build Gaussian kernel weights for window
    sigma = win_r / 2.0
    weights = np.array([np.exp(-0.5 * (i / sigma) ** 2) for i in range(win_r + 1)],
                       dtype=np.float32)

    result = []
    for i in range(n):
        acc = np.zeros_like(enhanced[i], dtype=np.float32)
        w_sum = 0.0
        for di in range(-win_r, win_r + 1):
            j = i + di
            if 0 <= j < n:
                w = weights[abs(di)]
                acc += w * enhanced[j].astype(np.float32)
                w_sum += w
        smoothed = (acc / w_sum).astype(np.uint8)
        blended = (temporal_blend * enhanced[i].astype(np.float32) +
                   (1 - temporal_blend) * smoothed.astype(np.float32))
        result.append(blended.astype(np.uint8))
    return result


def process_video(enhancer, input_path, output_path,
                  blend=0.85, batch_size=32, detect_every=2,
                  win_r=3, temporal_blend=0.85, deflicker=15):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n = len(frames)
    if n == 0:
        return None

    face_helper = enhancer.face_helper
    fsize = 512

    # ------------------------------------------------------------------ #
    # Phase 1: Face detection + bidirectional EMA affine smoothing         #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()

    key_indices = list(range(0, n, detect_every))
    if key_indices[-1] != n - 1:
        key_indices.append(n - 1)

    kf_affines = {}
    for idx in key_indices:
        face_helper.clean_all()
        face_helper.read_image(frames[idx])
        face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5)
        if face_helper.all_landmarks_5:
            lm = face_helper.all_landmarks_5[0].astype(np.float32)
            aff, _ = cv2.estimateAffinePartial2D(
                lm, FFHQ_512, method=cv2.RANSAC, ransacReprojThreshold=100)
            if aff is not None:
                kf_affines[idx] = aff

    all_affines = {}
    skeys = sorted(kf_affines)
    for i in range(n):
        if i in kf_affines:
            all_affines[i] = kf_affines[i]
        else:
            prev_k = max((k for k in skeys if k <= i), default=None)
            next_k = min((k for k in skeys if k >= i), default=None)
            if prev_k is not None and next_k is not None and prev_k != next_k:
                t = (i - prev_k) / (next_k - prev_k)
                all_affines[i] = interp_affine(kf_affines[prev_k], kf_affines[next_k], t)
            elif prev_k is not None:
                all_affines[i] = kf_affines[prev_k]
            elif next_k is not None:
                all_affines[i] = kf_affines[next_k]

    all_affines = smooth_affines_ema(all_affines, alpha=0.45)

    face_data = []
    for i in range(n):
        if i not in all_affines:
            continue
        crop = cv2.warpAffine(frames[i], all_affines[i], (fsize, fsize),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
        face_data.append((i, crop, all_affines[i]))

    t_detect = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Phase 2: Batch GFPGAN                                               #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    enhanced = []

    for bs in range(0, len(face_data), batch_size):
        items = face_data[bs:bs + batch_size]
        batch = np.stack([c for _, c, _ in items])
        t = torch.from_numpy(batch.copy()).to(enhancer.device).float()
        t = t[:, :, :, [2, 1, 0]].contiguous()
        t = t.permute(0, 3, 1, 2).contiguous() / 255.0
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            out = enhancer.gfpgan(t, return_rgb=False, weight=0.3)[0]
        out = ((out.clamp(-1, 1) + 1) / 2 * 255).byte()
        out = out.permute(0, 2, 3, 1)[:, :, :, [2, 1, 0]].cpu().numpy()
        enhanced.extend([out[i] for i in range(out.shape[0])])

    t_gfpgan = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Phase 3: Bidirectional Gaussian temporal blend                      #
    # ------------------------------------------------------------------ #
    enhanced = gaussian_temporal_blend(enhanced, win_r=win_r,
                                       temporal_blend=temporal_blend)

    # ------------------------------------------------------------------ #
    # Phase 4: CPU paste-back + FFmpeg                                    #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()

    # Pre-compute inverse affines and face-space masks
    fmask = box_mask(fsize, fsize, blur=0.4)
    face_map = {}
    for ei, (fidx, orig_crop, affine) in enumerate(face_data):
        inv_aff = cv2.invertAffineTransform(affine)
        inv_mask = cv2.warpAffine(fmask, inv_aff, (w, h))
        inv_mask = np.clip(inv_mask, 0, 1)
        blended_crop = cv2.addWeighted(enhanced[ei], blend, orig_crop, 1.0 - blend, 0)
        face_map[fidx] = (blended_crop, inv_aff, inv_mask)

    vf = [f'deflicker=size={deflicker}:mode=am'] if deflicker > 0 else []
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0',
        '-i', input_path, '-map', '0:v', '-map', '1:a?',
    ]
    if vf:
        cmd += ['-vf', ','.join(vf)]
    cmd += ['-c:v', 'libx264', '-preset', 'ultrafast',
            '-crf', '18', '-pix_fmt', 'yuv420p',
            '-c:a', 'copy', '-shortest', output_path]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i, frame in enumerate(frames):
        if i in face_map:
            blended_crop, inv_aff, inv_mask = face_map[i]
            inv_restored = cv2.warpAffine(blended_crop, inv_aff, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)
            m = inv_mask[:, :, np.newaxis]
            result = (inv_restored.astype(np.float32) * m +
                      frame.astype(np.float32) * (1 - m)).astype(np.uint8)
        else:
            result = frame
        proc.stdin.write(result.tobytes())

    proc.stdin.close()
    proc.wait()
    t_paste = time.perf_counter() - t0

    return {
        'frames': n, 'faces': len(face_data),
        't_detect': t_detect, 't_gfpgan': t_gfpgan, 't_paste': t_paste,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--blend', type=float, default=0.85)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--detect-every', type=int, default=2)
    parser.add_argument('--win-r', type=int, default=3)
    parser.add_argument('--temporal-blend', type=float, default=0.85)
    parser.add_argument('--deflicker', type=int, default=15)
    parser.add_argument('--pattern', default='*.mp4')
    parser.add_argument('--single', default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading GFPGAN...')
    t0 = time.perf_counter()
    enhancer = build_enhancer(args.model_path)
    print(f'Ready in {time.perf_counter()-t0:.1f}s')

    files = ([args.single] if args.single else
             sorted(glob.glob(os.path.join(args.input_dir, args.pattern))))
    files = [f for f in files
             if not any(s in os.path.basename(f)
                        for s in ('_enhanced', '_swapped', '_swap_enhance', '_compare'))]

    print(f'\nProcessing {len(files)} videos...')
    total_start = time.perf_counter()

    for i, fpath in enumerate(files):
        name = os.path.basename(fpath)
        base, ext = os.path.splitext(name)
        out = os.path.join(args.output_dir, f'{base}_enhanced{ext}')
        t = time.perf_counter()
        stats = process_video(
            enhancer, fpath, out,
            blend=args.blend, batch_size=args.batch_size,
            detect_every=args.detect_every,
            win_r=args.win_r,
            temporal_blend=args.temporal_blend,
            deflicker=args.deflicker)
        elapsed = time.perf_counter() - t
        if stats:
            print(f'  [{i+1}/{len(files)}] {name}: {elapsed:.1f}s '
                  f'(detect {stats["t_detect"]:.1f}s  '
                  f'gfpgan {stats["t_gfpgan"]:.1f}s  '
                  f'paste {stats["t_paste"]:.1f}s)')
        else:
            print(f'  [{i+1}/{len(files)}] {name}: FAILED')

    total = time.perf_counter() - total_start
    print(f'\nDone! {len(files)} videos in {total:.1f}s ({total/len(files):.1f}s avg)')


if __name__ == '__main__':
    main()
