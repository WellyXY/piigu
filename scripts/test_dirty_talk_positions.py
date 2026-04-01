#!/usr/bin/env python3
"""Dirty talk test for all positions except blow_job.

7 positions x 5 images = 35 videos (with V5 enhancement).
Runs via Parrot API.
"""

import json
import os
import sys
import time

import requests

API = "http://localhost:8000"
IMG_DIR = "/workspace/test_dt_imgs"
OUT_DIR = "/workspace/test_dt_results"
SEED = 42

PROMPTS = {
    "cowgirl": {
        "base": "The woman, in a cowgirl position atop the man, facing him, drives her hips downward with fierce, rhythmic intensity, guiding his penis deep into her vagina with each deliberate thrust. The penetration is forceful, each motion probing the depths and textures within, the slight resistance at the entrance amplifying their raw connection. Her hands tightly clutch a nearby surface or her thighs, her body arching with unrestrained fervor. Her head tilts back sharply, lips parting in a ragged, primal moan that echoes her overwhelming desire. Her eyes, clenched shut in a storm of pleasure tinged with fleeting pain, flutter open momentarily, revealing a fevered, almost pleading gaze. Her brows knit tightly, a faint wince crossing her face as the intensity peaks, yet her cheeks burn with a deep, crimson flush, betraying her immersion in the consuming ecstasy. Her breath comes in sharp, uneven gasps, her lower lip quivering as she bites it fiercely, a strained yet enraptured smile breaking through with each powerful motion. Her hair whips wildly with her movements, catching the soft light that bathes her glistening skin, heightening the primal allure.",
        "audio": 'rhythmic skin-on-skin contact, soft bed springs creaking, she moans loudly between heavy breaths, whispering "you feel so good" and "don\'t stop" in a sultry breathless voice, her moans building in intensity with each thrust, punctuated by sharp gasps.',
    },
    "doggy": {
        "base": "The man stands behind the woman, extending his penis toward her vagina. As his penis presses against the entrance, he enters with a forceful yet controlled rhythm, each thrust delving deep into the texture and warmth within. The slight resistance of the vaginal opening intensifies the precision of his movements, his hands gripping her hips firmly for support. The woman, on her hands and knees, arches her back sharply, her mouth opening slightly with each powerful thrust. Her head then turns to face forward, tilted slightly back, revealing the back of her head. The room's dim light casts stark shadows on her glistening skin, amplifying the raw intensity of the moment.",
        "audio": 'loud rhythmic slapping sounds, she moans deeply with each thrust, gasping "harder" and "right there" in a breathy desperate voice, the sounds of her gripping the sheets, heavy panting mixed with sharp cries of pleasure.',
    },
    "handjob": {
        "base": "In the scene, the woman tightly grasps the man's penis with both hands, moving them slowly up and down, the motion of her fingers clearly visible as they glide deliberately to the tip and descend, delivering a perfect handjob. Her expression is vivid, cheeks flushed with a deep blush, mouth wide open as she gasps heavily, letting out bold moans, her gaze intensely fixed on him before half-closing. Her eyebrows arch with a wicked charm, a sly smirk curling her lips, her breathing rapid, tongue grazing her lips, biting them with a hungry edge.",
        "audio": 'wet stroking sounds, she whispers "you like that?" and "come for me" in a teasing seductive voice, soft moaning between strokes, her breathing getting heavier and faster, occasional giggles mixed with sultry encouragement.',
    },
    "lift_clothes": {
        "base": "A female lifts her shirt to reveal her breasts. She cups and jiggles them with both hands. Her facial expression is neutral, and her lips are slightly parted. The pose is front view, and the motion level is moderate. The camera is static with a medium shot. The performance is suggestive and moderately paced.",
        "audio": 'soft fabric rustling, she giggles playfully and whispers "want to see more?" in a teasing innocent voice, light breathing, a soft moan as she touches herself, whispering "do you like what you see?"',
    },
    "masturbation": {
        "base": "The woman, reclining or seated, explores her body with slow, deliberate touches, her fingers tracing over her skin before settling on her clitoris with focused, rhythmic strokes. Each movement is intentional, alternating between gentle circles and firmer presses, the slick warmth of her arousal heightening the tactile intensity. Her other hand roams, teasing her breasts or inner thighs, amplifying the building sensation. Her head tilts back sharply, lips parting in a soft, primal moan that betrays her deepening pleasure. Her eyes, clenched shut in a wave of ecstasy tinged with fleeting intensity, flicker open briefly, revealing a wild, introspective glint. Her brows furrow subtly, a faint wince crossing her face as the sensation peaks, yet her cheeks flush with a deep, feverish blush, surrendering to the consuming desire. Her breath comes in ragged, uneven gasps, her lower lip trembling as she bites it gently, a strained yet rapturous smile breaking through with each pulsing touch. The soft light bathes her glistening skin, casting stark shadows that heighten the raw, intimate solitude of the moment.",
        "audio": 'wet rhythmic sounds, soft building moans, she whispers "oh god" and "yes" breathlessly, her breathing becoming ragged and desperate, crescendo of pleasure sounds, gasping and whimpering with increasing urgency.',
    },
    "missionary": {
        "base": "The woman, a fair-skinned blonde with piercing blue eyes, lies on her back in a missionary pose, her legs spread wide as she drives her hips upward with fierce, rhythmic intensity. Each motion is forceful, meeting the man's penetrating thrusts, the penis delving deep into her vagina, exploring its texture and warmth with relentless precision. The slight resistance of the entrance heightens their connection, her body arching with unrestrained fervor. Her head tilts back, lips parting in a gasping, primal moan that echoes her consuming desire. Her eyes, squeezed shut in a mix of overwhelming pleasure and fleeting pain, flicker open briefly, revealing a fevered, almost pleading gaze. Her brows knit tightly, a faint wince crossing her face with each powerful thrust, yet her cheeks flush with a deep, crimson blush, betraying her immersion in ecstasy. Her breath comes in ragged bursts, her lower lip quivering as she bites it hard, a strained yet rapturous smile breaking through, her body trembling in sync with the relentless rhythm. The camera, positioned between her legs, captures the glistening sheen of her skin under soft light, amplifying the raw intensity.",
        "audio": 'bed creaking rhythmically, skin slapping sounds, she moans "deeper" and "don\'t stop" with increasing desperation, heavy breathing mixed with sharp gasps, her voice breaking with pleasure, whispering "I\'m so close".',
    },
    "reverse_cowgirl": {
        "base": "The woman, positioned above the man and facing forward, drives her hips downward with fierce, rhythmic intensity, guiding his penis deep into her vagina with each deliberate thrust. The penetration is forceful, each motion probing the depths and textures within, the slight resistance of the entrance amplifying their raw connection. Her hands clutch a nearby surface or her thighs tightly, her body arching with unrestrained fervor. Her head is tilted slightly back, eyes fixed forward, maintaining a steady gaze ahead. Her hair whips wildly with her movements, catching the soft light that bathes her glistening skin, heightening the primal allure of the scene.",
        "audio": 'rhythmic bouncing sounds, skin contact, she moans with a deep primal intensity, gasping "you feel amazing" and "so deep" in a breathy voice, heavy panting and sharp exhales punctuating each downward thrust.',
    },
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    positions = sorted(PROMPTS.keys())
    total_videos = 0
    for pos in positions:
        imgs = sorted([
            os.path.join(IMG_DIR, pos, f)
            for f in os.listdir(os.path.join(IMG_DIR, pos)) if f.endswith(".webp")
        ])[:5]
        total_videos += len(imgs)

    print(f"Dirty Talk Test: {len(positions)} positions, {total_videos} videos total")
    print(f"Positions: {positions}\n")

    results = []
    total_t0 = time.time()

    for pi, pos in enumerate(positions):
        imgs = sorted([
            os.path.join(IMG_DIR, pos, f)
            for f in os.listdir(os.path.join(IMG_DIR, pos)) if f.endswith(".webp")
        ])[:5]

        prompt_data = PROMPTS[pos]
        full_prompt = prompt_data["base"]
        audio_desc = prompt_data["audio"]

        print(f"=== {pos} ({pi+1}/{len(positions)}, {len(imgs)} images) ===")

        for ii, img in enumerate(imgs):
            label = f"{pos}_{ii}"
            print(f"  [{label}] starting...", end=" ", flush=True)

            t0 = time.time()
            resp = requests.post(f"{API}/generate", json={
                "prompt": full_prompt,
                "position": pos,
                "image_path": img,
                "seed": SEED + ii,
                "lora_weights": {"nsfw": 1.0, "motion": 0.7, "position": 0.8},
                "enhance": True,
                "audio_description": audio_desc,
                "image_strength": 0.9,
            }, timeout=300)

            elapsed = time.time() - t0
            data = resp.json()

            raw = data.get("raw_video", "")
            enh = data.get("enhanced_video", "")

            if raw and os.path.exists(raw):
                new_raw = os.path.join(OUT_DIR, f"{label}_raw.mp4")
                os.rename(raw, new_raw)
                data["raw_video"] = new_raw
            if enh and os.path.exists(enh):
                new_enh = os.path.join(OUT_DIR, f"{label}_enhanced.mp4")
                os.rename(enh, new_enh)
                data["enhanced_video"] = new_enh

            print(f"done in {elapsed:.1f}s (infer={data.get('inference_s')}s enh={data.get('enhance_s')}s)")
            results.append({"position": pos, "image": os.path.basename(img), **data})

    total = time.time() - total_t0
    print(f"\n=== ALL DONE in {total:.0f}s ({total/60:.1f} min) ===")

    summary_path = os.path.join(OUT_DIR, "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {summary_path}")

    print(f"\n{'Position':<18} {'Videos':<8} {'Avg Infer(s)':<14} {'Avg Enhance(s)':<16} {'Avg Total(s)'}")
    for pos in positions:
        pr = [r for r in results if r["position"] == pos]
        if not pr:
            continue
        avg_infer = sum(r["inference_s"] for r in pr) / len(pr)
        avg_enh = sum(r.get("enhance_s") or 0 for r in pr) / len(pr)
        avg_total = sum(r["elapsed_s"] for r in pr) / len(pr)
        print(f"{pos:<18} {len(pr):<8} {avg_infer:<14.1f} {avg_enh:<16.1f} {avg_total:.1f}")


if __name__ == "__main__":
    main()
