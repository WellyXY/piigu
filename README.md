# Piigu — Parrot API

LTX-2.3 video generation API with persistent LoRA hot-swap.

## Structure
```
parrot-api/          FastAPI server + Ray actors
scripts/             Setup, enhancement, test scripts
loras/position/      Position LoRAs (8 positions)
loras/community/     Motion LoRA (NSFW LoRA stored separately)
```

## Quick Start (new pod)
1. Clone repo on local machine
2. Upload LoRAs + code to new pod
3. Run `bash scripts/fresh_setup.sh <HF_TOKEN>`
