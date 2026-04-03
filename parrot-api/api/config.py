import os
from pathlib import Path


class Settings:
    PROJECT_NAME: str = "Parrot Video API"
    API_VERSION: str = "v1"

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    BASE_MODEL_PATH: str = os.getenv("BASE_MODEL_PATH", "/raid/training/ai-toolkit/models/Wan2.2-I2V-A14B-Diffusers")
    LORA_DIR: str = os.getenv("LORA_DIR", "/raid/training/ai-toolkit/output_10s_conv")
    FAST_LORA_DIR: str = os.getenv("FAST_LORA_DIR", "/raid/training/ai-toolkit/models/fast_loras")

    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "/raid/storage/videos")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/raid/storage/uploads")

    VIDEO_BASE_URL: str = os.getenv("VIDEO_BASE_URL", "http://localhost:8000/v1/jobs")

    NUM_GPUS: int = int(os.getenv("NUM_GPUS", "8"))

    DEFAULT_NUM_FRAMES: int = 161  # 10s at 16fps
    DEFAULT_HEIGHT: int = 768
    DEFAULT_WIDTH: int = 512
    DEFAULT_FPS: int = 16
    OUTPUT_FPS: int = 30  # after interpolation
    OUTPUT_RESOLUTION: str = "720p"

    JOB_TTL_HOURS: int = int(os.getenv("JOB_TTL_HOURS", "24"))
    QUEUE_EXPIRE_SECONDS: int = int(os.getenv("QUEUE_EXPIRE_SECONDS", "300"))

    POSTPROCESS_ENABLED: bool = os.getenv("POSTPROCESS_ENABLED", "true").lower() == "true"
    RIFE_DIR: str = os.getenv("RIFE_DIR", "/raid/training/ai-toolkit/Practical-RIFE")
    UPSCALE_FACTOR: int = 2
    TARGET_FPS: int = 30


settings = Settings()

Path(settings.STORAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
