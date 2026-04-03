from __future__ import annotations

import os
import shutil
from pathlib import Path

from api.config import settings


def get_job_video_dir(job_id: str) -> str:
    d = os.path.join(settings.STORAGE_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_video(job_id: str, source_path: str, filename: str = "output.mp4") -> str:
    dest_dir = get_job_video_dir(job_id)
    dest = os.path.join(dest_dir, filename)
    shutil.move(source_path, dest)
    return dest


def get_video_path(job_id: str, filename: str = "output.mp4") -> str | None:
    path = os.path.join(settings.STORAGE_DIR, job_id, filename)
    if os.path.isfile(path):
        return path
    return None


def cleanup_job_files(job_id: str):
    d = os.path.join(settings.STORAGE_DIR, job_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)

    upload_pattern = os.path.join(settings.UPLOAD_DIR)
    # Upload images are cleaned up separately via cron
