"""
Post-processing wrapper: calls the existing postprocess.py via subprocess.
Runs on CPU so it doesn't block GPU inference.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def run_postprocess(
    input_path: str,
    output_path: str,
    target_fps: int = 30,
    upscale_factor: int = 2,
    no_interpolate: bool = False,
    no_upscale: bool = False,
) -> float:
    """
    Run the post-processing pipeline. Returns elapsed time in seconds.
    Uses the standalone postprocess.py script.
    """
    script = os.path.join(os.path.dirname(__file__), "..", "..", "postprocess.py")
    script = os.path.abspath(script)

    if not os.path.isfile(script):
        logger.warning(f"postprocess.py not found at {script}, skipping")
        import shutil
        shutil.copy2(input_path, output_path)
        return 0.0

    cmd = [
        sys.executable, script,
        "--input", input_path,
        "--output", output_path,
        "--target_fps", str(target_fps),
        "--upscale_factor", str(upscale_factor),
    ]
    if no_interpolate:
        cmd.append("--no_interpolate")
    if no_upscale:
        cmd.append("--no_upscale")

    t0 = time.time()
    logger.info(f"Post-processing: {input_path} → {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Post-processing failed: {result.stderr}")
        import shutil
        shutil.copy2(input_path, output_path)
    else:
        logger.info(f"Post-processing done in {elapsed:.1f}s")

    return elapsed
