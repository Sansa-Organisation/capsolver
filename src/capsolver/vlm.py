"""vlm.py - Optional VLM fallback for 95%+ accuracy (behind CAPSOLVER_VLM env flag).

Not required for 90%+ first-try OpenCV path. Only loaded if CAPSOLVER_VLM=1.

This is a placeholder for future VLM integration (e.g., Qwen2-VL 2B fine-tuned on gap detection).
The idea: if OpenCV confidence <0.5 and fails, try VLM.

VLM prompt: "In this 300x300 image, there is a gap filled with AI inpaint (smooth blurry region) of width ~{W}px and height ~{H}px. Find its left X coordinate."

We keep this file lightweight, no heavy deps unless enabled.
"""

from __future__ import annotations
import os

ENABLED = os.getenv("CAPSOLVER_VLM", "0") == "1"

def is_enabled() -> bool:
    return ENABLED

def solve_with_vlm(main_rgb, puzzle_rgba):
    """Placeholder - implement with transformers + Qwen2-VL if needed."""
    if not ENABLED:
        raise RuntimeError("VLM not enabled, set CAPSOLVER_VLM=1")
    # TODO: Load model and inference
    # from transformers import AutoModelForVision2Seq, AutoProcessor
    # ...
    raise NotImplementedError("VLM fallback not yet implemented - OpenCV achieves 90%+ first-try")
