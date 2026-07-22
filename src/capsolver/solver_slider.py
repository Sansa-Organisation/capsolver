"""Slider solver for Aliyun CAPTCHA V3 SLIDER type (classic slider).

SLIDER vs INPAINTING differences:
- SLIDER: Classic jigsaw puzzle, gap is often simple rectangular or with small tabs,
  main image has a visible dark gap (not AI inpainted, just empty or dark),
  puzzle piece is cut from main and needs to be dragged to align.
- INPAINTING: Gap filled with AI inpaint (smooth blurry), irregular shape, more complex.

But solving method is similar: find gap position via low variance / edge detection.
SLIDER often easier because gap is more visible (dark vs bright).

We reuse INPAINTING solver with tuned weights for SLIDER:
- SLIDER gap is usually dark (low brightness) vs INPAINTING can be white or dark
- Edge detection more reliable for SLIDER (gap has strong edge)

This module wraps inpainting solver with SLIDER-specific tuning.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from .solver_inpainting import detect_gap as detect_inpainting, GapDetection as InpaintingDetection, Candidate as InpaintingCandidate
except ImportError:
    from capsolver.solver_inpainting import detect_gap as detect_inpainting, GapDetection as InpaintingDetection, Candidate as InpaintingCandidate
    # fallback for PYTHONPATH=src
    try:
        from capsolver.solver_inpainting import detect_gap as detect_inpainting
    except:
        from src.capsolver.solver_inpainting import detect_gap as detect_inpainting  # type: ignore
        from src.capsolver.solver_inpainting import GapDetection as InpaintingDetection, Candidate as InpaintingCandidate  # type: ignore


@dataclass
class SliderCandidate:
    x: int
    mvar: float
    mlap: float
    mean: float
    score: float
    boundary_ratio: float
    depth: float
    is_dark_gap: bool  # SLIDER gaps often dark


@dataclass
class SliderDetection:
    x: int
    x_refined: float
    method: str
    confidence: float
    scene: str
    candidates: List[SliderCandidate]
    debug: dict


def detect_slider_gap(main_rgb: np.ndarray, puzzle_rgba: np.ndarray) -> SliderDetection:
    """
    SLIDER type: classic slider puzzle.
    Reuses INPAINTING solver but with SLIDER-tuned logic.
    """
    # Use inpainting solver as base
    det = detect_inpainting(main_rgb, puzzle_rgba)

    # Convert to SliderDetection with additional dark-gap check
    main_gray = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2GRAY)
    puz_alpha = puzzle_rgba[:, :, 3] if puzzle_rgba.shape[2] == 4 else np.ones(puzzle_rgba.shape[:2], dtype=np.uint8)*255

    # Check if gap is dark (common for SLIDER)
    # Dark gap: mean brightness low (<80) at best position
    is_dark = det.debug.get("best_mvar", 0) < 100 and det.candidates[0].mean < 80 if det.candidates else False

    candidates = []
    for c in det.candidates[:10]:
        candidates.append(SliderCandidate(
            x=c.x,
            mvar=c.mvar,
            mlap=c.mlap,
            mean=c.mean,
            score=c.score,
            boundary_ratio=c.boundary_ratio,
            depth=c.depth,
            is_dark_gap=c.mean < 80,
        ))

    # For SLIDER, dark gaps are common, so boost candidates with low mean (dark)
    # If best candidate is not dark but second is dark and close score, prefer dark for SLIDER
    if not candidates[0].is_dark_gap and len(candidates) > 1:
        for alt in candidates[1:4]:
            if alt.is_dark_gap and alt.mvar < candidates[0].mvar * 1.5:
                # Prefer dark gap for SLIDER type
                best = alt
                candidates = [best] + [c for c in candidates if c.x != best.x]
                break

    best = candidates[0]
    # Confidence: SLIDER typically higher conf than INPAINTING complex because gap more visible
    confidence = det.confidence
    if best.is_dark_gap:
        confidence = max(confidence, 0.85)  # dark gaps easy

    return SliderDetection(
        x=best.x,
        x_refined=det.x_refined,
        method=f"slider_{det.method}",
        confidence=float(confidence),
        scene=det.scene.type if hasattr(det.scene, 'type') else str(det.scene),
        candidates=candidates,
        debug={**det.debug, "is_dark_gap": best.is_dark_gap, "original_method": det.method},
    )


def solve_slider_from_files(main_path: str, puzzle_path: str) -> SliderDetection:
    main = np.array(Image.open(main_path).convert("RGB"))
    puzzle = np.array(Image.open(puzzle_path))
    return detect_slider_gap(main, puzzle)


# Legacy alias for compatibility
def detect_gap(main_rgb, puzzle_rgba):
    return detect_slider_gap(main_rgb, puzzle_rgba)
