"""Shim for v7 - imports from solver.py"""

try:
    from .solver import (
        detect_gap,
        GapDetection,
        Candidate,
        SceneInfo,
        solve_gap_from_files,
        solve_inpainting,
        classify_inpainting_scene,
        InpaintingResult,
    )
except ImportError:
    # fallback when run as src.capsolver
    from src.capsolver.solver import (
        detect_gap,
        GapDetection,
        Candidate,
        SceneInfo,
        solve_gap_from_files,
        solve_inpainting,
        classify_inpainting_scene,
        InpaintingResult,
    )

__all__ = [
    "detect_gap",
    "GapDetection",
    "Candidate",
    "SceneInfo",
    "solve_gap_from_files",
    "solve_inpainting",
    "classify_inpainting_scene",
    "InpaintingResult",
]
