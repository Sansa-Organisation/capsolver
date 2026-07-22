"""capsolver package - supports all Aliyun CAPTCHA V3 types."""

try:
    from .solver_inpainting import detect_gap, GapDetection
    from .drag import puzzle_to_slider, slider_to_puzzle
    from .registry import get_solver, SUPPORTED_TYPES, detect_captcha_type
except ImportError:
    from capsolver.solver_inpainting import detect_gap, GapDetection
    from capsolver.drag import puzzle_to_slider, slider_to_puzzle
    from capsolver.registry import get_solver, SUPPORTED_TYPES, detect_captcha_type

__all__ = [
    "detect_gap",
    "GapDetection",
    "puzzle_to_slider",
    "slider_to_puzzle",
    "get_solver",
    "SUPPORTED_TYPES",
    "detect_captcha_type",
]
