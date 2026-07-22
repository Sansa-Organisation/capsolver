"""Pydantic models for capsolver API - supports all CAPTCHA types v0.3.0.

v0.3.0 adds RECAPTCHA + HCAPTCHA support:
- RECAPTCHA_V2 (checkbox + image), RECAPTCHA_V2_INVISIBLE, RECAPTCHA_V3, RECAPTCHA_ENTERPRISE
- HCAPTCHA, HCAPTCHA_ENTERPRISE
- FUNCAPTCHA, GEETEST, TURNSTILE (placeholder/bypass)
Backward compat: puzzle_x can be None for ICON/RECAPTCHA.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CandidateOut(BaseModel):
    x: int
    mvar: float
    mlap: float
    mean: float
    score: float
    boundary_ratio: Optional[float] = 0.0
    depth: Optional[float] = 0.0
    overlap: Optional[float] = 0.0


class SolveRequest(BaseModel):
    main_b64: Optional[str] = Field(None, description="Base64 PNG/JPEG or data URL for main image")
    puzzle_b64: Optional[str] = Field(None, description="Base64 PNG/JPEG or data URL for puzzle image")
    main_url: Optional[str] = Field(None, description="URL for main image (or site URL for recaptcha checkbox)")
    puzzle_url: Optional[str] = None
    site_url: Optional[str] = Field(None, description="Site URL for RECAPTCHA/HCAPTCHA (optional, for checkbox bypass)")
    site_key: Optional[str] = Field(None, description="Site key for RECAPTCHA/HCAPTCHA (optional)")
    captcha_type: Optional[str] = Field(
        None,
        description=(
            "Captcha type: INPAINTING, SLIDER, ICON, NOCAPTCHA, SMART, DEFAULT, "
            "RECAPTCHA_V2, RECAPTCHA_V2_INVISIBLE, RECAPTCHA_V3, RECAPTCHA_ENTERPRISE, "
            "HCAPTCHA, HCAPTCHA_ENTERPRISE, FUNCAPTCHA, GEETEST, TURNSTILE. "
            "Auto-detect if not provided"
        ),
    )
    challenge_text: Optional[str] = Field(
        None,
        description="For ICON/RECAPTCHA/HCAPTCHA: challenge text describing which icons/images to click",
    )


class SolveResponse(BaseModel):
    captcha_type: str = Field(description="Detected or requested captcha type (supports all types)")
    puzzle_x: Optional[int] = Field(None, description="Gap position X for INPAINTING/SLIDER, None for ICON/RECAPTCHA")
    slider_x: Optional[float] = Field(None, description="Slider button left position")
    confidence: float
    method: str
    candidates: List[CandidateOut] = []
    # For ICON type
    icons: Optional[List[dict]] = None
    click_positions: Optional[List[List[int]]] = Field(
        None, description="For ICON/RECAPTCHA/HCAPTCHA: list of [x,y] click positions in order"
    )
    debug: dict = {}
    # RECAPTCHA / HCAPTCHA fields (v0.3.0)
    token: Optional[str] = Field(None, description="g-recaptcha-response / h-captcha-response token (RECAPTCHA/HCAPTCHA)")
    score: Optional[float] = Field(None, description="For RECAPTCHA_V3: score 0.0-1.0")
    challenge_type: Optional[str] = Field(None, description="Detailed challenge type: checkbox, image, invisible, v3, enterprise, etc")


class ChallengeFetchResponse(BaseModel):
    session_id: str
    captcha_type: str
    main_url: str
    puzzle_url: str
    puzzle_x: Optional[int] = None
    slider_x: Optional[float] = None
    confidence: float
    candidates: List[CandidateOut] = []
    click_positions: Optional[List[List[int]]] = None


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = "Test User"
    captcha_type: Optional[str] = None


class SignupResponse(BaseModel):
    success: bool
    captcha_type: Optional[str] = None
    captcha_param: Optional[str] = None
    session_id: Optional[str] = None
    info: dict = {}
    error: Optional[str] = None


class TypesResponse(BaseModel):
    supported_types: List[str]
    aliases: dict
    description: dict
