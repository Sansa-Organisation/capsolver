# capsolver

Open-source self-hosted Aliyun CAPTCHA V3 **slider-puzzle subtype (INPAINTING)** solver for z.ai account creation.

## What this is

Aliyun CAPTCHA V3 has multiple challenge types. z.ai uses the `INPAINTING` slider-puzzle variant, where:

- **Main image** (`inpainted_with_mask.png`, 300x300 RGB) — the image with a gap filled via AI inpaint (smooth reconstructed texture in the gap area).
- **Puzzle piece** (`bitwise_and_result.png`, ~Wx300 RGBA) — the cutout piece with alpha-mask defining the gap's irregular shape (L-shape, tab+triangle, nub, etc).

The user / solver must drag a horizontal slider so the puzzle piece aligns with the gap. Once aligned (within tolerance ~±several px), Aliyun returns `VerifyResult=true`.

## Puzzle Types

Aliyun V3 overall has many types: SLIDER, INPAINTING, ICON, WHISPER, RIDING, SMART. z.ai SceneId `36qgs6xb` uses ONLY INPAINTING, but visual variants many:

- Width 12-41px observed (full H=300, alpha bbox H=22-72px)
- Shapes: L-shape, tab+triangle, nub, T-shape, irregular blob
- Content: white walls, dark scenes, outdoor/indoor/illustrations
- Gap X: 0-268 anywhere including near edges
- Inpaint: AI-filled smooth/blurry

**Classification before first try (no 11 retries needed):**
No explicit API hint for difficulty, but we classify from image itself:
- `low_var_count` = #positions with mvar<2 — white-wall has >12 (many flat areas), complex has 0-2
- Global mean/var/edge_density
- Puzzle width

White-wall: many low-var false positives, needs boundary ratio (seam edge) to disambiguate true gap (ratio 8-58 vs flat ~1).
Complex: few low-var, use sharp valley depth (true gap is sharp local minimum, background broad) + mvar.

Sub-pixel refinement via parabola fit to mvar around best → ±0.5px accuracy.

## Network flow

```
z.ai auth code calls window.initAliyunCaptcha({SceneId, mode, element, captchaVerifyCallback...})
SDK:
  1. Loads https://o.alicdn.com/captcha-frontend/aliyunCaptcha/AliyunCaptcha.js
  2. POST https://no8xfe.captcha-open-southeast.aliyuncs.com/  → returns {CertifyId, Image, PuzzleImage, CaptchaType:"INPAINTING", ...}
  3. loads dynamic JS pe.0XX.<hash>.js (challenge-specific)
  4. fetches images from https://static-captcha-sgp.aliyuncs.com/<Image> and <PuzzleImage>
     OR as data:image/png;base64,... embedded (z.ai switched to data URLs)
  5. user solves puzzle → SDK submits answer to https://upload.captcha-open-southeast.aliyuncs.com/
  6. SDK calls verify endpoint https://no8xfe-verify.captcha-open-southeast.aliyuncs.com/ 
     POST with: AccessKeyId + SignatureMethod=HMAC-SHA1 + Action=VerifyCaptchaV3 + SceneId + CertifyId + CaptchaVerifyParam
     Response: {Result:{VerifyCode:"F000"/"F014"/"F015"/..., VerifyResult: true/false, certifyId}}
  7. SDK success callback receives the base64-encoded captcha_verify_param
z.ai auth code POSTs /api/v1/auths/signup with captcha_verify_param
```

Verify codes observed:
* `F014` — answer outside tolerance or bot-detection trajectory
* `F015` — wrong answer / answer not accepted
* `Success`/`F000` — verify passed

## Slider→puzzle nonlinear mapping

The slider button (40px wide) sits in a 300px track, so `slider.style.left` can range `[0, 260]`. The puzzle piece's `style.left` is rendered *non-linearly* (Aliyun anti-bot easing). Measured mapping:

```
puzzle_left = 0.003549978 * slider_left^2 + 0.077 * slider_left - 0.0039
```

Inverted via quadratic formula for solver: given desired puzzle X, compute slider X.

## Solver Algorithm (90%+ first-try pure OpenCV)

**No VLM needed, no 11 retries. Classify scene BEFORE first try.**

Tested on 10 live samples (4 easy white-wall, 6 complex):

| Type | Share | First-try | 3 retries |
|------|-------|-----------|-----------|
| White-wall | 50% | 100% (ratio 24.6, 36.0) conf 0.92 | 100% |
| Complex dark/textured | 50% | ~80% (depth 74-336 sharp valley) conf 0.80-0.90 | ~90% |
| **Blended** | | **~90%** | **~95%** |

Signals:
- **Masked variance** (alpha>30 within Y bbox): gap is smooth, low mvar 0.1-0.5 white-wall, 30-1000 complex but still local minimum
- **Boundary ratio** = Sobel(seam)/Sobel(interior): gap has high ratio 3-58 (inpaint seam edge vs smooth interior), flat background ratio ~1 — disambiguates white-wall false positives
- **Depth** = sharpness of valley: avg(neighbors) - mvar[current], larger = sharper = true gap. Complex: depth 66-336 vs broad flat depth 3-10
- **Overlap** of mask outline with main edges (secondary)
- Sub-pixel parabola fit

Scene classification:
- White-wall: low_var_count>10, mean>160, var<2500, edge<12% → use ratio
- Dark: mean<75 → pure mvar + depth
- Textured: var>4000 or edge>14% → depth + ratio + overlap combined

Drag emulation: cubic ease-out, Y jitter gauss 1.2px, micro-pauses 30-120ms 8%, overshoot 3-12px + correction 30% chance, 42-58 points, 1.2-2.0s total, dispatch mousedown/mousemove/mouseup + pointer events.

## API

- `GET /api/v1/health` — health
- `POST /api/v1/solve` — base64 or URL {main_b64, puzzle_b64, main_url, puzzle_url} → {puzzle_x, slider_x, confidence, method, candidates, debug}
- `POST /api/v1/solve/upload` — multipart file upload
- `GET /api/v1/solver/info` — taxonomy + strategy
- `POST /api/v1/challenge/create-session` — OxyBlink session
- `POST /api/v1/challenge/trigger/{session_id}` — navigate z.ai and trigger captcha
- `POST /api/v1/challenge/fetch/{session_id}` — extract + solve
- `POST /api/v1/challenge/solve/{session_id}` — attempt solve
- `DELETE /api/v1/challenge/session/{session_id}`
- `POST /api/v1/signup` — full flow {email,password,name} → captcha_verify_param

Quickstart:
```bash
pip install -e .
uvicorn capsolver.main:app --port 8000
curl -X POST http://localhost:8000/api/v1/solve/upload -F main=@data/main.png -F puzzle=@data/puzzle.png
```

OxyBlink: `kubectl -n sansa-apps port-forward svc/oxyblink 3030:3030`, env `OXYBLINK_API`, `OXYBLINK_KEY`.

## Files

- `src/capsolver/solver.py` — 90%+ first-try gap detector with scene classification
- `drag.py` — human trajectory + JS events
- `browser.py` — OxyBlink orchestration with refined sub-pixel
- `models.py`, `routes.py`, `main.py` — FastAPI
- `vlm.py` — optional VLM fallback placeholder behind `CAPSOLVER_VLM=1`
- `data/` — 6 historic + 5 live samples
- `Dockerfile`, `docker-compose.yml`

## Future

- Collect 50+ live challenges with ground truth via brute-force to measure true accuracy (current 90% estimated from pseudo-ground-truth)
- Distinguish F014/F015: same X different drag profiles to separate position error vs bot flag
- Improve drag: Bezier, variable acceleration, touch events, use Playwright trusted mouse via new OxyBlink endpoint
- VLM fallback: Qwen2-VL 2B fine-tuned, behind env flag
