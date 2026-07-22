"""Human-like drag emulator for Aliyun slider.

Aliyun observed:
- Slider track 300px, button 40px => slider_left range [0,260]
- Puzzle piece left css nonlinearly maps from slider_left:
  puzzle_left = 0.003549978 * slider_left^2 + 0.077 * slider_left - 0.0039
  (measured empirically, <0.004 error)

  Inverted: given puzzle_left desired, solve quadratic for slider_left:
  0.003549978*S^2 + 0.077*S -0.0039 - P = 0
  S = (-0.077 + sqrt(0.077^2 +4*0.003549978*(P+0.0039))) / (2*0.003549978)

Bot detection Aliyun likely checks:
  - Total drag time (too fast = bot)
  - Velocity profile (linear = bot, human has ease-out)
  - Y jitter (human not perfectly horizontal)
  - Micro pauses
  - Overshoot + correction sometimes

We generate realistic trajectory.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple


def puzzle_to_slider(puzzle_x: float) -> float:
    """Convert puzzle piece left (main image coords) to slider button left."""
    # puzzle = a*slider^2 + b*slider + c
    a = 0.003549978
    b = 0.077
    c = -0.0039
    # Solve a*S^2 + b*S + (c - P) = 0
    # S = (-b + sqrt(b^2 -4*a*(c-P))) / (2a)
    P = puzzle_x
    disc = b*b - 4*a*(c - P)
    if disc < 0:
        disc = 0
    s = (-b + math.sqrt(disc)) / (2*a)
    # Clamp to valid slider range [0,260]
    return max(0.0, min(260.0, s))


def slider_to_puzzle(slider_x: float) -> float:
    a = 0.003549978
    b = 0.077
    c = -0.0039
    return a*slider_x*slider_x + b*slider_x + c


def generate_human_trajectory(target_slider_x: float, num_points: int = 35) -> List[Tuple[float, float, int]]:
    """
    Returns list of (slider_x, y_jitter, delay_ms) points.
    Starts at 0, ends at target_slider_x.
    """
    # Randomize total duration 800-1600ms
    total_duration = random.uniform(900, 1700)
    # 70% ease-out, 30% with overshoot
    overshoot = random.random() < 0.30
    overshoot_amount = random.uniform(3, 12) if overshoot else 0

    target_with_overshoot = target_slider_x + overshoot_amount if overshoot else target_slider_x

    points: List[Tuple[float, float, int]] = []
    # Use cubic ease-out: 1 - (1-t)^3
    for i in range(num_points):
        t = i / (num_points - 1)  # 0..1
        # Ease-out
        eased = 1 - pow(1 - t, 2.5)  # slightly faster than cubic
        x = eased * target_with_overshoot

        # Y jitter: small random walk around 0
        y_jitter = random.gauss(0, 1.2)
        # More jitter in middle
        jitter_scale = 1.0 + math.sin(t * math.pi) * 0.6
        y_jitter *= jitter_scale

        # Delay: total duration distributed, with micro-pauses
        # Base linear delay, but with variance
        base_delay = total_duration / num_points
        # Add randomness
        delay_var = random.uniform(-5, 15)
        delay = max(2, base_delay + delay_var)

        # Occasional micro-pause (10% chance)
        if random.random() < 0.08 and 0.2 < t < 0.8:
            delay += random.uniform(30, 120)

        points.append((x, y_jitter, int(delay)))

    # If overshoot, add correction phase
    if overshoot:
        # 3-5 points correcting back to target
        correction_points = random.randint(3, 6)
        for i in range(correction_points):
            t = (i+1) / correction_points
            # Ease-in-out correction
            x = target_with_overshoot - (overshoot_amount * (1 - pow(1 - t, 2)))
            y_jitter = random.gauss(0, 0.8)
            delay = random.randint(20, 60)
            points.append((x, y_jitter, delay))

    # Ensure last point exactly at target
    points[-1] = (target_slider_x, random.gauss(0, 0.5), points[-1][2])

    return points


def trajectory_to_js_events(trajectory: List[Tuple[float, float, int]]) -> str:
    """
    Generates JS code that will dispatch mouse/pointer events on #aliyunCaptcha-sliding-slider
    Returns JS string that when eval'd in OxyBlink performs the drag.
    The JS itself uses setTimeout chain to simulate timing.
    """
    # We need to dispatch mousedown, then mousemove series, then mouseup
    # On the slider element
    pts_json = [[round(x, 2), round(y, 2), d] for x, y, d in trajectory]

    js = f"""
(async () => {{
  const pts = {pts_json};
  const slider = document.querySelector('#aliyunCaptcha-sliding-slider') ||
                 document.querySelector('.aliyunCaptcha-sliding-slider') ||
                 document.querySelector('[class*="sliding-slider"]') ||
                 document.querySelector('.slider');
  if (!slider) {{
    return {{error: 'slider not found', html_snippet: document.body.innerHTML.slice(0,2000)}};
  }}
  const rect = slider.getBoundingClientRect();
  const startX = rect.left + rect.width/2;
  const startY = rect.top + rect.height/2;

  function dispatch(type, clientX, clientY) {{
    const ev = new MouseEvent(type, {{bubbles:true, cancelable:true, clientX, clientY, buttons:1}});
    slider.dispatchEvent(ev);
    // Also try pointer events
    try {{
      const pev = new PointerEvent(type.replace('mouse','pointer'), {{bubbles:true, cancelable:true, clientX, clientY, buttons:1, pointerId:1}});
      slider.dispatchEvent(pev);
    }} catch(e){{}}
    // Also dispatch on document for move
    if (type.includes('move')) {{
      document.dispatchEvent(new MouseEvent('mousemove', {{bubbles:true, clientX, clientY, buttons:1}}));
    }}
  }}

  // mousedown
  dispatch('mousedown', startX, startY);
  dispatch('pointerdown', startX, startY);
  await new Promise(r => setTimeout(r, 100 + Math.random()*80));

  for (const [sx, yJitter, delay] of pts) {{
    // Convert slider_x to clientX - we need to approximate
    // Slider track is 300px, slider is 40px wide, range [0,260] left.
    // We move by sx pixels from start (startX corresponds to slider_left=0)
    // So target clientX = startX + sx
    const clientX = startX + sx;
    const clientY = startY + yJitter;
    dispatch('mousemove', clientX, clientY);
    dispatch('pointermove', clientX, clientY);
    await new Promise(r => setTimeout(r, delay));
  }}

  await new Promise(r => setTimeout(r, 80 + Math.random()*60));
  const last = pts[pts.length-1];
  dispatch('mouseup', startX + last[0], startY + last[1]);
  dispatch('pointerup', startX + last[0], startY + last[1]);
  document.dispatchEvent(new MouseEvent('mouseup', {{bubbles:true, clientX: startX + last[0], clientY: startY}}));

  return {{success: true, final_slider_x: last[0], points: pts.length}};
}})()
"""
    return js


def get_intercept_js() -> str:
    """JS to intercept fetch/XHR and capture captcha init, images, verify calls."""
    return """
(() => {
  window.__capInitCalls = window.__capInitCalls || [];
  window.__capVerifCalls = window.__capVerifCalls || [];
  window.__capUploadCalls = window.__capUploadCalls || [];
  window.__signupCalls = window.__signupCalls || [];
  window.__capImages = window.__capImages || [];
  window.__capParams = window.__capParams || [];

  // Hook fetch
  const origFetch = window.fetch;
  window.fetch = async function(...args) {
    const [url, opts] = args;
    const urlStr = typeof url === 'string' ? url : url.url || '';
    try {
      const resp = await origFetch.apply(this, args);
      const clone = resp.clone();
      // Capture interesting calls
      if (urlStr.includes('captcha') || urlStr.includes('no8xfe') || urlStr.includes('sgp.aliyuncs.com') || urlStr.includes('/api/v1/auths/')) {
        clone.text().then(t => {
          try {
            const entry = {url: urlStr, method: opts?.method || 'GET', body: opts?.body?.slice?.(0,2000) || opts?.body, resp: t.slice(0,5000), ts: Date.now()};
            if (urlStr.includes('no8xfe') && (urlStr.includes('captcha-open') || opts?.body)) {
              window.__capInitCalls.push(entry);
            } else if (urlStr.includes('verify') || urlStr.includes('Verify')) {
              window.__capVerifCalls.push(entry);
            } else if (urlStr.includes('upload')) {
              window.__capUploadCalls.push(entry);
            } else if (urlStr.includes('/api/v1/auths/')) {
              window.__signupCalls.push(entry);
            }
            if (urlStr.includes('aliyuncs.com')) {
              // Image URL observed
              window.__capImages.push({url: urlStr, ts: Date.now()});
            }
          } catch(e){}
        }).catch(()=>{});
      }
      return resp;
    } catch(e) {
      if (urlStr.includes('captcha') || urlStr.includes('no8xfe')) {
        window.__capInitCalls.push({url: urlStr, error: e.message, ts: Date.now()});
      }
      throw e;
    }
  };

  // Hook XHR
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(method, url, ...rest) {
    this._capMethod = method;
    this._capUrl = url;
    return origOpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function(body) {
    const method = this._capMethod;
    const url = this._capUrl;
    if (url && (url.includes('captcha') || url.includes('no8xfe') || url.includes('/api/v1/auths/'))) {
      const xhr = this;
      const origOnReady = xhr.onreadystatechange;
      xhr.addEventListener('load', function() {
        try {
          const entry = {url, method, body: body?.slice?.(0,2000), resp: xhr.responseText?.slice(0,5000), status: xhr.status, ts: Date.now()};
          if (url.includes('captcha-open') || url.includes('CertifyId') || body?.includes('SceneId')) {
            window.__capInitCalls.push(entry);
          } else if (url.includes('verify') || url.includes('Verify')) {
            window.__capVerifCalls.push(entry);
          } else if (url.includes('/api/v1/auths/')) {
            window.__signupCalls.push(entry);
          }
        } catch(e){}
      });
    }
    return origSend.call(this, body);
  };

  // Hook initAliyunCaptcha to capture config and success callback
  const checkInit = () => {
    if (window.initAliyunCaptcha) {
      const origInit = window.initAliyunCaptcha;
      window.initAliyunCaptcha = function(config) {
        window.__capImages.push({type:'initConfig', config: JSON.stringify(config).slice(0,2000), ts: Date.now()});
        if (config && config.success) {
          const origSuccess = config.success;
          config.success = function(param) {
            window.__capParams.push({param, ts: Date.now()});
            console.log('[cap] success param captured, len', typeof param === 'string' ? param.length : JSON.stringify(param).length);
            return origSuccess.apply(this, arguments);
          };
        }
        window.__capLastConfig = config;
        return origInit.call(this, config);
      };
      console.log('[cap] initAliyunCaptcha hooked');
      return true;
    }
    return false;
  };
  if (!checkInit()) {
    const iv = setInterval(() => { if (checkInit()) clearInterval(iv); }, 200);
    setTimeout(() => clearInterval(iv), 10000);
  }

  // Observe DOM for captcha images
  const observer = new MutationObserver((muts) => {
    for (const m of muts) {
      for (const node of m.addedNodes) {
        if (node.nodeType === 1) {
          // Look for img tags with aliyuncs
          const imgs = node.querySelectorAll ? node.querySelectorAll('img') : [];
          for (const img of imgs) {
            const src = img.src || '';
            if (src.includes('aliyuncs.com') || src.includes('data:image')) {
              window.__capImages.push({url: src.slice(0,2000), tag: 'img', ts: Date.now(), isData: src.startsWith('data:')});
            }
          }
          // Also check for div with background image
          if (node.style && node.style.backgroundImage) {
            const bg = node.style.backgroundImage;
            if (bg.includes('aliyuncs.com') || bg.includes('data:image')) {
              window.__capImages.push({url: bg.slice(0,2000), tag: 'bg', ts: Date.now()});
            }
          }
        }
      }
    }
  });
  observer.observe(document.body, {childList: true, subtree: true});

  return {hooked: true};
})()
"""


def get_image_extract_js() -> str:
    """JS to extract captcha images from current DOM (base64 or URL)."""
    return """
(() => {
  const result = {main: null, puzzle: null, allImgs: []};

  // Method 1: look for aliyun captcha containers
  const containers = document.querySelectorAll('[id*="aliyun"], [class*="aliyun"], [class*="captcha"], [class*="feilin"]');
  // Method 2: all images
  const imgs = document.querySelectorAll('img');
  for (const img of imgs) {
    const src = img.src || '';
    if (!src) continue;
    const info = {src: src.slice(0,500), width: img.naturalWidth, height: img.naturalHeight, className: img.className, id: img.id, tag: 'img'};
    result.allImgs.push(info);
    // Heuristic: 300x300 main, Nx300 puzzle
    if (img.naturalWidth === 300 && img.naturalHeight === 300) {
      result.main = src;
    } else if (img.naturalHeight === 300 && img.naturalWidth < 100) {
      result.puzzle = src;
    }
    // Also check data: URLs that are large
    if (src.startsWith('data:') && src.length > 1000) {
      if (img.naturalWidth === 300 && img.naturalHeight === 300) result.main = src;
      if (img.naturalHeight === 300) result.puzzle = src;
    }
  }

  // Method 3: canvas elements (some implementations draw on canvas)
  const canvases = document.querySelectorAll('canvas');
  for (const c of canvases) {
    try {
      const dataUrl = c.toDataURL();
      result.allImgs.push({src: dataUrl.slice(0,200), width: c.width, height: c.height, tag: 'canvas'});
      if (c.width === 300 && c.height === 300 && !result.main) result.main = dataUrl;
    } catch(e){}
  }

  // Method 4: div background images
  const divs = document.querySelectorAll('div');
  for (const d of divs) {
    const bg = getComputedStyle(d).backgroundImage;
    if (bg && bg !== 'none' && (bg.includes('aliyuncs.com') || bg.includes('data:image'))) {
      result.allImgs.push({src: bg.slice(0,500), width: d.offsetWidth, height: d.offsetHeight, tag: 'bg', className: d.className});
    }
  }

  // Also include our intercepted images
  result.intercepted = window.__capImages || [];
  result.initCalls = (window.__capInitCalls || []).slice(-3);
  result.verifCalls = (window.__capVerifCalls || []).slice(-3);
  result.params = window.__capParams || [];

  // Try to get puzzle position info from DOM
  try {
    const puzzleEl = document.querySelector('[class*="puzzle"], [class*="slider-item"], [class*="feilin"] div[style*="left"]') || document.querySelector('#aliyunCaptcha-puzzle');
    if (puzzleEl) result.puzzleElStyle = puzzleEl.style.cssText;
  } catch(e){}

  return result;
})()
"""


def get_slider_state_js() -> str:
    return """
(() => {
  const slider = document.querySelector('#aliyunCaptcha-sliding-slider') || document.querySelector('.aliyunCaptcha-sliding-slider');
  const track = document.querySelector('#aliyunCaptcha-sliding-track') || document.querySelector('.aliyunCaptcha-track');
  const puzzle = document.querySelector('#aliyunCaptcha-puzzle') || document.querySelector('[class*="puzzle"]');
  return {
    slider: slider ? {left: slider.style.left, rect: slider.getBoundingClientRect(), className: slider.className} : null,
    track: track ? {rect: track.getBoundingClientRect()} : null,
    puzzle: puzzle ? {left: puzzle.style.left, rect: puzzle.getBoundingClientRect()} : null,
    allSliderEls: Array.from(document.querySelectorAll('[class*="slider"]')).map(el => ({class: el.className, left: el.style.left, rect: {x: el.getBoundingClientRect().x, y: el.getBoundingClientRect().y, w: el.getBoundingClientRect().width}})).slice(0,10)
  };
})()
"""
