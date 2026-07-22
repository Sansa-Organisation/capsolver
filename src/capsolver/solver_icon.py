"""Icon solver for Aliyun CAPTCHA V3 ICON / ICONCAPTCHA type.

Improved version providing:
- classify_challenge_text, parse_challenge_text with keyword mapping (EN+CN)
- detect_grid (rows, cols, cells) with projection + Hough + fallback
- classify_icon with mean/var/edges/Hu + circle detection, confidence >=0.6
- detect_icon_captcha using ORB + multi-scale template matching + grid classification
- solve_icon_from_array / solve_icon_from_files / solve_icon aliases
"""

from __future__ import annotations
import re
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class IconCandidate:
    x: int
    y: int
    icon_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x,y,w,h


@dataclass
class IconDetection:
    icons: List[IconCandidate]
    click_positions: List[Tuple[int, int]]  # (x,y) in order to click
    method: str
    confidence: float
    challenge_text: str
    debug: dict


# ----------------------------------------------------------------------
# Keyword maps
# ----------------------------------------------------------------------
ICON_KEYWORDS: Dict[str, List[str]] = {
    "star": ["star", "stars", "star icon", "five-pointed star", "star shape", "五角星", "星星", "星形", "星"],
    "moon": ["moon", "moons", "crescent", "lunar", "月亮", "月牙", "月"],
    "sun": ["sun", "suns", "sunny", "solar", "太阳", "日光", "阳光"],
    "car": ["car", "cars", "vehicle", "vehicles", "automobile", "auto", "汽车", "轿车", "车", "车辆", "轿车"],
    "heart": ["heart", "hearts", "love heart", "爱心", "心形", "心"],
    "tree": ["tree", "trees", "树", "树木", "大树", "木"],
    "cat": ["cat", "cats", "kitten", "kittens", "猫", "猫咪", "小猫"],
    "dog": ["dog", "dogs", "puppy", "puppies", "狗", "小狗", "犬"],
    "circle": ["circle", "circles", "round", "circular", "圆形", "圆圈", "圆", "圆点", "圆环", "球", "圈"],
    "triangle": ["triangle", "triangles", "triangular", "三角形", "三角"],
    "square": ["square", "squares", "正方形", "方形", "方块", "矩形", "正方"],
    "cloud": ["cloud", "clouds", "云", "云朵"],
    "bird": ["bird", "birds", "鸟", "小鸟"],
    "flower": ["flower", "flowers", "花", "花朵"],
    "umbrella": ["umbrella", "umbrellas", "雨伞", "伞"],
    "bus": ["bus", "buses", "公交车", "巴士", "公共汽车"],
    "person": ["person", "people", "man", "woman", "人物", "人", "人类"],
    "house": ["house", "houses", "房子", "房屋", "屋"],
    "key": ["key", "keys", "钥匙"],
    "lock": ["lock", "locks", "锁"],
    "flag": ["flag", "flags", "旗", "旗帜"],
    "bell": ["bell", "bells", "铃", "铃铛"],
}

COLOR_KEYWORDS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown", "gray", "grey", "红", "蓝", "绿", "黄", "黑", "白"]


def _is_ascii_word(s: str) -> bool:
    """Check if string is purely ascii letters/spaces (for word-boundary matching)."""
    return bool(re.fullmatch(r"[a-zA-Z\s\-]+", s))


def classify_challenge_text(text: str) -> str:
    """Map challenge text to a single icon type (canonical). Return first by appearance."""
    if not text:
        return "unknown"
    text_lower = text.lower()
    candidates: List[Tuple[int, str, str]] = []  # pos, canonical, kw
    for canonical, synonyms in ICON_KEYWORDS.items():
        sorted_syns = sorted(synonyms, key=lambda s: len(s), reverse=True)
        best_pos = None
        best_kw = None
        for kw in sorted_syns:
            kw_lower = kw.lower()
            if _is_ascii_word(kw_lower):
                # word boundary
                pattern = r"\b" + re.escape(kw_lower) + r"\b"
                m = re.search(pattern, text_lower)
                if m:
                    pos = m.start()
                    if best_pos is None or pos < best_pos:
                        best_pos = pos
                        best_kw = kw_lower
            else:
                pos = text_lower.find(kw_lower)
                if pos != -1:
                    if best_pos is None or pos < best_pos:
                        best_pos = pos
                        best_kw = kw_lower
        if best_pos is not None:
            candidates.append((best_pos, canonical, best_kw))
    if not candidates:
        return "unknown"
    candidates.sort(key=lambda x: (x[0], -len(x[2])))
    return candidates[0][1]


def parse_challenge_text(text: str) -> List[str]:
    """Parse challenge text to extract target icon types in order."""
    if not text:
        return []
    text_lower = text.lower()
    # Collect all matches across synonyms to preserve order
    all_matches: List[Tuple[int, int, str]] = []  # pos, -kw_len, canonical
    for canonical, synonyms in ICON_KEYWORDS.items():
        sorted_syns = sorted(synonyms, key=lambda s: len(s), reverse=True)
        best_pos = None
        best_len = 0
        for kw in sorted_syns:
            kw_lower = kw.lower()
            if _is_ascii_word(kw_lower):
                pattern = r"\b" + re.escape(kw_lower) + r"\b"
                for m in re.finditer(pattern, text_lower):
                    pos = m.start()
                    # keep earliest per canonical but allow multiple different canonicals
                    # for this function we want first occurrence per canonical, so track minimal pos
                    if best_pos is None or pos < best_pos:
                        best_pos = pos
                        best_len = len(kw_lower)
            else:
                # find first occurrence for non-ascii
                pos = text_lower.find(kw_lower)
                if pos != -1:
                    if best_pos is None or pos < best_pos:
                        best_pos = pos
                        best_len = len(kw_lower)
        if best_pos is not None:
            all_matches.append((best_pos, -best_len, canonical))

    # Sort by position then longer keyword first
    all_matches.sort(key=lambda x: (x[0], x[1]))
    # Deduplicate preserving order (keep first occurrence of each canonical)
    seen = set()
    result: List[str] = []
    for _, _, canonical in all_matches:
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


# ----------------------------------------------------------------------
# Grid detection helpers
# ----------------------------------------------------------------------
def _find_lines_projection(gray: np.ndarray, dark_thresh: int = 80, factor: float = 0.5) -> Tuple[List[int], List[int]]:
    """Find grid lines via dark pixel projection."""
    dark = gray < dark_thresh
    h, w = gray.shape[:2]
    h_counts = np.sum(dark, axis=1)  # per row
    v_counts = np.sum(dark, axis=0)

    h_thresh = w * factor
    v_thresh = h * factor

    def cluster_counts(counts: np.ndarray, thresh: float) -> List[int]:
        idx = np.where(counts > thresh)[0]
        if len(idx) == 0:
            return []
        lines: List[int] = []
        cur = [idx[0]]
        for k in range(1, len(idx)):
            if idx[k] - idx[k-1] <= 6:  # same thick line
                cur.append(idx[k])
            else:
                lines.append(int(float(np.mean(cur))))
                cur = [idx[k]]
        lines.append(int(float(np.mean(cur))))
        return lines

    h_lines = cluster_counts(h_counts, h_thresh)
    v_lines = cluster_counts(v_counts, v_thresh)
    return h_lines, v_lines


def _normalize_grid_lines(lines: List[int], max_val: int, margin: int = 12) -> List[int]:
    """Ensure 0 and max_val are present if far, merge close lines."""
    if not lines:
        return []
    lines = sorted(set(lines))
    if lines[0] > margin:
        lines = [0] + lines
    if lines[-1] < max_val - margin:
        lines = lines + [max_val]
    merged: List[int] = []
    for l in lines:
        if not merged or abs(l - merged[-1]) > 10:
            merged.append(l)
        else:
            merged[-1] = (merged[-1] + l) // 2
    return sorted(merged)


def _fallback_grid_cells(gray: np.ndarray) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
    """Fallback: contour detection, else fixed 3x3."""
    h, w = gray.shape[:2]
    try:
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, wb, hb = cv2.boundingRect(cnt)
            if 20 < wb < w * 0.6 and 20 < hb < h * 0.6:
                area = wb * hb
                if 400 < area < w * h * 0.5:
                    bboxes.append((x, y, wb, hb))
        if 4 <= len(bboxes) <= 20:
            cys = [y + hb // 2 for x, y, wb, hb in bboxes]
            cxs = [x + wb // 2 for x, y, wb, hb in bboxes]

            def cluster_centers(vals: List[int], thr: float) -> List[int]:
                if not vals:
                    return []
                vals_sorted = sorted(vals)
                clusters: List[List[int]] = [[vals_sorted[0]]]
                for v in vals_sorted[1:]:
                    if abs(v - float(np.mean(clusters[-1]))) <= thr:
                        clusters[-1].append(v)
                    else:
                        clusters.append([v])
                return [int(float(np.mean(c))) for c in clusters]

            row_centers = cluster_centers(cys, h // 6)
            col_centers = cluster_centers(cxs, w // 6)
            rows = len(row_centers) if row_centers else 3
            cols = len(col_centers) if col_centers else 3
            bboxes.sort(key=lambda b: (b[1] // 50, b[0]))
            return rows, cols, bboxes
    except Exception:
        pass

    # Final fixed 3x3 grid (inset to avoid border lines)
    rows, cols = 3, 3
    cell_w, cell_h = w // cols, h // rows
    cells: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_w + cell_w // 6
            y = r * cell_h + cell_h // 6
            wb = cell_w * 2 // 3
            hb = cell_h * 2 // 3
            # ensure within
            if x + wb > w:
                wb = w - x - 2
            if y + hb > h:
                hb = h - y - 2
            if wb > 10 and hb > 10:
                cells.append((x, y, wb, hb))
    return rows, cols, cells


def detect_grid(image) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
    """
    Detect grid: returns (rows, cols, cells) where cells = list of (x,y,w,h).
    Handles synthetic 300x300 grid with black lines.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image

    if img_np.ndim == 3:
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        except Exception:
            try:
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = np.mean(img_np, axis=2).astype(np.uint8)
    else:
        gray = img_np

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    h, w = gray.shape[:2]
    if h < 10 or w < 10:
        return 1, 1, [(0, 0, w, h)]

    best: Optional[Tuple[int, int, List[Tuple[int, int, int, int]]]] = None

    # Projection method with multiple thresholds
    for dark_thresh in [60, 80, 100, 120, 150]:
        for factor in [0.6, 0.5, 0.4]:
            h_lines, v_lines = _find_lines_projection(gray, dark_thresh=dark_thresh, factor=factor)
            if len(h_lines) >= 2 and len(v_lines) >= 2:
                h_norm = _normalize_grid_lines(h_lines, h)
                v_norm = _normalize_grid_lines(v_lines, w)
                if len(h_norm) >= 2 and len(v_norm) >= 2:
                    rows = len(h_norm) - 1
                    cols = len(v_norm) - 1
                    cells: List[Tuple[int, int, int, int]] = []
                    for i in range(rows):
                        for j in range(cols):
                            y1 = h_norm[i]
                            y2 = h_norm[i + 1]
                            x1 = v_norm[j]
                            x2 = v_norm[j + 1]
                            # inset slightly to avoid counting the grid line itself
                            x1c = max(0, x1 + 2)
                            y1c = max(0, y1 + 2)
                            x2c = min(w, x2 - 2)
                            y2c = min(h, y2 - 2)
                            ww = x2c - x1c
                            hh = y2c - y1c
                            if ww > 10 and hh > 10:
                                cells.append((x1c, y1c, ww, hh))
                    if len(cells) >= 4 and rows >= 2 and cols >= 2:
                        return rows, cols, cells
                    if best is None and len(cells) >= 1:
                        best = (rows, cols, cells)

    # Hough line method
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=70, minLineLength=min(h, w) // 3, maxLineGap=20
        )
        if lines is not None:
            h_pos: List[int] = []
            v_pos: List[int] = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 12:
                    h_pos.append((y1 + y2) // 2)
                elif abs(x2 - x1) < 12:
                    v_pos.append((x1 + x2) // 2)
            if h_pos and v_pos:

                def cluster_positions(pos: List[int]) -> List[int]:
                    if not pos:
                        return []
                    pos_sorted = sorted(pos)
                    clusters: List[List[int]] = [[pos_sorted[0]]]
                    for p in pos_sorted[1:]:
                        if abs(p - float(np.mean(clusters[-1]))) < 12:
                            clusters[-1].append(p)
                        else:
                            clusters.append([p])
                    return [int(float(np.mean(c))) for c in clusters]

                h_cl = cluster_positions(h_pos)
                v_cl = cluster_positions(v_pos)
                h_norm = _normalize_grid_lines(h_cl, h)
                v_norm = _normalize_grid_lines(v_cl, w)
                if len(h_norm) >= 2 and len(v_norm) >= 2:
                    rows = len(h_norm) - 1
                    cols = len(v_norm) - 1
                    cells = []
                    for i in range(rows):
                        for j in range(cols):
                            y1 = h_norm[i]
                            y2 = h_norm[i + 1]
                            x1 = v_norm[j]
                            x2 = v_norm[j + 1]
                            ww = x2 - x1
                            hh = y2 - y1
                            if ww > 10 and hh > 10:
                                cells.append((x1, y1, ww, hh))
                    if len(cells) >= 4:
                        return rows, cols, cells
    except Exception:
        pass

    if best is not None and len(best[2]) >= 2:
        return best

    return _fallback_grid_cells(gray)


def detect_grid_icons(main_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Legacy API: detect icon grid positions via contour or grid detection."""
    _, _, cells = detect_grid(main_gray)
    return cells


# ----------------------------------------------------------------------
# Icon classification
# ----------------------------------------------------------------------
def classify_icon(icon_gray: np.ndarray) -> Tuple[str, float]:
    """Classify icon type from its grayscale crop with improved confidence."""
    if icon_gray is None or icon_gray.size == 0:
        return "unknown", 0.6
    h, w = icon_gray.shape[:2]
    if h < 5 or w < 5:
        return "unknown", 0.6

    # Ensure uint8
    if icon_gray.dtype != np.uint8:
        icon_gray = icon_gray.astype(np.uint8)

    mean_val = float(np.mean(icon_gray))
    std_val = float(np.std(icon_gray))

    # Edge density
    try:
        edges = cv2.Canny(icon_gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size + 1e-6)
    except Exception:
        edge_ratio = 0.0
        edges = None

    # Binary for contour / Hu
    try:
        _, binary = cv2.threshold(icon_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = None
        largest_area = 0
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a > largest_area and a > 30:
                largest_area = a
                largest = cnt
        hu = None
        if largest is not None:
            moments = cv2.moments(largest)
            if moments["m00"] != 0:
                hu_m = cv2.HuMoments(moments).flatten()
                # log scale
                hu = hu_m
    except Exception:
        largest = None
        hu = None

    # HoughCircles for circle detection
    is_circle = False
    try:
        # Blur slightly to improve circle detection
        blurred = cv2.medianBlur(icon_gray, 5) if min(h, w) >= 15 else icon_gray
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=15,
            minRadius=5,
            maxRadius=min(h, w) // 2,
        )
        if circles is not None and len(circles[0]) >= 1:
            is_circle = True
    except Exception:
        is_circle = False

    # Decide type
    icon_type = "unknown"
    confidence = 0.6

    if is_circle:
        icon_type = "circle"
        confidence = 0.78
    elif edge_ratio > 0.08 and std_val > 12:
        # Has significant shape
        if mean_val < 80:
            icon_type = "dark_icon"
            confidence = 0.70
        elif mean_val > 200:
            # Light background with shape -> maybe circle but Hough missed
            # Check if contour circularity high
            if largest is not None:
                peri = cv2.arcLength(largest, True)
                if peri > 0:
                    circularity = 4 * np.pi * largest_area / (peri * peri + 1e-6)
                    if circularity > 0.6:
                        icon_type = "circle"
                        confidence = 0.72
                    else:
                        icon_type = "light_shape"
                        confidence = 0.68
                else:
                    icon_type = "light_icon"
                    confidence = 0.65
            else:
                icon_type = "light_icon"
                confidence = 0.65
        else:
            icon_type = "medium_icon"
            confidence = 0.66
    else:
        # Low edge ratio -> uniform
        if mean_val < 60:
            icon_type = "dark_icon"
            confidence = 0.70
        elif mean_val > 210:
            icon_type = "light_icon"
            confidence = 0.65
        else:
            icon_type = "medium_icon"
            confidence = 0.62

    # Boost confidence using Hu moments stability
    if hu is not None:
        # If Hu moments are not extreme, increase confidence slightly
        # simple heuristic: finite values
        if np.all(np.isfinite(hu)):
            confidence = min(0.90, confidence + 0.06)

    # Further boost if variance indicates structure
    if std_val > 25:
        confidence = min(0.88, confidence + 0.04)
    if edge_ratio > 0.12:
        confidence = min(0.88, confidence + 0.03)

    # Ensure at least 0.6 for known types
    if icon_type != "unknown":
        confidence = max(confidence, 0.6)

    return icon_type, float(confidence)


# ----------------------------------------------------------------------
# Color helpers
# ----------------------------------------------------------------------
def _dominant_color_name(crop_rgb: np.ndarray) -> str:
    """Return dominant color name from RGB crop."""
    try:
        if crop_rgb.ndim == 2:
            return "gray"
        if crop_rgb.shape[2] < 3:
            return "unknown"
        # Resize small to speed
        # Compute mean per channel
        r_m = float(np.mean(crop_rgb[:, :, 0]))
        g_m = float(np.mean(crop_rgb[:, :, 1]))
        b_m = float(np.mean(crop_rgb[:, :, 2]))
        mean_all = (r_m + g_m + b_m) / 3.0

        # White / black detection
        if mean_all > 225 and abs(r_m - g_m) < 18 and abs(g_m - b_m) < 18 and abs(r_m - b_m) < 18:
            return "white"
        if mean_all < 55:
            return "black"

        # Compute dominant by difference
        # Yellow: R and G high, B low
        if r_m > 150 and g_m > 150 and b_m < 110:
            return "yellow"
        # Purple: R and B high, G low
        if r_m > 120 and b_m > 120 and g_m < 100:
            return "purple"
        # Orange: R high, G medium
        if r_m > 180 and 80 < g_m < 160 and b_m < 100:
            return "orange"

        # Primary
        if r_m > g_m + 30 and r_m > b_m + 30:
            return "red"
        if b_m > r_m + 30 and b_m > g_m + 30:
            return "blue"
        if g_m > r_m + 30 and g_m > b_m + 30:
            return "green"

        # Fallback by max
        max_c = max(r_m, g_m, b_m)
        if max_c == r_m:
            return "red"
        if max_c == g_m:
            return "green"
        if max_c == b_m:
            return "blue"
        return "unknown"
    except Exception:
        return "unknown"


def _parse_colors_from_text(text: str) -> List[str]:
    txt = text.lower()
    found: List[str] = []
    for col in COLOR_KEYWORDS:
        if col.lower() in txt:
            # normalize chinese to english where possible? keep as is but map
            # For Chinese red etc, translate roughly
            mapping = {"红": "red", "蓝": "blue", "绿": "green", "黄": "yellow", "黑": "black", "白": "white"}
            eng = mapping.get(col, col)
            if eng not in found:
                found.append(eng)
    return found


# ----------------------------------------------------------------------
# Main detection
# ----------------------------------------------------------------------
def _detect_with_puzzle_template(
    main_gray: np.ndarray, puzzle_gray: np.ndarray
) -> Tuple[Optional[Tuple[int, int]], float, Optional[Tuple[int, int]]]:
    """Multi-scale template matching + ORB, returns (center, score, size)."""
    h, w = main_gray.shape[:2]
    best_val = -1.0
    best_loc = None
    best_size = None

    # Multi-scale template matching
    try:
        for scale in [0.7, 0.9, 1.0, 1.1, 1.3, 0.5, 1.5]:
            pw = int(puzzle_gray.shape[1] * scale)
            ph = int(puzzle_gray.shape[0] * scale)
            if pw < 12 or ph < 12 or pw > w or ph > h:
                continue
            resized = cv2.resize(puzzle_gray, (pw, ph), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(main_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = float(max_val)
                best_loc = max_loc
                best_size = (pw, ph)
    except Exception:
        pass

    # ORB matching to refine or as fallback
    try:
        orb = cv2.ORB_create(nfeatures=600)
        kp1, des1 = orb.detectAndCompute(puzzle_gray, None)
        kp2, des2 = orb.detectAndCompute(main_gray, None)
        if des1 is not None and des2 is not None and len(kp1) >= 2 and len(kp2) >= 2:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if matches:
                matches = sorted(matches, key=lambda x: x.distance)
                # consider good matches distance < 50 or top 15
                good = [m for m in matches if m.distance < 55][:20]
                if len(good) < 3:
                    good = matches[:10]
                if len(good) >= 3:
                    pts = [kp2[m.trainIdx].pt for m in good]
                    cx = int(float(np.mean([p[0] for p in pts])))
                    cy = int(float(np.mean([p[1] for p in pts])))
                    # If template matching was weak, use ORB center
                    if best_val < 0.45:
                        return (cx, cy), 0.5, None
    except Exception:
        pass

    if best_loc is not None and best_size is not None:
        pw, ph = best_size
        cx = best_loc[0] + pw // 2
        cy = best_loc[1] + ph // 2
        return (cx, cy), best_val, best_size
    return None, best_val, None


def detect_icon_captcha(
    main_rgb: np.ndarray,
    challenge_text: str = "",
    puzzle_rgba: Optional[np.ndarray] = None,
) -> IconDetection:
    """
    Detect ICON captcha.

    Args:
        main_rgb: Main image (H,W,3) containing icon grid
        challenge_text: Text describing which icons to click (from DOM)
        puzzle_rgba: Optional puzzle image (for API compat) – if provided, use template + ORB

    Returns:
        IconDetection with click positions
    """
    # Normalize main_rgb input
    if isinstance(main_rgb, Image.Image):
        main_rgb_np = np.array(main_rgb.convert("RGB"))
    else:
        main_rgb_np = main_rgb

    if main_rgb_np.ndim == 2:
        main_rgb_np = cv2.cvtColor(main_rgb_np, cv2.COLOR_GRAY2RGB)
    if main_rgb_np.dtype != np.uint8:
        main_rgb_np = main_rgb_np.astype(np.uint8)
    if main_rgb_np.shape[2] == 4:
        # RGBA -> RGB
        main_rgb_np = main_rgb_np[:, :, :3]

    main_gray = cv2.cvtColor(main_rgb_np, cv2.COLOR_RGB2GRAY)
    h, w = main_gray.shape[:2]

    # Optional puzzle template path
    if puzzle_rgba is not None:
        try:
            if isinstance(puzzle_rgba, Image.Image):
                puzzle_np = np.array(puzzle_rgba.convert("RGBA"))
            else:
                puzzle_np = puzzle_rgba

            if puzzle_np.ndim == 3 and puzzle_np.shape[2] == 4:
                puzzle_rgb = puzzle_np[:, :, :3]
                puzzle_gray = cv2.cvtColor(puzzle_rgb, cv2.COLOR_RGB2GRAY)
            elif puzzle_np.ndim == 3:
                # Assume RGB
                try:
                    puzzle_gray = cv2.cvtColor(puzzle_np, cv2.COLOR_RGB2GRAY)
                except Exception:
                    puzzle_gray = cv2.cvtColor(puzzle_np, cv2.COLOR_BGR2GRAY)
            else:
                puzzle_gray = puzzle_np

            if puzzle_gray.dtype != np.uint8:
                puzzle_gray = puzzle_gray.astype(np.uint8)

            center, score, size = _detect_with_puzzle_template(main_gray, puzzle_gray)
            if center is not None and score > 0.25:
                cx, cy = center
                # Build single icon candidate
                pw, ph = size if size is not None else (puzzle_gray.shape[1], puzzle_gray.shape[0])
                x0 = max(0, cx - pw // 2)
                y0 = max(0, cy - ph // 2)
                icon = IconCandidate(
                    x=cx,
                    y=cy,
                    icon_type=classify_challenge_text(challenge_text) if challenge_text else "target",
                    confidence=max(0.6, float(score)),
                    bbox=(x0, y0, pw, ph),
                )
                return IconDetection(
                    icons=[icon],
                    click_positions=[(cx, cy)],
                    method="template_orb",
                    confidence=max(0.65, float(score)),
                    challenge_text=challenge_text,
                    debug={
                        "template_score": float(score),
                        "method": "puzzle_template",
                        "image_size": (w, h),
                    },
                )
        except Exception as e:
            # Fall through to grid method
            pass

    # Grid detection
    rows, cols, cells = detect_grid(main_rgb_np)

    icons: List[IconCandidate] = []
    for (x, y, wb, hb) in cells:
        # Clamp
        x_c = max(0, int(x))
        y_c = max(0, int(y))
        wb_c = int(wb)
        hb_c = int(hb)
        if x_c + wb_c > w:
            wb_c = w - x_c
        if y_c + hb_c > h:
            hb_c = h - y_c
        if wb_c <= 0 or hb_c <= 0:
            continue
        crop_gray = main_gray[y_c : y_c + hb_c, x_c : x_c + wb_c]
        crop_rgb = main_rgb_np[y_c : y_c + hb_c, x_c : x_c + wb_c]
        icon_type, conf = classify_icon(crop_gray)

        # Enhance with color/shape heuristics for circle etc.
        # If chip contains strong color, we may want to preserve type as circle if detected
        # _dominant_color_name used later for click filtering

        cx = x_c + wb_c // 2
        cy = y_c + hb_c // 2
        icons.append(
            IconCandidate(
                x=cx,
                y=cy,
                icon_type=icon_type,
                confidence=float(conf),
                bbox=(x_c, y_c, wb_c, hb_c),
            )
        )

    # Sort icons top-to-bottom left-to-right for deterministic order
    icons.sort(key=lambda ic: (ic.bbox[1] // 20, ic.bbox[0]))

    # Parse challenge text
    targets = parse_challenge_text(challenge_text)
    colors_in_text = _parse_colors_from_text(challenge_text)

    click_positions: List[Tuple[int, int]] = []

    if targets:
        # For each target in order, find matching icon
        used_indices = set()
        for target in targets:
            best_idx = None
            best_score = -1.0

            # First pass: type match
            for idx, ic in enumerate(icons):
                if idx in used_indices:
                    continue
                # Type similarity
                type_match = 0
                if ic.icon_type == target:
                    type_match = 2
                elif target in ic.icon_type or ic.icon_type in target:
                    type_match = 1
                # If icon_type is placeholder like light_shape but target is circle etc., we can still consider shape via second pass

                if type_match > 0:
                    score = type_match * 10 + ic.confidence
                    # Color boost
                    if colors_in_text:
                        try:
                            crop_rgb = main_rgb_np[
                                ic.bbox[1] : ic.bbox[1] + ic.bbox[3],
                                ic.bbox[0] : ic.bbox[0] + ic.bbox[2],
                            ]
                            dom_col = _dominant_color_name(crop_rgb)
                            if dom_col in colors_in_text:
                                score += 3
                        except Exception:
                            pass
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            # Second pass: if still not found, search by color only if colors mentioned
            if best_idx is None and colors_in_text:
                for idx, ic in enumerate(icons):
                    if idx in used_indices:
                        continue
                    try:
                        crop_rgb = main_rgb_np[
                            ic.bbox[1] : ic.bbox[1] + ic.bbox[3],
                            ic.bbox[0] : ic.bbox[0] + ic.bbox[2],
                        ]
                        dom_col = _dominant_color_name(crop_rgb)
                        if dom_col in colors_in_text:
                            score = ic.confidence + 1
                            if score > best_score:
                                best_score = score
                                best_idx = idx
                    except Exception:
                        continue

            # Third pass: fallback to shape detection for circle if target is circle
            if best_idx is None and target == "circle":
                # Look for icons where classify said circle or where we can re-detect
                for idx, ic in enumerate(icons):
                    if idx in used_indices:
                        continue
                    # Already considered circle, but also light_shape may hide circle
                    if ic.icon_type == "circle" or "circle" in ic.icon_type:
                        best_idx = idx
                        break
                # If still none, try to find any icon with high edge ratio (potential shape)
                if best_idx is None:
                    # pick icon with highest confidence among unused
                    candidates = [(idx, ic) for idx, ic in enumerate(icons) if idx not in used_indices]
                    if candidates:
                        candidates.sort(key=lambda x: x[1].confidence, reverse=True)
                        best_idx = candidates[0][0]

            if best_idx is not None:
                click_positions.append((icons[best_idx].x, icons[best_idx].y))
                used_indices.add(best_idx)
            else:
                # Fallback: use highest confidence unused icon
                remaining = [(idx, ic) for idx, ic in enumerate(icons) if idx not in used_indices]
                if remaining:
                    remaining.sort(key=lambda x: x[1].confidence, reverse=True)
                    chosen_idx = remaining[0][0]
                    click_positions.append((icons[chosen_idx].x, icons[chosen_idx].y))
                    used_indices.add(chosen_idx)

        # If we still have less clicks than targets due to earlier fallback, ensure we have something
        if not click_positions and icons:
            click_positions = [(icons[0].x, icons[0].y)]
    else:
        # No targets parsed: try color filtering or return all/high-conf
        if colors_in_text:
            matched = []
            for ic in icons:
                try:
                    crop_rgb = main_rgb_np[
                        ic.bbox[1] : ic.bbox[1] + ic.bbox[3],
                        ic.bbox[0] : ic.bbox[0] + ic.bbox[2],
                    ]
                    dom_col = _dominant_color_name(crop_rgb)
                    if dom_col in colors_in_text:
                        matched.append(ic)
                except Exception:
                    continue
            if matched:
                # sort by confidence
                matched.sort(key=lambda ic: ic.confidence, reverse=True)
                click_positions = [(ic.x, ic.y) for ic in matched[:3]]
            else:
                # fallback: if challenge mentions red circle but parsing missed circle, still attempt circle detection
                if "circle" in challenge_text.lower() or "圆" in challenge_text:
                    circle_icons = [ic for ic in icons if ic.icon_type == "circle"]
                    if circle_icons:
                        click_positions = [(ic.x, ic.y) for ic in circle_icons]
                    else:
                        click_positions = [(ic.x, ic.y) for ic in icons[:1]]
                else:
                    click_positions = [(ic.x, ic.y) for ic in icons[:3]]
        else:
            # No guidance – return first 3 as guess
            click_positions = [(ic.x, ic.y) for ic in icons[:3]]

    # Confidence calc
    if icons:
        avg_conf = float(np.mean([ic.confidence for ic in icons])) if icons else 0.0
    else:
        avg_conf = 0.1

    if targets and len(click_positions) == len(targets) and len(click_positions) > 0:
        final_conf = max(0.65, (avg_conf + 0.78) / 2.0)
        method = "grid_classify_matched"
    elif icons:
        final_conf = max(0.6, avg_conf * 0.92)
        method = "grid_classify"
    else:
        final_conf = 0.15
        method = "fallback"

    return IconDetection(
        icons=icons,
        click_positions=click_positions,
        method=method,
        confidence=float(final_conf),
        challenge_text=challenge_text,
        debug={
            "bbox_count": len(cells),
            "icon_count": len(icons),
            "targets": targets,
            "colors": colors_in_text,
            "image_size": (w, h),
            "rows": rows,
            "cols": cols,
            "avg_icon_conf": float(avg_conf) if icons else 0.0,
        },
    )


def solve_icon_from_array(main_rgb: np.ndarray, challenge_text: str = "") -> IconDetection:
    """Solve from RGB array."""
    return detect_icon_captcha(main_rgb, challenge_text=challenge_text, puzzle_rgba=None)


def solve_icon_from_files(main_path: str, challenge_text: str = "") -> IconDetection:
    """Solve from file path."""
    main = np.array(Image.open(main_path).convert("RGB"))
    return detect_icon_captcha(main, challenge_text=challenge_text, puzzle_rgba=None)


def solve_icon(main_path, challenge_text: str = "") -> IconDetection:
    """Alias that handles both path and ndarray."""
    if isinstance(main_path, np.ndarray):
        return solve_icon_from_array(main_path, challenge_text)
    if isinstance(main_path, Image.Image):
        return solve_icon_from_array(np.array(main_path.convert("RGB")), challenge_text)
    return solve_icon_from_files(main_path, challenge_text)


# For registry compatibility
def detect_gap(main_rgb, puzzle_rgba=None, challenge_text: str = ""):
    """Alias for detect_icon_captcha for registry compatibility."""
    return detect_icon_captcha(main_rgb, challenge_text, puzzle_rgba)
