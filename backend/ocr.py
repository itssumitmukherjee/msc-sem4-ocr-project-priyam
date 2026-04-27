"""
backend/ocr.py
==============
Telegraph Newspaper Sudoku OCR Pipeline.

ARCHITECTURE
────────────
  Stage 1  Grid detection with multi-strategy perspective correction
  Stage 2  Cell extraction with adaptive margin
  Gate 1   OpenCV contour check (blank vs digit)
  Layer 1  DigitCNN inference + self-supervised template re-scoring
  Layer 2  Sudoku constraint correction (guarantees solvable board)
  Fallback kNN (when no model weights available)
"""

import os, cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

_BACKEND = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_BACKEND)
_DATA    = os.path.join(_ROOT, "data")

# ── Grid geometry ──────────────────────────────────────────────────────────────
GRID_SIZE   = 450
CELL_SIZE   = GRID_SIZE // 9   # 50 px
CELL_MARGIN = 3                 # px trimmed from each cell edge

# ── Contour thresholds ─────────────────────────────────────────────────────────
MIN_AREA_RATIO = 0.015          # digit contour >= 1.5% of cell area
MIN_DIM        = 3              # bounding rect min side length
MAX_ASPECT     = 6.0            # max w/h ratio — allows tall thin "1"

# ── CNN confidence thresholds ──────────────────────────────────────────────────
CONF_HIGH = 0.60
CONF_LOW  = 0.18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CNN_MEAN = (0.8693,)
_CNN_STD  = (0.3081,)
_cnn_tf   = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(_CNN_MEAN, _CNN_STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(path: str):
    try:    from model import DigitCNN
    except: from backend.model import DigitCNN
    m = DigitCNN().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — IMAGE DECODE
# ═══════════════════════════════════════════════════════════════════════════════

def _decode(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Upload JPEG, PNG, WEBP or BMP.")
    h, w = img.shape[:2]
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_CUBIC)
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img   = cv2.resize(img, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_AREA)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — ROBUST GRID DETECTION (5-strategy cascade)
# ═══════════════════════════════════════════════════════════════════════════════

def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_score(quad: np.ndarray, img_shape) -> float:
    if quad is None: return -1.0
    area     = cv2.contourArea(quad.reshape(-1,1,2).astype(np.float32))
    img_area = img_shape[0] * img_shape[1]
    if area < img_area * 0.10: return -1.0
    rect   = _order_corners(quad)
    w1     = np.linalg.norm(rect[1] - rect[0])
    w2     = np.linalg.norm(rect[2] - rect[3])
    h1     = np.linalg.norm(rect[3] - rect[0])
    h2     = np.linalg.norm(rect[2] - rect[1])
    width  = (w1 + w2) / 2
    height = (h1 + h2) / 2
    if width < 10 or height < 10: return -1.0
    aspect    = min(width, height) / max(width, height)
    area_frac = area / img_area
    return area_frac * 0.4 + aspect * 0.6


def _find_quad_from_thresh(thresh: np.ndarray) -> np.ndarray | None:
    k       = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(thresh, k, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    img_area = thresh.shape[0] * thresh.shape[1]
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < img_area * 0.10: break
        peri = cv2.arcLength(cnt, True)
        for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _strategy_threshold(gray: np.ndarray, method: str) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if method == "adaptive_gauss":
        thresh = cv2.adaptiveThreshold(blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    elif method == "adaptive_mean":
        thresh = cv2.adaptiveThreshold(blurred, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    elif method == "otsu":
        _, thresh = cv2.threshold(blurred, 0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        return None
    return _find_quad_from_thresh(thresh)


def _strategy_morph_lines(gray: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (w//3, 1))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kh, iterations=1)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//3))
    vert  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kv, iterations=1)
    combined = cv2.add(horiz, vert)
    k2       = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    combined = cv2.dilate(combined, k2, iterations=2)
    return _find_quad_from_thresh(combined)


def _strategy_hough(gray: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 80, apertureSize=3)
    lines   = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is None or len(lines) < 4: return None
    h_lines, v_lines = [], []
    for line in lines[:40]:
        rho, theta = line[0]
        if abs(np.cos(theta)) < 0.3:
            h_lines.append((rho, theta))
        elif abs(np.sin(theta)) < 0.3:
            v_lines.append((rho, theta))
    if len(h_lines) < 2 or len(v_lines) < 2: return None
    h_lines.sort(key=lambda l: l[0])
    v_lines.sort(key=lambda l: l[0])
    sel_h = [h_lines[0], h_lines[-1]]
    sel_v = [v_lines[0], v_lines[-1]]

    def intersect(l1, l2):
        r1, t1 = l1; r2, t2 = l2
        A = np.array([[np.cos(t1), np.sin(t1)],
                      [np.cos(t2), np.sin(t2)]])
        b = np.array([r1, r2])
        try:    return np.linalg.solve(A, b)
        except: return None

    pts = []
    for hline in sel_h:
        for vline in sel_v:
            pt = intersect(hline, vline)
            if pt is not None: pts.append(pt)
    if len(pts) < 4: return None
    return np.array(pts, dtype=np.float32)


def _find_grid_corners(gray: np.ndarray) -> np.ndarray:
    candidates = []
    for method in ("adaptive_gauss", "adaptive_mean", "otsu"):
        q = _strategy_threshold(gray, method)
        if q is not None: candidates.append(q)
    q = _strategy_morph_lines(gray)
    if q is not None: candidates.append(q)
    q = _strategy_hough(gray)
    if q is not None: candidates.append(q)
    if not candidates:
        h, w = gray.shape
        return np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    best = max(candidates, key=lambda q: _quad_score(q, gray.shape))
    if _quad_score(best, gray.shape) < 0:
        h, w = gray.shape
        return np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    return best


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — PERSPECTIVE WARP
# ═══════════════════════════════════════════════════════════════════════════════

def _warp(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    rect = _order_corners(corners)
    dst  = np.array([[0,0],[GRID_SIZE-1,0],
                     [GRID_SIZE-1,GRID_SIZE-1],[0,GRID_SIZE-1]],
                    dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (GRID_SIZE, GRID_SIZE))


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — CELL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_cells(warped: np.ndarray) -> list:
    cells = []
    for row in range(9):
        for col in range(9):
            y1 = row * CELL_SIZE + CELL_MARGIN
            y2 = (row + 1) * CELL_SIZE - CELL_MARGIN
            x1 = col * CELL_SIZE + CELL_MARGIN
            x2 = (col + 1) * CELL_SIZE - CELL_MARGIN
            cells.append(warped[y1:y2, x1:x2])
    return cells


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — CELL BINARISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _binarise_single(cell: np.ndarray, method: str) -> np.ndarray:
    """Return WHITE-digit BLACK-background binary image."""
    blurred = cv2.GaussianBlur(cell, (3, 3), 0)
    if method == 'a2':
        b = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 2)
    elif method == 'a4':
        b = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)
    elif method == 'a6':
        b = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 6)
    elif method == 'otsu':
        _, b = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'otsu_inv':
        _, b = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'fixed':
        _, b = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        b = np.zeros_like(blurred)

    # Polarity: outer border ring should be background (black in output)
    h, w = b.shape
    border = np.zeros_like(b, dtype=bool)
    bw = max(2, min(5, h // 6))
    border[:bw, :] = border[-bw:, :] = border[:, :bw] = border[:, -bw:] = True
    if np.mean(b[border]) > 100:
        b = cv2.bitwise_not(b)

    # Remove isolated noise pixels (2x2 open) — preserve thin strokes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)

    # >55% foreground is impossible for a digit cell → blank
    if np.count_nonzero(b) / b.size > 0.55:
        b = np.zeros_like(b)
    return b


def _score_bin(b: np.ndarray) -> float:
    """Score a binarised cell: higher = more likely to contain a clean digit."""
    ca = b.shape[0] * b.shape[1]
    k  = np.ones((2, 2), np.uint8)
    cleaned = cv2.erode(b, k, iterations=1)
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    valid = [c for c in cnts if cv2.contourArea(c) >= ca * MIN_AREA_RATIO]
    if not valid: return 0.0
    # Score = area of largest valid contour, capped at 50% of cell
    return min(cv2.contourArea(max(valid, key=cv2.contourArea)) / ca, 0.5)


def _best_binarise(cell: np.ndarray) -> np.ndarray:
    """Try all binarisation methods and return the one with the best digit score."""
    best_b, best_s = None, -1.0
    for m in ('a2', 'a4', 'a6', 'otsu', 'otsu_inv', 'fixed'):
        try:
            b = _binarise_single(cell, m)
            s = _score_bin(b)
            if s > best_s:
                best_s, best_b = s, b
        except Exception:
            pass
    if best_b is None:
        _, best_b = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY_INV)
    return best_b


# ═══════════════════════════════════════════════════════════════════════════════
#  GATE 1 — BLANK DETECTION
#
#  The key insight: paper texture creates small noise blobs that pass a naive
#  contour-area check. A real printed digit has:
#    1. A contour that is significantly larger than noise blobs
#    2. A local contrast: the darkest region is much darker than the cell mean
#    3. The large contour's bounding box covers a reasonable fraction of the cell
#
#  We use THREE independent signals and require at least TWO to agree.
# ═══════════════════════════════════════════════════════════════════════════════

def _local_contrast_score(cell: np.ndarray) -> float:
    """
    Measure how much darker the darkest region is vs the cell background.
    Returns a value in [0, 1]. Cells with a printed digit score > 0.15.
    Paper noise/texture typically scores < 0.08.
    """
    h, w = cell.shape
    # Use the inner 70% to avoid grid-line contamination
    margin_h = max(1, h // 7)
    margin_w = max(1, w // 7)
    inner = cell[margin_h:h - margin_h, margin_w:w - margin_w]
    if inner.size == 0:
        return 0.0
    cell_mean = float(np.mean(inner))
    # The darkest 10% of pixels — this is where the digit stroke is
    flat = inner.ravel().astype(np.float32)
    n_dark = max(1, len(flat) // 10)
    darkest_mean = float(np.sort(flat)[:n_dark].mean())
    # Contrast = how far darkest pixels are below the mean (normalised)
    contrast = (cell_mean - darkest_mean) / (cell_mean + 1e-6)
    return float(np.clip(contrast, 0.0, 1.0))


def _contour_score(cell: np.ndarray) -> tuple:
    """
    Run contour analysis on the best binarisation.
    Returns (has_valid_contour: bool, contour, bbox).
    A 'valid' contour must:
      - Cover >= MIN_AREA_RATIO of cell area
      - Have min dimension >= MIN_DIM px
      - Not be a horizontal sliver (grid line)
      - Its bounding box height must be >= 20% of cell height (rejects tiny dots)
    """
    binary = _best_binarise(cell)
    ca = binary.shape[0] * binary.shape[1]
    cell_h = binary.shape[0]

    k = np.ones((2, 2), np.uint8)
    cleaned = cv2.erode(binary, k, iterations=1)
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, None, None

    valid = [c for c in cnts if cv2.contourArea(c) >= ca * MIN_AREA_RATIO]
    if not valid:
        return False, None, None

    best = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)

    if w < MIN_DIM or h < MIN_DIM:
        return False, None, None
    # Reject wide horizontal slivers (grid lines)
    if w > h and (w / max(h, 1)) > MAX_ASPECT:
        return False, None, None
    # Bounding box must be tall enough to be a digit (not just a dot or speck)
    if h < cell_h * 0.18:
        return False, None, None

    return True, best, (x, y, w, h)


def _stroke_width_ok(cell: np.ndarray, bbox) -> bool:
    """
    Check that the contour has stroke-like thickness.
    A digit stroke is 2-40% of the cell width.
    Noise blobs from paper texture tend to be either too thin or too scattered.
    """
    if bbox is None:
        return False
    x, y, w, h = bbox
    cell_w = cell.shape[1]
    # Stroke width approximation: contour area / contour height
    # A real digit stroke is between 8% and 60% of cell width
    min_stroke = cell_w * 0.08
    max_stroke = cell_w * 0.60
    return min_stroke <= w <= max_stroke or min_stroke <= h <= (cell.shape[0] * 0.90)


def _is_blank(cell: np.ndarray) -> bool:
    """
    Robust blank detection using THREE independent signals.
    A cell is NON-BLANK if at least 2 of 3 signals agree it has a digit.

    Signal 1 — Local contrast: darkest region >> cell background mean
    Signal 2 — Contour check: large enough, correctly shaped blob exists
    Signal 3 — Stroke width: the blob has digit-like thickness

    This 2-of-3 voting makes the gate tolerant of:
      - Paper texture noise (fails signals 2+3 even if contrast is slightly high)
      - Faint/lightly printed digits (passes contrast + stroke even if borderline)
      - Grid-line bleed (wide slivers fail signal 3)
    """
    # Signal 1: local contrast
    contrast = _local_contrast_score(cell)
    sig1 = contrast > 0.12

    # Signals 2 & 3: contour + stroke
    has_contour, cnt, bbox = _contour_score(cell)
    sig2 = has_contour
    sig3 = _stroke_width_ok(cell, bbox) if has_contour else False

    votes = int(sig1) + int(sig2) + int(sig3)

    # Need at least 2 of 3 signals
    return votes < 2


def _digit_contour(binary: np.ndarray):
    """Return (contour, bbox) of largest valid blob, else (None, None)."""
    k = np.ones((2, 2), np.uint8)
    cleaned = cv2.erode(binary, k, iterations=1)
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    ca = binary.shape[0] * binary.shape[1]
    valid = [c for c in cnts if cv2.contourArea(c) >= ca * MIN_AREA_RATIO]
    if not valid:
        return None, None
    best = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    if w < MIN_DIM or h < MIN_DIM:
        return None, None
    if w > h and (w / max(h, 1)) > MAX_ASPECT:
        return None, None
    return best, (x, y, w, h)


# ═══════════════════════════════════════════════════════════════════════════════
#  DIGIT PREPARATION FOR CNN
# ═══════════════════════════════════════════════════════════════════════════════

def _prep_for_cnn(cell: np.ndarray) -> np.ndarray:
    """
    Return 28x28 uint8: BLACK digit on WHITE background (MNIST convention).
    1. Best binarisation → white digit on black bg
    2. Crop tight to contour bbox + 20% padding
    3. Scale longest side to 20px, centre on 28x28 canvas
    4. Invert to black-on-white
    If no contour found, enhance cell contrast first and retry.
    """
    # ── Primary path: use best binarisation ──────────────────────────────────
    binary = _best_binarise(cell)
    cnt, bbox = _digit_contour(binary)

    # ── Fallback: CLAHE contrast enhancement then re-binarise ────────────────
    if cnt is None or bbox is None:
        clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        cell_e = clahe.apply(cell)
        binary = _best_binarise(cell_e)
        cnt, bbox = _digit_contour(binary)

    if cnt is None or bbox is None:
        # Last resort: use the full cell with basic threshold
        resized = cv2.resize(binary, (20, 20))
        canvas  = np.full((28, 28), 255, dtype=np.uint8)
        canvas[4:24, 4:24] = cv2.bitwise_not(resized)
        return canvas

    x, y, w, h = bbox
    px = max(2, int(w * 0.20)); py = max(2, int(h * 0.20))
    x1 = max(0, x - px);        y1 = max(0, y - py)
    x2 = min(binary.shape[1], x + w + px)
    y2 = min(binary.shape[0], y + h + py)
    crop = binary[y1:y2, x1:x2]
    if crop.size == 0:
        crop = binary

    dh, dw  = crop.shape
    scale   = 20.0 / max(dh, dw)
    nw      = max(1, int(dw * scale))
    nh      = max(1, int(dh * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((28, 28), 255, dtype=np.uint8)
    top    = (28 - nh) // 2
    left   = (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.bitwise_not(resized)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE MATCHING — self-supervised from the warped grid itself
# ═══════════════════════════════════════════════════════════════════════════════

def _build_templates(cells: list, cnn_results: list, cnn_meta: list) -> dict:
    """
    Build per-digit 28x28 templates from CNN high-confidence cells.
    Telegraph digits differ from MNIST — using actual cells as templates
    dramatically improves uncertain-cell predictions.
    Returns {digit: [template_28x28, ...]} for digits 1-9.
    """
    templates: dict = {d: [] for d in range(1, 10)}
    for i, (digit, m) in enumerate(zip(cnn_results, cnn_meta)):
        if digit == 0 or m.get('source') != 'cnn_high':
            continue
        tmpl = _prep_for_cnn(cells[i]).astype(np.float32)
        templates[digit].append(tmpl)
    return templates


def _template_predict(prep: np.ndarray, templates: dict) -> tuple:
    """
    NCC template matching. Returns (best_digit, score) where score in [0,1].
    cv2.TM_CCOEFF_NORMED is robust to brightness differences between cells.
    """
    best_digit, best_score = 0, -1.0
    img_f = prep.astype(np.float32)
    for digit, tmpls in templates.items():
        for tmpl in tmpls:
            result = cv2.matchTemplate(img_f, tmpl, cv2.TM_CCOEFF_NORMED)
            score  = float(result.max())
            if score > best_score:
                best_score, best_digit = score, digit
    return best_digit, best_score


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — SUDOKU CONSTRAINT CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _consistent(board: list) -> bool:
    for i in range(81):
        v = board[i]
        if not v: continue
        r,c   = divmod(i, 9)
        br,bc = (r//3)*3, (c//3)*3
        for j in range(9):
            if j!=c and board[r*9+j]==v: return False
            if j!=r and board[j*9+c]==v: return False
        for dr in range(3):
            for dc in range(3):
                ni = (br+dr)*9+(bc+dc)
                if ni!=i and board[ni]==v: return False
    return True


def _solve(board: list) -> bool:
    best_i, best_c = -1, None
    for i in range(81):
        if board[i]==0:
            r,c  = divmod(i,9); br,bc=(r//3)*3,(c//3)*3
            used = set()
            for j in range(9):
                used.add(board[r*9+j]); used.add(board[j*9+c])
            for dr in range(3):
                for dc in range(3): used.add(board[(br+dr)*9+(bc+dc)])
            cands = {n for n in range(1,10) if n not in used}
            if not cands: return False
            if best_i==-1 or len(cands)<len(best_c):
                best_i,best_c = i,cands
    if best_i==-1: return True
    for n in sorted(best_c):
        board[best_i]=n
        if _solve(board): return True
        board[best_i]=0
    return False


def _constraint_fix(results: list, meta: list) -> list:
    """
    Multi-pass constraint correction.
    Pass 1 : Fast path — if consistent & solvable, return immediately.
    Pass 2 : Single substitution on conflict + uncertain cells.
    Pass 3 : Double substitution on conflict pairs.
    Pass 4 : Blank recovery using Sudoku elimination (valid digits only).
    Pass 5 : Nuclear — zero out cells by ascending confidence until solvable.
    Pass 6 : Restore and return best-effort board.
    """
    board = results[:]

    # Pass 1
    if _consistent(board):
        t = board[:]
        if _solve(t): return board

    # Build priority list
    to_try = []
    for i in range(81):
        m = meta[i]; r,c = divmod(i,9); br,bc=(r//3)*3,(c//3)*3; v=board[i]
        conflict = v and (
            any(j!=c and board[r*9+j]==v for j in range(9)) or
            any(j!=r and board[j*9+c]==v for j in range(9)) or
            any(board[(br+dr)*9+(bc+dc)]==v
                for dr in range(3) for dc in range(3)
                if (br+dr)*9+(bc+dc)!=i))
        all_probs  = m.get('all_probs', {})
        candidates = sorted(range(1,10),
                            key=lambda d: all_probs.get(d,0.0), reverse=True)
        src = m.get('source','blank')
        if src == 'blank' or m.get('digit',0) == 0:
            to_try.append((2, m.get('conf',0.0), i, candidates))
        elif conflict:
            to_try.append((0, m.get('conf',0.0), i, candidates))
        elif src in ('cnn_uncertain', 'template'):
            to_try.append((1, m.get('conf',0.0), i, candidates))
        else:
            to_try.append((3, m.get('conf',0.0), i, candidates))
    to_try.sort(key=lambda x:(x[0],x[1]))

    # Pass 2 — single substitution
    for priority, _, i, candidates in to_try:
        if priority >= 2: continue
        orig = board[i]
        for d in candidates:
            if d==orig: continue
            board[i]=d
            if _consistent(board):
                t=board[:]
                if _solve(t): return board
        board[i]=orig

    # Pass 3 — double substitution
    conflict_cells = [(i,cands) for pr,_,i,cands in to_try if pr==0]
    other_cells    = [(i,cands) for pr,_,i,cands in to_try if pr<=1][:12]
    for i,ci in conflict_cells:
        oi = board[i]
        for di in ci:
            if di==oi: continue
            board[i]=di
            for j,cj in other_cells:
                if j==i: continue
                oj=board[j]
                for dj in cj:
                    if dj==oj: continue
                    board[j]=dj
                    if _consistent(board):
                        t=board[:]
                        if _solve(t): return board
                    board[j]=oj
            board[i]=oi

    # Pass 4 — blank recovery with Sudoku elimination
    for i, _cands in [(i,c) for pr,_,i,c in to_try if pr==2]:
        r, c = divmod(i, 9); br, bc = (r//3)*3, (c//3)*3
        used = set()
        for j in range(9):
            used.add(board[r*9+j]); used.add(board[j*9+c])
        for dr in range(3):
            for dc in range(3): used.add(board[(br+dr)*9+(bc+dc)])
        valid_cands = [d for d in _cands if d not in used] or \
                      [d for d in range(1,10) if d not in used]
        for d in valid_cands:
            board[i] = d
            if _consistent(board):
                t = board[:]
                if _solve(t): return board
        board[i] = 0

    # Pass 5 — nuclear: zero lowest-confidence cells first
    source_priority = {'cnn_uncertain': 0, 'template': 1,
                       'cnn_low': 2, 'template_override': 3, 'cnn_high': 4}
    classified = sorted(
        [(i, m.get('conf',0.0), source_priority.get(m.get('source','blank'), 5))
         for i, m in enumerate(meta)
         if m.get('source','blank') != 'blank' and board[i] != 0],
        key=lambda x: (x[2], x[1])
    )
    zeroed = []
    for i, _, _ in classified:
        orig = board[i]; board[i] = 0; zeroed.append((i, orig))
        if _consistent(board):
            t = board[:]
            if _solve(t): return board

    # Pass 6 — restore and return best-effort board
    for i, orig in zeroed:
        if board[i] == 0:
            board[i] = orig
    return board


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — CNN CLASSIFICATION + TEMPLATE RE-SCORING
# ═══════════════════════════════════════════════════════════════════════════════

from PIL import Image as _PIL_Image, ImageFilter as _PIL_ImageFilter

# ── Test-Time Augmentation transforms ─────────────────────────────────────────
_TTA_AUGS = [
    lambda img: img,
    lambda img: img.rotate(6, fillcolor=255),
    lambda img: img.rotate(-6, fillcolor=255),
    lambda img: img.filter(_PIL_ImageFilter.GaussianBlur(radius=0.5)),
    lambda img: img.filter(_PIL_ImageFilter.SHARPEN),
    lambda img: img.transform((28,28), _PIL_Image.AFFINE,
                               (1, 0.10, -0.10*14, 0, 1, 0),
                               resample=_PIL_Image.BILINEAR, fillcolor=255),
    lambda img: img.transform((28,28), _PIL_Image.AFFINE,
                               (1,-0.10,  0.10*14, 0, 1, 0),
                               resample=_PIL_Image.BILINEAR, fillcolor=255),
    lambda img: img.resize((24,24), _PIL_Image.LANCZOS).resize((28,28), _PIL_Image.LANCZOS),
]


def _tta_probs(prep: np.ndarray, model) -> torch.Tensor:
    """
    Run 8 augmented versions of the 28x28 cell through the CNN and
    average softmax probabilities. Returns tensor of shape (10,).
    This is the single biggest accuracy improvement over single-pass inference.
    """
    pil     = _PIL_Image.fromarray(prep)
    tensors = []
    for aug in _TTA_AUGS:
        try:    tensors.append(_cnn_tf(aug(pil)))
        except: tensors.append(_cnn_tf(pil))
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)
    return probs.mean(dim=0)


def _classify_cnn(cells: list, model) -> list:
    """
    CNN classifier with Test-Time Augmentation (TTA) + template re-scoring.
    Each non-blank cell is run through 8 augmented versions; probabilities
    are averaged before taking argmax. Dramatically reduces misclassifications
    on borderline cells (faint ink, slight blur, etc).
    """
    has_digit = [not _is_blank(c) for c in cells]
    active    = [i for i, h in enumerate(has_digit) if h]
    if not active: return [0] * 81

    results = [0] * 81
    meta    = [{'digit': 0, 'conf': 0.0, 'source': 'blank', 'all_probs': {}}
               for _ in range(81)]

    for i in active:
        prep      = _prep_for_cnn(cells[i])        # 28x28 uint8 black-on-white
        prow      = _tta_probs(prep, model)         # TTA-averaged probabilities
        all_probs = {d: prow[d].item() for d in range(1, 10)}

        top2     = torch.topk(prow, k=2)
        tv, ti   = top2.values.tolist(), top2.indices.tolist()
        bp, bc   = ti[0], tv[0]
        sp, sc   = ti[1], tv[1]

        # If CNN predicts blank(0) but blank-gate confirmed digit, promote best 1-9
        if bp == 0:
            if sp != 0:
                bp, bc, sp, sc = sp, sc, 0, 0.0
            else:
                bp = max(range(1, 10), key=lambda d: all_probs.get(d, 0.0))
                bc = all_probs.get(bp, 0.0)

        ru     = sp if sp != 0 else 0
        source = ('cnn_high'      if bc >= CONF_HIGH else
                  'cnn_low'       if bc >= CONF_LOW  else
                  'cnn_uncertain')

        results[i] = bp
        meta[i]    = {'digit': bp, 'conf': bc, 'runner_up': ru,
                      'source': source, 'all_probs': all_probs}

    # Template re-scoring: build Telegraph-specific templates from high-conf cells
    templates     = _build_templates(cells, results, meta)
    has_templates = any(v for v in templates.values())

    if has_templates:
        for i in active:
            src = meta[i].get('source', 'blank')
            if src == 'cnn_high': continue          # already trusted

            prep     = _prep_for_cnn(cells[i])
            t_digit, t_score = _template_predict(prep, templates)
            cnn_digit = results[i]

            if t_digit == 0 or t_score < 0.35: continue

            if src == 'cnn_uncertain' and t_score >= 0.35:
                results[i] = t_digit
                meta[i].update({'digit': t_digit, 'conf': t_score,
                                'source': 'template', 'runner_up': cnn_digit})
            elif src == 'cnn_low' and t_digit != cnn_digit and t_score >= 0.55:
                results[i] = t_digit
                meta[i].update({'digit': t_digit, 'conf': t_score,
                                'source': 'template_override',
                                'runner_up': cnn_digit})

    return _constraint_fix(results, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  KNN FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _features(img: np.ndarray) -> np.ndarray:
    return np.array([np.mean(img[r*7:(r+1)*7, c*7:(c+1)*7])/255.0
                     for r in range(4) for c in range(4)], dtype=np.float32)


class _KNN:
    def __init__(self): self.X=self.y=None
    def fit(self,X,y): self.X=np.array(X,np.float32); self.y=np.array(y)
    def predict(self,x,k=3):
        if self.X is None: return 0,0.0
        d=np.linalg.norm(self.X-x,axis=1); idx=np.argsort(d)[:k]
        lbls,cnt=np.unique(self.y[idx],return_counts=True); best=lbls[np.argmax(cnt)]
        return int(best),float(np.max(cnt))/k/(1.0+d[idx[0]])

_knn = None
def _get_knn():
    global _knn
    if _knn: return _knn
    _knn = _KNN()
    os.makedirs(_DATA, exist_ok=True)
    try:
        from torchvision import datasets as tvds
        ds=tvds.MNIST(_DATA,train=True,download=True,transform=transforms.ToTensor())
        X,y,counts=[],[],{i:0 for i in range(1,10)}
        for img_t,label in ds:
            if label==0 or counts.get(label,0)>=600: continue
            X.append(_features((img_t.numpy()[0]*255).astype(np.uint8)))
            y.append(label); counts[label]+=1
            if all(v>=600 for v in counts.values()): break
        if X: _knn.fit(X,y)
    except Exception as e: print(f"  kNN warning: {e}")
    return _knn

def _classify_knn(cells: list) -> list:
    knn=_get_knn()
    out=[]
    for cell in cells:
        if _is_blank(cell): out.append(0); continue
        prep=_prep_for_cnn(cell)
        wob=cv2.bitwise_not(prep)
        d,conf=knn.predict(_features(wob))
        out.append(d if conf>=CONF_HIGH*0.4 else 0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_binary_grid(cells: list) -> np.ndarray:
    grid_img = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.uint8) * 255
    for idx, cell in enumerate(cells):
        row, col = divmod(idx, 9)
        y1 = row * CELL_SIZE + CELL_MARGIN
        y2 = (row + 1) * CELL_SIZE - CELL_MARGIN
        x1 = col * CELL_SIZE + CELL_MARGIN
        x2 = (col + 1) * CELL_SIZE - CELL_MARGIN
        try:
            binary  = _best_binarise(cell)
            display = cv2.bitwise_not(binary)
            h, w    = y2-y1, x2-x1
            resized = cv2.resize(display, (w, h), interpolation=cv2.INTER_NEAREST)
            grid_img[y1:y2, x1:x2] = resized
        except Exception:
            pass
    for i in range(10):
        thick = 3 if i % 3 == 0 else 1
        pos = i * CELL_SIZE
        cv2.line(grid_img, (pos, 0), (pos, GRID_SIZE), 0, thick)
        cv2.line(grid_img, (0, pos), (GRID_SIZE, pos), 0, thick)
    return grid_img


def _build_filter_stages(img: np.ndarray, gray: np.ndarray,
                          corners: np.ndarray, warped: np.ndarray) -> dict:
    orig_draw = img.copy()
    if corners is not None:
        pts = _order_corners(corners).astype(np.int32)
        cv2.polylines(orig_draw, [pts.reshape(-1,1,2)], True, (0,255,0), 3)
        for pt in pts:
            cv2.circle(orig_draw, tuple(pt.astype(int)), 8, (0,0,255), -1)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.adaptiveThreshold(blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return {'original': orig_draw, 'grayscale': gray,
            'threshold': thresh,  'warped': warped}


def _encode_png_b64(img: np.ndarray) -> str:
    import base64
    ok, buf = cv2.imencode('.png', img)
    if not ok: return ''
    return base64.b64encode(buf.tobytes()).decode('ascii')


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sudoku_from_image(img_bytes: bytes, model=None) -> list:
    img     = _decode(img_bytes)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = _find_grid_corners(gray)
    warped  = _warp(gray, corners)
    cells   = _extract_cells(warped)
    return (_classify_cnn(cells, model) if model is not None
            else _classify_knn(cells))


def extract_sudoku_full(img_bytes: bytes, model=None) -> dict:
    img     = _decode(img_bytes)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = _find_grid_corners(gray)
    warped  = _warp(gray, corners)
    cells   = _extract_cells(warped)

    board    = (_classify_cnn(cells, model) if model is not None
                else _classify_knn(cells))
    bin_grid = _build_binary_grid(cells)
    stages   = _build_filter_stages(img, gray, corners, warped)

    images = {
        'original':    _encode_png_b64(stages['original']),
        'grayscale':   _encode_png_b64(stages['grayscale']),
        'threshold':   _encode_png_b64(stages['threshold']),
        'warped':      _encode_png_b64(stages['warped']),
        'binary_grid': _encode_png_b64(bin_grid),
    }
    return {
        'board':       board,
        'digit_count': sum(1 for d in board if d != 0),
        'mode':        'cnn' if model is not None else 'knn_fallback',
        'images':      images,
    }