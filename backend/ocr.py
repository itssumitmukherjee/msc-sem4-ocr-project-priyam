"""
backend/ocr.py
==============
Telegraph Newspaper Sudoku OCR Pipeline.

PIPELINE OVERVIEW
─────────────────
  Stage 1  Decode + resize image
  Stage 2  Grid detection — 5-strategy cascade (handles severe perspective)
  Stage 3  Perspective warp → 450×450 px normalised grid
  Stage 4  Cell extraction (81 cells, 44×44 px each after margin trim)
  Stage 5  Cell binarisation — 6 methods, best selected by contour score
  Gate 1   Blank detection — 2-of-3 signal voting
  Layer 1  DigitCNN inference with Test-Time Augmentation (8 passes)
  Layer 2  Self-supervised template re-scoring
  Layer 3  Sudoku constraint correction (guarantees consistent board)
  Fallback kNN on MNIST features (when no model weights available)

NORMALISATION — must match train.py exactly:
  mean = (0.8693,)   std = (0.3081,)
  Images: BLACK digit on WHITE background before normalisation.
"""

import os, cv2, json, base64
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as _PIL_Image, ImageFilter as _PIL_ImageFilter

_BACKEND = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_BACKEND)
_DATA    = os.path.join(_ROOT, "data")

# ── Grid geometry ──────────────────────────────────────────────────────────────
GRID_SIZE   = 450
CELL_SIZE   = GRID_SIZE // 9        # 50 px
CELL_MARGIN = 3                      # px trimmed per edge → 44×44 cell

# ── Contour thresholds ─────────────────────────────────────────────────────────
MIN_AREA_RATIO = 0.015               # digit contour ≥ 1.5% of cell area
MIN_DIM        = 3                   # bounding box min side (px)
MAX_ASPECT     = 6.0                 # max w/h — rejects horizontal slivers

# ── CNN confidence thresholds ──────────────────────────────────────────────────
CONF_HIGH = 0.60
CONF_LOW  = 0.18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CNN_MEAN = (0.8693,)
_CNN_STD  = (0.3081,)
# Transform for numpy uint8 input → tensor  (used everywhere except TTA)
_cnn_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(_CNN_MEAN, _CNN_STD),
])

# Transform for PIL Image input → tensor  (used inside TTA)
# torchvision >= 0.14 raises TypeError if ToPILImage() receives a PIL Image.
# TTA augmentations already return PIL Images, so we skip ToPILImage here.
_cnn_tf_pil = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(_CNN_MEAN, _CNN_STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(path: str):
    """
    Load DigitCNN weights safely.
    Imports the DigitCNN class from model.py — the single source of truth
    for the architecture.  Raises a clear error if weights don't match.
    """
    try:
        from model import DigitCNN
    except ImportError:
        from backend.model import DigitCNN

    m     = DigitCNN().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)

    try:
        m.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"\n{'='*60}"
            f"\ndigit_cnn.pth does not match the current DigitCNN architecture."
            f"\nThis happens when model.py was changed after the last training run."
            f"\nFix: delete {path} and retrain:"
            f"\n     python backend/train.py"
            f"\n{'='*60}"
        ) from e

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
    # Ensure at least 800px on the long side for reliable grid detection
    if max(h, w) < 800:
        s   = 800 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
    # Cap at 2000px to keep processing fast
    if max(h, w) > 2000:
        s   = 2000 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — GRID DETECTION  (5-strategy cascade)
# ═══════════════════════════════════════════════════════════════════════════════

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners in order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    d    = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _quad_score(quad: np.ndarray, img_shape: tuple) -> float:
    if quad is None:
        return -1.0
    area     = cv2.contourArea(quad.reshape(-1, 1, 2).astype(np.float32))
    img_area = img_shape[0] * img_shape[1]
    if area < img_area * 0.10:
        return -1.0
    rect   = _order_corners(quad)
    w_     = (np.linalg.norm(rect[1]-rect[0]) + np.linalg.norm(rect[2]-rect[3])) / 2
    h_     = (np.linalg.norm(rect[3]-rect[0]) + np.linalg.norm(rect[2]-rect[1])) / 2
    if w_ < 10 or h_ < 10:
        return -1.0
    aspect = min(w_, h_) / max(w_, h_)      # 1.0 = perfect square
    return (area / img_area) * 0.4 + aspect * 0.6


def _largest_quad(thresh: np.ndarray):
    k       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, k, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    img_area = thresh.shape[0] * thresh.shape[1]
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < img_area * 0.10:
            break
        peri = cv2.arcLength(cnt, True)
        for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _strategy_thresh(gray: np.ndarray, method: str):
    blr = cv2.GaussianBlur(gray, (5, 5), 0)
    if method == "ag":
        t = cv2.adaptiveThreshold(blr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    elif method == "am":
        t = cv2.adaptiveThreshold(blr, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    else:   # otsu
        _, t = cv2.threshold(blr, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _largest_quad(t)


def _strategy_morph(gray: np.ndarray):
    blr = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(blr, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = b.shape
    hk   = cv2.getStructuringElement(cv2.MORPH_RECT, (w//3, 1))
    vk   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//3))
    lines = cv2.add(cv2.morphologyEx(b, cv2.MORPH_OPEN, hk),
                    cv2.morphologyEx(b, cv2.MORPH_OPEN, vk))
    dk   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return _largest_quad(cv2.dilate(lines, dk, iterations=2))


def _strategy_hough(gray: np.ndarray):
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is None or len(lines) < 4:
        return None
    hl, vl = [], []
    for r, t in lines[:40, 0]:
        (hl if abs(np.cos(t)) < 0.3 else vl if abs(np.sin(t)) < 0.3
         else []).append((r, t))
    if len(hl) < 2 or len(vl) < 2:
        return None
    hl.sort(); vl.sort()
    pts = []
    for r1, t1 in [hl[0], hl[-1]]:
        for r2, t2 in [vl[0], vl[-1]]:
            A = np.array([[np.cos(t1), np.sin(t1)],
                          [np.cos(t2), np.sin(t2)]])
            try:
                pts.append(np.linalg.solve(A, [r1, r2]))
            except np.linalg.LinAlgError:
                pass
    return np.array(pts, dtype=np.float32) if len(pts) == 4 else None


def _find_grid_corners(gray: np.ndarray) -> np.ndarray:
    candidates = []
    for m in ("ag", "am", "otsu"):
        q = _strategy_thresh(gray, m)
        if q is not None: candidates.append(q)
    for fn in (_strategy_morph, _strategy_hough):
        q = fn(gray)
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
    dst  = np.array([[0, 0], [GRID_SIZE-1, 0],
                     [GRID_SIZE-1, GRID_SIZE-1], [0, GRID_SIZE-1]],
                    dtype=np.float32)
    return cv2.warpPerspective(gray,
                                cv2.getPerspectiveTransform(rect, dst),
                                (GRID_SIZE, GRID_SIZE))


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
#  Six methods tried per cell; best selected by contour-area score.
#  This is IDENTICAL to the pipeline used in train.py's _pipeline_binarise()
#  so the model always sees the exact image format it was trained on.
# ═══════════════════════════════════════════════════════════════════════════════

def _binarise(cell: np.ndarray, method: str) -> np.ndarray:
    """Return WHITE-digit-BLACK-background binary image."""
    blr = cv2.GaussianBlur(cell, (3, 3), 0)
    if method == 'a2':
        b = cv2.adaptiveThreshold(blr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 2)
    elif method == 'a4':
        b = cv2.adaptiveThreshold(blr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)
    elif method == 'a6':
        b = cv2.adaptiveThreshold(blr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 6)
    elif method == 'otsu':
        _, b = cv2.threshold(blr, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'otsu_inv':
        _, b = cv2.threshold(blr, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:   # 'fixed'
        _, b = cv2.threshold(blr, 127, 255, cv2.THRESH_BINARY_INV)

    # Polarity: border ring should be background (black in output)
    h, w = b.shape
    bw   = max(2, min(5, h // 6))
    mask = np.zeros_like(b, dtype=bool)
    mask[:bw, :] = mask[-bw:, :] = mask[:, :bw] = mask[:, -bw:] = True
    if np.mean(b[mask]) > 100:
        b = cv2.bitwise_not(b)

    # Remove isolated noise pixels
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)

    # >55% foreground is impossible for a digit cell
    if np.count_nonzero(b) / b.size > 0.55:
        b = np.zeros_like(b)
    return b


def _bin_score(b: np.ndarray) -> float:
    ca = b.shape[0] * b.shape[1]
    cl = cv2.erode(b, np.ones((2,2), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) >= ca * MIN_AREA_RATIO]
    return (min(cv2.contourArea(max(valid, key=cv2.contourArea)) / ca, 0.5)
            if valid else 0.0)


def _best_binarise(cell: np.ndarray) -> np.ndarray:
    best_b, best_s = None, -1.0
    for m in ('a2', 'a4', 'a6', 'otsu', 'otsu_inv', 'fixed'):
        try:
            b = _binarise(cell, m)
            s = _bin_score(b)
            if s > best_s:
                best_s, best_b = s, b
        except Exception:
            pass
    if best_b is None:
        _, best_b = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY_INV)
    return best_b


# ═══════════════════════════════════════════════════════════════════════════════
#  GATE 1 — BLANK DETECTION  (2-of-3 signal voting)
#
#  Three independent signals, any two must agree for non-blank:
#    S1 Local contrast  — digit darkens its cell significantly vs background
#    S2 Contour check   — large enough, correct shape blob present
#    S3 Stroke width    — blob has digit-like proportions (not paper dust)
#
#  Voting prevents both false positives (paper noise → hallucinated digit)
#  and false negatives (faint digit → missed).
# ═══════════════════════════════════════════════════════════════════════════════

def _contrast_signal(cell: np.ndarray) -> bool:
    """True if the darkest 10% of pixels are significantly below cell mean."""
    h, w = cell.shape
    mh, mw = max(1, h//7), max(1, w//7)
    inner  = cell[mh:h-mh, mw:w-mw].ravel().astype(np.float32)
    if inner.size == 0:
        return False
    n_dark = max(1, len(inner) // 10)
    dark   = np.sort(inner)[:n_dark].mean()
    mean_  = inner.mean()
    # Contrast = relative drop of darkest pixels below mean
    return (mean_ - dark) / (mean_ + 1e-6) > 0.12


def _contour_signal(binary: np.ndarray) -> tuple:
    """Returns (valid: bool, contour, bbox)."""
    ca = binary.shape[0] * binary.shape[1]
    ch = binary.shape[0]
    for src in (cv2.erode(binary, np.ones((2,2),np.uint8), iterations=1), binary):
        cnts, _ = cv2.findContours(src, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) >= ca * MIN_AREA_RATIO]
        if not valid:
            continue
        best = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        if w < MIN_DIM or h < MIN_DIM:
            continue
        if w > h and (w / max(h, 1)) > MAX_ASPECT:
            continue   # horizontal sliver = grid line
        if h < ch * 0.18:
            continue   # too short to be a digit
        return True, best, (x, y, w, h)
    return False, None, None


def _stroke_signal(cell: np.ndarray, bbox) -> bool:
    """True if blob dimensions are consistent with a printed digit stroke."""
    if bbox is None:
        return False
    x, y, w, h = bbox
    cw, ch = cell.shape[1], cell.shape[0]
    return (cw * 0.08 <= w <= cw * 0.92 and
            ch * 0.18 <= h <= ch * 0.92)


def _is_blank(cell: np.ndarray) -> bool:
    """2-of-3 voting: returns True only if < 2 signals indicate a digit."""
    binary = _best_binarise(cell)
    s1     = _contrast_signal(cell)
    ok, _, bbox = _contour_signal(binary)
    s2     = ok
    s3     = _stroke_signal(cell, bbox) if ok else False
    return (int(s1) + int(s2) + int(s3)) < 2


def _digit_contour(binary: np.ndarray):
    """Return (contour, bbox) of best digit-shaped blob, else (None, None)."""
    ca = binary.shape[0] * binary.shape[1]
    for src in (cv2.erode(binary, np.ones((2,2),np.uint8), iterations=1), binary):
        cnts, _ = cv2.findContours(src, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) >= ca * MIN_AREA_RATIO]
        if not valid:
            continue
        best = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        if w < MIN_DIM or h < MIN_DIM:
            continue
        if w > h and (w / max(h, 1)) > MAX_ASPECT:
            continue
        return best, (x, y, w, h)
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  CELL PREPARATION FOR CNN
#  Mirrors _pipeline_binarise() from train.py exactly:
#    binarise → crop tight → scale longest side to 20px → centre on 28×28
# ═══════════════════════════════════════════════════════════════════════════════

def _prep_for_cnn(cell: np.ndarray) -> np.ndarray:
    """Return 28×28 uint8: BLACK digit on WHITE background."""
    binary    = _best_binarise(cell)
    cnt, bbox = _digit_contour(binary)

    # CLAHE fallback for very low-contrast cells
    if cnt is None or bbox is None:
        clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        binary = _best_binarise(clahe.apply(cell))
        cnt, bbox = _digit_contour(binary)

    if cnt is None or bbox is None:
        resized = cv2.resize(binary, (20, 20), interpolation=cv2.INTER_AREA)
        canvas  = np.full((28, 28), 255, dtype=np.uint8)
        canvas[4:24, 4:24] = cv2.bitwise_not(resized)
        return canvas

    x, y, w, h = bbox
    px = max(2, int(w * 0.20)); py = max(2, int(h * 0.20))
    x1 = max(0, x-px);          y1 = max(0, y-py)
    x2 = min(binary.shape[1], x+w+px)
    y2 = min(binary.shape[0], y+h+py)
    crop = binary[y1:y2, x1:x2]
    if crop.size == 0:
        crop = binary

    dh, dw  = crop.shape
    scale   = 20.0 / max(dh, dw)
    nw_, nh_ = max(1, int(dw*scale)), max(1, int(dh*scale))
    resized  = cv2.resize(crop, (nw_, nh_), interpolation=cv2.INTER_AREA)

    canvas = np.full((28, 28), 255, dtype=np.uint8)
    top    = (28 - nh_) // 2; left = (28 - nw_) // 2
    canvas[top:top+nh_, left:left+nw_] = cv2.bitwise_not(resized)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST-TIME AUGMENTATION (TTA)
#  8 augmented passes per cell; probabilities averaged before argmax.
#  This is the single largest inference-time accuracy improvement:
#  a cell that gets 55% on one pass might average 85% over 8 rotated/sheared
#  views, revealing the correct digit more reliably.
# ═══════════════════════════════════════════════════════════════════════════════

_TTA_AUGS = [
    lambda img: img,
    lambda img: img.rotate(6,  fillcolor=255),
    lambda img: img.rotate(-6, fillcolor=255),
    lambda img: img.filter(_PIL_ImageFilter.GaussianBlur(radius=0.5)),
    lambda img: img.filter(_PIL_ImageFilter.SHARPEN),
    lambda img: img.transform((28,28), _PIL_Image.AFFINE,
                               (1, 0.10,-0.10*14, 0, 1, 0),
                               resample=_PIL_Image.BILINEAR, fillcolor=255),
    lambda img: img.transform((28,28), _PIL_Image.AFFINE,
                               (1,-0.10, 0.10*14, 0, 1, 0),
                               resample=_PIL_Image.BILINEAR, fillcolor=255),
    lambda img: img.resize((24,24), _PIL_Image.LANCZOS).resize((28,28),
                            _PIL_Image.LANCZOS),
]


def _tta_probs(prep: np.ndarray, model):
    """Run 8 augmented versions through CNN; return averaged softmax (shape 10).
    Uses _cnn_tf_pil because TTA augmentations return PIL Images.
    torchvision >= 0.14 rejects PIL input to ToPILImage() with TypeError.
    """
    pil     = _PIL_Image.fromarray(prep)   # numpy uint8 -> PIL once
    tensors = []
    for aug in _TTA_AUGS:
        try:
            tensors.append(_cnn_tf_pil(aug(pil)))   # PIL aug -> tensor
        except Exception:
            tensors.append(_cnn_tf_pil(pil))         # fallback: original PIL
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)
    return probs.mean(dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-SUPERVISED TEMPLATE MATCHING
#  High-confidence CNN cells become Telegraph-specific templates.
#  Normalised cross-correlation against these catches uncertain cells that
#  the MNIST-tuned CNN mis-scores.
# ═══════════════════════════════════════════════════════════════════════════════

def _build_templates(cells: list, results: list, meta: list) -> dict:
    templates = {d: [] for d in range(1, 10)}
    for i, (digit, m) in enumerate(zip(results, meta)):
        if digit == 0 or m.get('source') != 'cnn_high':
            continue
        templates[digit].append(_prep_for_cnn(cells[i]).astype(np.float32))
    return templates


def _template_predict(prep: np.ndarray, templates: dict) -> tuple:
    best_d, best_s = 0, -1.0
    img_f = prep.astype(np.float32)
    for digit, tmpls in templates.items():
        for tmpl in tmpls:
            s = float(cv2.matchTemplate(img_f, tmpl,
                                         cv2.TM_CCOEFF_NORMED).max())
            if s > best_s:
                best_s, best_d = s, digit
    return best_d, best_s


# ═══════════════════════════════════════════════════════════════════════════════
#  SUDOKU CONSTRAINT SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def _consistent(board: list) -> bool:
    for i in range(81):
        v = board[i]
        if not v:
            continue
        r, c  = divmod(i, 9)
        br,bc = (r//3)*3, (c//3)*3
        for j in range(9):
            if j != c and board[r*9+j] == v: return False
            if j != r and board[j*9+c] == v: return False
        for dr in range(3):
            for dc in range(3):
                ni = (br+dr)*9+(bc+dc)
                if ni != i and board[ni] == v: return False
    return True


def _solve(board: list) -> bool:
    best_i, best_c = -1, None
    for i in range(81):
        if board[i] == 0:
            r, c   = divmod(i, 9); br,bc = (r//3)*3,(c//3)*3
            used   = set()
            for j in range(9):
                used.add(board[r*9+j]); used.add(board[j*9+c])
            for dr in range(3):
                for dc in range(3): used.add(board[(br+dr)*9+(bc+dc)])
            cands = {n for n in range(1,10) if n not in used}
            if not cands: return False
            if best_i == -1 or len(cands) < len(best_c):
                best_i, best_c = i, cands
    if best_i == -1: return True
    for n in sorted(best_c):
        board[best_i] = n
        if _solve(board): return True
        board[best_i] = 0
    return False


def _constraint_fix(results: list, meta: list) -> list:
    """
    6-pass constraint correction guarantees a consistent board is returned.

    Pass 1: Fast path — board is already consistent and solvable.
    Pass 2: Single substitution of conflict + uncertain cells.
    Pass 3: Double substitution of conflict pairs.
    Pass 4: Blank recovery using Sudoku elimination (valid candidates only).
    Pass 5: Nuclear — zero out cells by ascending confidence until solvable.
    Pass 6: Restore zeroed cells; return best-effort board for frontend.
    """
    board = results[:]

    if _consistent(board):
        t = board[:]
        if _solve(t): return board

    # Build priority queue
    to_try = []
    for i in range(81):
        m   = meta[i]
        r, c = divmod(i, 9); br,bc = (r//3)*3,(c//3)*3; v = board[i]
        conflict = v and (
            any(j!=c and board[r*9+j]==v for j in range(9)) or
            any(j!=r and board[j*9+c]==v for j in range(9)) or
            any(board[(br+dr)*9+(bc+dc)]==v
                for dr in range(3) for dc in range(3)
                if (br+dr)*9+(bc+dc)!=i))
        all_probs  = m.get('all_probs', {})
        candidates = sorted(range(1,10),
                            key=lambda d: all_probs.get(d,0.0), reverse=True)
        src = m.get('source', 'blank')
        if src == 'blank' or m.get('digit', 0) == 0:
            to_try.append((2, m.get('conf',0.0), i, candidates))
        elif conflict:
            to_try.append((0, m.get('conf',0.0), i, candidates))
        elif src in ('cnn_uncertain', 'template'):
            to_try.append((1, m.get('conf',0.0), i, candidates))
        else:
            to_try.append((3, m.get('conf',0.0), i, candidates))
    to_try.sort(key=lambda x: (x[0], x[1]))

    # Pass 2: single substitution
    for priority, _, i, candidates in to_try:
        if priority >= 2: continue
        orig = board[i]
        for d in candidates:
            if d == orig: continue
            board[i] = d
            if _consistent(board):
                t = board[:]
                if _solve(t): return board
        board[i] = orig

    # Pass 3: double substitution
    conflict_cells = [(i,c) for pr,_,i,c in to_try if pr==0]
    other_cells    = [(i,c) for pr,_,i,c in to_try if pr<=1][:12]
    for i, ci in conflict_cells:
        oi = board[i]
        for di in ci:
            if di == oi: continue
            board[i] = di
            for j, cj in other_cells:
                if j == i: continue
                oj = board[j]
                for dj in cj:
                    if dj == oj: continue
                    board[j] = dj
                    if _consistent(board):
                        t = board[:]
                        if _solve(t): return board
                    board[j] = oj
            board[i] = oi

    # Pass 4: blank recovery with Sudoku elimination
    for i, _cands in [(i,c) for pr,_,i,c in to_try if pr==2]:
        r, c = divmod(i, 9); br,bc = (r//3)*3,(c//3)*3
        used = set()
        for j in range(9):
            used.add(board[r*9+j]); used.add(board[j*9+c])
        for dr in range(3):
            for dc in range(3): used.add(board[(br+dr)*9+(bc+dc)])
        valid = [d for d in _cands if d not in used] or \
                [d for d in range(1,10) if d not in used]
        for d in valid:
            board[i] = d
            if _consistent(board):
                t = board[:]
                if _solve(t): return board
        board[i] = 0

    # Pass 5: nuclear — zero by ascending confidence
    src_ord = {'cnn_uncertain':0,'template':1,'cnn_low':2,
               'template_override':3,'cnn_high':4}
    classified = sorted(
        [(i, m.get('conf',0.0), src_ord.get(m.get('source','blank'),5))
         for i,m in enumerate(meta)
         if m.get('source','blank') != 'blank' and board[i] != 0],
        key=lambda x: (x[2], x[1]))
    zeroed = []
    for i, _, _ in classified:
        orig = board[i]; board[i] = 0; zeroed.append((i, orig))
        if _consistent(board):
            t = board[:]
            if _solve(t): return board

    # Pass 6: restore
    for i, orig in zeroed:
        if board[i] == 0:
            board[i] = orig
    return board


# ═══════════════════════════════════════════════════════════════════════════════
#  CNN CLASSIFIER (with TTA + template re-scoring)
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_cnn(cells: list, model) -> list:
    """
    Full 4-stage classification:
      A. TTA CNN inference on all non-blank cells
      B. Build Telegraph-specific templates from high-confidence cells
      C. Template re-score uncertain / low-confidence cells
      D. Constraint correction
    """
    has_digit = [not _is_blank(c) for c in cells]
    active    = [i for i,h in enumerate(has_digit) if h]
    if not active:
        return [0] * 81

    results = [0] * 81
    meta    = [{'digit':0,'conf':0.0,'source':'blank','all_probs':{}}
               for _ in range(81)]

    # Stage A: TTA CNN pass
    for i in active:
        prep      = _prep_for_cnn(cells[i])
        prow      = _tta_probs(prep, model)
        all_probs = {d: prow[d].item() for d in range(1,10)}
        top2      = torch.topk(prow, k=2)
        tv, ti    = top2.values.tolist(), top2.indices.tolist()
        bp, bc    = ti[0], tv[0]
        sp, sc    = ti[1], tv[1]

        # Blank-gate already confirmed digit presence; promote best 1-9
        if bp == 0:
            bp, bc = ((sp, sc) if sp != 0 else
                      (max(range(1,10), key=lambda d: all_probs.get(d,0.0)),
                       max(all_probs.values())))

        source = ('cnn_high'     if bc >= CONF_HIGH else
                  'cnn_low'      if bc >= CONF_LOW  else
                  'cnn_uncertain')

        results[i] = bp
        meta[i]    = {'digit':bp, 'conf':bc,
                      'runner_up': sp if sp != 0 else 0,
                      'source':source, 'all_probs':all_probs}

    # Stage B: Telegraph-specific templates from high-conf cells
    templates     = _build_templates(cells, results, meta)
    has_templates = any(v for v in templates.values())

    # Stage C: template re-scoring
    if has_templates:
        for i in active:
            src = meta[i].get('source','blank')
            if src == 'cnn_high':
                continue

            prep     = _prep_for_cnn(cells[i])
            t_digit, t_score = _template_predict(prep, templates)
            cnn_digit = results[i]

            if t_digit == 0 or t_score < 0.35:
                continue

            if src == 'cnn_uncertain' and t_score >= 0.35:
                results[i] = t_digit
                meta[i].update({'digit':t_digit,'conf':t_score,
                                'source':'template','runner_up':cnn_digit})
            elif src == 'cnn_low' and t_digit != cnn_digit and t_score >= 0.55:
                results[i] = t_digit
                meta[i].update({'digit':t_digit,'conf':t_score,
                                'source':'template_override',
                                'runner_up':cnn_digit})

    # Stage D: constraint correction
    return _constraint_fix(results, meta)


# ═══════════════════════════════════════════════════════════════════════════════
#  KNN FALLBACK  (when no model weights available)
# ═══════════════════════════════════════════════════════════════════════════════

def _features(img: np.ndarray) -> np.ndarray:
    return np.array([np.mean(img[r*7:(r+1)*7, c*7:(c+1)*7]) / 255.0
                     for r in range(4) for c in range(4)], dtype=np.float32)


class _KNN:
    def __init__(self): self.X = self.y = None
    def fit(self, X, y): self.X=np.array(X,np.float32); self.y=np.array(y)
    def predict(self, x, k=3):
        if self.X is None: return 0, 0.0
        d   = np.linalg.norm(self.X-x, axis=1); idx=np.argsort(d)[:k]
        lbl, cnt = np.unique(self.y[idx], return_counts=True)
        best = lbl[np.argmax(cnt)]
        return int(best), float(np.max(cnt))/k/(1.0+d[idx[0]])

_knn_cache = None
def _get_knn():
    global _knn_cache
    if _knn_cache: return _knn_cache
    _knn_cache = _KNN()
    os.makedirs(_DATA, exist_ok=True)
    try:
        from torchvision import datasets
        ds = datasets.MNIST(_DATA, train=True, download=True,
                             transform=transforms.ToTensor())
        X, y, counts = [], [], {i:0 for i in range(1,10)}
        for img_t, label in ds:
            if label==0 or counts.get(label,0)>=600: continue
            X.append(_features((img_t.numpy()[0]*255).astype(np.uint8)))
            y.append(label); counts[label]+=1
            if all(v>=600 for v in counts.values()): break
        if X: _knn_cache.fit(X, y)
    except Exception as e:
        print(f"  kNN warning: {e}")
    return _knn_cache


def _classify_knn(cells: list) -> list:
    knn = _get_knn()
    out = []
    for cell in cells:
        if _is_blank(cell): out.append(0); continue
        prep = _prep_for_cnn(cell)
        d, conf = knn.predict(_features(cv2.bitwise_not(prep)))
        out.append(d if conf >= CONF_HIGH * 0.4 else 0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE EXPORT HELPERS  (for UI pipeline visualisation)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_binary_grid(cells: list) -> np.ndarray:
    grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.uint8) * 255
    for idx, cell in enumerate(cells):
        row, col = divmod(idx, 9)
        y1 = row*CELL_SIZE+CELL_MARGIN; y2=(row+1)*CELL_SIZE-CELL_MARGIN
        x1 = col*CELL_SIZE+CELL_MARGIN; x2=(col+1)*CELL_SIZE-CELL_MARGIN
        try:
            b = _best_binarise(cell)
            d = cv2.bitwise_not(b)
            h, w = y2-y1, x2-x1
            grid[y1:y2, x1:x2] = cv2.resize(d,(w,h),
                                              interpolation=cv2.INTER_NEAREST)
        except Exception:
            pass
    for i in range(10):
        th = 3 if i%3==0 else 1; p = i*CELL_SIZE
        cv2.line(grid,(p,0),(p,GRID_SIZE),0,th)
        cv2.line(grid,(0,p),(GRID_SIZE,p),0,th)
    return grid


def _filter_stages(img, gray, corners, warped) -> dict:
    draw = img.copy()
    if corners is not None:
        pts = _order_corners(corners).astype(np.int32)
        cv2.polylines(draw,[pts.reshape(-1,1,2)],True,(0,255,0),3)
        for pt in pts: cv2.circle(draw,tuple(pt.astype(int)),8,(0,0,255),-1)
    blr = cv2.GaussianBlur(gray,(5,5),0)
    thr = cv2.adaptiveThreshold(blr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,11,2)
    return {'original':draw,'grayscale':gray,'threshold':thr,'warped':warped}


def _to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf.tobytes()).decode('ascii') if ok else ''


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(img_bytes: bytes, model=None) -> tuple:
    img     = _decode(img_bytes)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = _find_grid_corners(gray)
    warped  = _warp(gray, corners)
    cells   = _extract_cells(warped)

    if model is not None:
        board = _classify_cnn(cells, model)
        mode  = 'cnn'
    else:
        board = _classify_knn(cells)
        mode  = 'knn_fallback'

    return board, cells, img, gray, corners, warped, mode


def extract_sudoku_from_image(img_bytes: bytes, model=None) -> list:
    """Return list of 81 ints (0=blank, 1-9=digit)."""
    board, *_ = _run_pipeline(img_bytes, model)
    return board


def extract_sudoku_full(img_bytes: bytes, model=None) -> dict:
    """Return board + base64 pipeline images for the UI."""
    board, cells, img, gray, corners, warped, mode = _run_pipeline(
        img_bytes, model)

    stages = _filter_stages(img, gray, corners, warped)
    images = {
        'original':    _to_b64(stages['original']),
        'grayscale':   _to_b64(stages['grayscale']),
        'threshold':   _to_b64(stages['threshold']),
        'warped':      _to_b64(stages['warped']),
        'binary_grid': _to_b64(_build_binary_grid(cells)),
    }
    return {
        'board':       board,
        'digit_count': sum(1 for d in board if d),
        'mode':        mode,
        'images':      images,
    }