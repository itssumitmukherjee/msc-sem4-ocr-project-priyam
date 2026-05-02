"""
backend/train.py
================
Training script for DigitCNN — Telegraph Newspaper Sudoku Edition.

HOW TO RUN
──────────
    cd <project_root>
    python backend/train.py

Produces:
    backend/model_weights/digit_cnn.pth        ← best validation checkpoint
    backend/model_weights/digit_cnn_last.pth   ← last epoch checkpoint

DESIGN PHILOSOPHY
─────────────────
The single biggest reason previous models failed on Telegraph images is the
DOMAIN GAP: the model was trained on clean MNIST digits but at inference it
sees images that have been through the OCR pipeline (binarisation → crop →
resize → normalise).  The fix applied here is called PIPELINE-AWARE TRAINING:

    Render digit → Degrade → OCR binarise → CNN

By running the EXACT same OpenCV binarisation pipeline used in ocr.py during
training, the model sees the same image format it will see at inference time.
This alone accounts for the largest accuracy improvement.

DATASET COMPOSITION (total ~1.1 million training samples)
───────────────────────────────────────────────────────────
  TelegraphOCR × 10   Rendered + degraded + exact-OCR-pipeline binarisation
                       PRIMARY dataset.  Model trains on what it will see.
  TelegraphFont × 5   Rendered + degraded, no OCR binarise step
                       Bridges raw appearance to binarised appearance.
  MNIST inverted × 1  Standard MNIST, colour-inverted to black-on-white
  MNIST hard     × 4  Blurred + thresholded — stress-tests 1↔7, 3↔8, 6↔9
  MNIST persp    × 2  RandomPerspective + RandomAffine
  MNIST noise    × 2  Gaussian noise + brightness jitter

TRAINING DETAILS
────────────────
  Optimiser  : AdamW  (lr=1e-3, weight_decay=3e-4)
  Schedule   : 5-epoch linear warm-up → cosine annealing to epoch 100
  Loss       : CrossEntropyLoss with label_smoothing=0.10
  AMP        : Mixed-precision on CUDA (fp16 forward, fp32 gradients)
  Grad clip  : max_norm=2.0 — stabilises noisy batches
  Batch size : 256 GPU / 128 CPU

NORMALISATION (must match ocr.py identically)
──────────────────────────────────────────────
  mean = (0.8693,)   std = (0.3081,)
  Convention: BLACK digit on WHITE background before normalisation.
"""

import os, sys, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image, ImageFilter, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKEND = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_BACKEND)
_DATA    = os.path.join(_ROOT, "data")
_WDIR    = os.path.join(_BACKEND, "model_weights")
_BEST    = os.path.join(_WDIR, "digit_cnn.pth")
_LAST    = os.path.join(_WDIR, "digit_cnn_last.pth")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ── Training hyperparameters ───────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256 if torch.cuda.is_available() else 128
EPOCHS     = 100
LR         = 1e-3
WARMUP_EP  = 5          # linear warm-up epochs
SPD        = 8000       # samples per digit per copy in TelegraphOCR

# ── Normalisation — MUST be identical in ocr.py ───────────────────────────────
_MEAN      = (0.8693,)
_STD       = (0.3081,)
_normalise = transforms.Normalize(_MEAN, _STD)
_to_tensor = transforms.ToTensor()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION
#  Imported from model.py so train.py and ocr.py always use the identical class.
# ═══════════════════════════════════════════════════════════════════════════════

from model import DigitCNN          # noqa: E402  (after sys.path insert)


# ═══════════════════════════════════════════════════════════════════════════════
#  FONT DISCOVERY  (Windows + Linux + macOS)
# ═══════════════════════════════════════════════════════════════════════════════

_WIN_SYS  = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
_WIN_USR  = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                          r"Microsoft\Windows\Fonts")
_LIN_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
]


def _find_font(*names: str):
    """Return the path of the first font filename found, or None."""
    for name in names:
        for folder in [_WIN_SYS, _WIN_USR] + _LIN_DIRS:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                return p
        for base in _LIN_DIRS:
            for root, _, files in os.walk(base):
                if name in files:
                    return os.path.join(root, name)
    return None


_FONTS = [p for p in [
    _find_font("LiberationSans-Regular.ttf"),
    _find_font("LiberationSans-Bold.ttf"),
    _find_font("FreeSans.ttf"),
    _find_font("DejaVuSans.ttf"),
    _find_font("DejaVuSans-Bold.ttf"),
    _find_font("arial.ttf", "Arial.ttf"),
    _find_font("calibri.ttf", "Calibri.ttf"),
    _find_font("times.ttf", "Times New Roman.ttf", "timesnewroman.ttf"),
    _find_font("georgia.ttf", "Georgia.ttf"),
    _find_font("trebuc.ttf"),
    _find_font("NotoSans-Regular.ttf"),
    _find_font("Ubuntu-R.ttf"),
] if p is not None]

if not _FONTS:
    print("  WARNING: No TTF fonts found — using PIL default bitmap font.")
    print("  Install Liberation Sans for best accuracy.")
    print("  Ubuntu/Debian: sudo apt install fonts-liberation")


# ═══════════════════════════════════════════════════════════════════════════════
#  OCR PIPELINE BINARISATION
#  This is an EXACT copy of the binarisation logic from ocr.py.
#  Keeping them in sync is critical: the model must train on the same image
#  format it will see at inference time.
# ═══════════════════════════════════════════════════════════════════════════════

def _pipeline_binarise(pil_img: Image.Image) -> Image.Image:
    """
    Apply the same adaptive-threshold binarisation that ocr.py uses.
    Returns a 28×28 PIL image: BLACK digit on WHITE background.

    Steps mirror ocr.py's _best_binarise() + _prep_for_cnn():
      1. Try 4 thresholding methods; pick best by largest valid contour.
      2. Polarity check: border ring must be background.
      3. Remove noise (morphological open).
      4. Crop tight to digit bounding box + 20% padding.
      5. Scale to 20×20, centre on 28×28 canvas.
      6. Invert to black-digit-on-white.
    """
    cell    = np.array(pil_img.convert("L"), dtype=np.uint8)
    h, w    = cell.shape
    blurred = cv2.GaussianBlur(cell, (3, 3), 0)

    best_b, best_s = None, -1.0

    for method in ('a2', 'a4', 'a6', 'otsu'):
        if method == 'a2':
            b = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 2)
        elif method == 'a4':
            b = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, 4)
        elif method == 'a6':
            b = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 6)
        else:
            _, b = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Polarity check — border ring should be background (black)
        bw = max(2, min(5, h // 6))
        mask = np.zeros_like(b, dtype=bool)
        mask[:bw, :] = mask[-bw:, :] = mask[:, :bw] = mask[:, -bw:] = True
        if np.mean(b[mask]) > 100:
            b = cv2.bitwise_not(b)

        # Remove isolated noise pixels
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)

        # Guard: >55% foreground is not a digit
        if np.count_nonzero(b) / b.size > 0.55:
            b = np.zeros_like(b)

        # Score by largest valid contour
        k2 = np.ones((2, 2), np.uint8)
        cl = cv2.erode(b, k2, iterations=1)
        cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) >= h * w * 0.015]
        s = (min(cv2.contourArea(max(valid, key=cv2.contourArea)) / (h*w), 0.5)
             if valid else 0.0)
        if s > best_s:
            best_s, best_b = s, b

    if best_b is None:
        _, best_b = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY_INV)

    # Invert to black-digit-on-white
    result = cv2.bitwise_not(best_b)

    # Crop tight to digit bounding box
    k3 = np.ones((2, 2), np.uint8)
    cl2 = cv2.erode(best_b, k3, iterations=1)
    cnts2, _ = cv2.findContours(cl2, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    valid2 = [c for c in cnts2
              if cv2.contourArea(c) >= h * w * 0.015]

    if valid2:
        x, y, bw_, bh_ = cv2.boundingRect(max(valid2, key=cv2.contourArea))
        px = max(2, int(bw_ * 0.20)); py = max(2, int(bh_ * 0.20))
        x1 = max(0, x - px);  y1 = max(0, y - py)
        x2 = min(w, x + bw_ + px); y2 = min(h, y + bh_ + py)
        crop = result[y1:y2, x1:x2]
        if crop.size > 0:
            dh, dw   = crop.shape
            scale    = 20.0 / max(dh, dw)
            nw_, nh_ = max(1, int(dw * scale)), max(1, int(dh * scale))
            resized  = cv2.resize(crop, (nw_, nh_), interpolation=cv2.INTER_AREA)
            canvas   = np.full((28, 28), 255, dtype=np.uint8)
            top      = (28 - nh_) // 2; left = (28 - nw_) // 2
            canvas[top:top + nh_, left:left + nw_] = resized
            return Image.fromarray(canvas)

    return Image.fromarray(result)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — TelegraphOCR (PRIMARY)
#  Renders digits → degrades → applies exact OCR binarisation pipeline.
#  The model trains on exactly what it will see at inference.
# ═══════════════════════════════════════════════════════════════════════════════

class TelegraphOCRDataset(Dataset):
    """
    PRIMARY training dataset.

    Pipeline per sample:
      1. Render digit at 64×64 with random font + size
      2. Downscale to 28×28 (anti-aliased)
      3. Apply realistic photographic degradation
      4. Run through _pipeline_binarise() — same as ocr.py at inference
      5. Normalise

    This is the key innovation: the model trains on binarised images, so
    there is zero domain gap between training and inference.
    """

    def __init__(self, samples_per_digit: int = SPD):
        self.spd   = samples_per_digit
        self.total = 9 * samples_per_digit

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        digit = idx // self.spd + 1    # label: 1..9

        img = self._render(str(digit))
        img = self._degrade(img)
        img = _pipeline_binarise(img)  # exact OCR pipeline
        return _normalise(_to_tensor(img)), digit

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _render(self, text: str) -> Image.Image:
        sz   = 64
        img  = Image.new("L", (sz, sz), 255)
        draw = ImageDraw.Draw(img)

        if _FONTS:
            try:
                font = ImageFont.truetype(
                    random.choice(_FONTS),
                    random.randint(32, 54))
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (sz - tw) // 2 - bbox[0] + random.randint(-4, 4)
        y = (sz - th) // 2 - bbox[1] + random.randint(-4, 4)
        draw.text((x, y), text, font=font, fill=0)

        return img.resize((28, 28), Image.LANCZOS)

    def _degrade(self, img: Image.Image) -> Image.Image:
        # Rotation ±12°
        if random.random() > 0.25:
            img = img.rotate(random.uniform(-12, 12),
                              fillcolor=255, resample=Image.BILINEAR)

        # Scale jitter: 65-97% of cell size
        if random.random() > 0.20:
            scale = random.uniform(0.65, 0.97)
            ns    = max(10, int(28 * scale))
            img   = img.resize((ns, ns), Image.LANCZOS)
            c     = Image.new("L", (28, 28), 255)
            ox    = (28 - ns) // 2 + random.randint(-3, 3)
            oy    = (28 - ns) // 2 + random.randint(-3, 3)
            c.paste(img, (max(0, ox), max(0, oy)))
            img   = c

        arr = np.array(img, dtype=np.float32)

        # Newsprint grain — Gaussian noise simulates paper texture
        sigma = random.uniform(2, 18)
        arr  += np.random.normal(0, sigma, arr.shape)
        arr   = np.clip(arr, 0, 255)

        # Uneven illumination — models hand shadow or curved newspaper page
        if random.random() > 0.35:
            h, w = arr.shape
            gx   = np.linspace(random.uniform(0.78, 1.0),
                                random.uniform(0.78, 1.0), w)
            gy   = np.linspace(random.uniform(0.78, 1.0),
                                random.uniform(0.78, 1.0), h)
            arr  = np.clip(arr * np.outer(gy, gx), 0, 255)

        # Ink density: light print (faint) to heavy print (bold)
        if random.random() > 0.30:
            ink  = random.uniform(0.50, 1.55)
            arr  = 255 - np.clip((255 - arr) * ink, 0, 255)

        img = Image.fromarray(arr.astype(np.uint8))

        # Camera blur / focus variation
        if random.random() > 0.30:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 1.4)))

        # Perspective shear — camera angle
        if random.random() > 0.35:
            shx = random.uniform(-0.18, 0.18)
            shy = random.uniform(-0.12, 0.12)
            img = img.transform(
                (28, 28), Image.AFFINE,
                (1, shx, -shx * 14, shy, 1, -shy * 14),
                resample=Image.BILINEAR, fillcolor=255)

        # Polarity sanity check — background should be bright
        arr2 = np.array(img, dtype=np.float32)
        bm   = (arr2[:3, :].mean() + arr2[-3:, :].mean() +
                arr2[:, :3].mean() + arr2[:, -3:].mean()) / 4
        if bm < 100:
            arr2 = 255 - arr2

        return Image.fromarray(arr2.astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — TelegraphFont (SECONDARY)
#  Same render + degrade pipeline but WITHOUT the OCR binarise step.
#  Provides intermediate-domain samples between raw renders and binarised cells.
# ═══════════════════════════════════════════════════════════════════════════════

class TelegraphFontDataset(Dataset):

    def __init__(self, samples_per_digit: int = SPD):
        self.spd   = samples_per_digit
        self.total = 9 * samples_per_digit
        # Reuse TelegraphOCRDataset's render + degrade methods
        self._ocr = TelegraphOCRDataset(samples_per_digit)

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        digit = idx // self.spd + 1
        img   = self._ocr._render(str(digit))
        img   = self._ocr._degrade(img)
        # No OCR binarise — raw degraded render
        return _normalise(_to_tensor(img)), digit


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — MNIST-based augmentation sets
#  MNIST labels include 0 (blank), which we skip at collation.
# ═══════════════════════════════════════════════════════════════════════════════

class _InvertedMNIST(Dataset):
    """MNIST colour-inverted to black-digit-on-white + optional transform."""

    def __init__(self, mnist_ds, transform=None):
        self.ds  = mnist_ds
        self.tfm = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_t, label = self.ds[idx]
        img_t = 1.0 - img_t           # invert: white bg → black digit

        if self.tfm is not None:
            pil   = transforms.ToPILImage()(img_t)
            img_t = self.tfm(pil)
        else:
            img_t = _normalise(img_t)

        return img_t, label


class _HardPairTransform:
    """
    Stress-tests visually similar digit pairs: 1↔7, 3↔8, 6↔9.
    Applies: scale jitter → heavy blur → hard threshold → sharpen → rotation → shear.
    Forces the model to learn subtle distinguishing features.
    """
    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")

        # Scale jitter
        sc  = random.uniform(0.62, 0.96)
        ns  = max(10, int(28 * sc))
        img = img.resize((ns, ns), Image.LANCZOS)
        c   = Image.new("L", (28, 28), 255)
        c.paste(img, ((28 - ns) // 2, (28 - ns) // 2))
        img = c

        # Heavy blur then hard threshold — simulates over-compressed newsprint
        img = img.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(0.8, 2.5)))
        thr = random.randint(50, 200)
        img = img.point(lambda p: 0 if p < thr else 255)

        if random.random() > 0.3:
            img = img.filter(ImageFilter.SHARPEN)
        if random.random() > 0.4:
            img = img.rotate(random.uniform(-12, 12), fillcolor=255)
        if random.random() > 0.45:
            shx = random.uniform(-0.20, 0.20)
            img = img.transform(
                (28, 28), Image.AFFINE, (1, shx, 0, 0, 1, 0),
                resample=Image.BILINEAR, fillcolor=255)

        return _normalise(_to_tensor(img))


class _PerspectiveTransform:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        if random.random() > 0.3:
            img = transforms.RandomPerspective(
                distortion_scale=random.uniform(0.12, 0.40),
                p=1.0, fill=255)(img)
        img = transforms.RandomAffine(
            degrees=10, translate=(0.12, 0.12),
            scale=(0.78, 1.18), fill=255)(img)
        return _normalise(_to_tensor(img))


class _NoiseTransform:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        arr = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, random.uniform(6, 20), arr.shape)
        arr  = np.clip(arr, 0, 255)
        if random.random() > 0.4:
            arr = np.clip(
                arr * random.uniform(0.75, 1.20) + random.uniform(-15, 15),
                0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
        if random.random() > 0.4:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 0.9)))
        return _normalise(_to_tensor(img))


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def _collate_skip_zero(batch):
    """Drop label-0 (blank) samples — we never classify blank cells."""
    batch = [(img, lbl) for img, lbl in batch if lbl != 0]
    if not batch:
        return torch.zeros(1, 1, 28, 28), torch.zeros(1, dtype=torch.long)
    imgs, lbls = zip(*batch)
    return torch.stack(imgs), torch.tensor(lbls, dtype=torch.long)


def _build_loaders():
    os.makedirs(_DATA, exist_ok=True)

    # ── Primary: binarised Telegraph-style renders ─────────────────────────────
    # 10 independent copies × SPD samples × 9 digits
    tel_ocr  = ConcatDataset([TelegraphOCRDataset(SPD) for _ in range(10)])

    # ── Secondary: raw degraded Telegraph-style renders ────────────────────────
    tel_font = ConcatDataset([TelegraphFontDataset(SPD) for _ in range(5)])

    # ── MNIST augmentation variants ────────────────────────────────────────────
    mnist_tr = datasets.MNIST(_DATA, train=True, download=True,
                               transform=transforms.ToTensor())

    base    = _InvertedMNIST(mnist_tr)                          # ×1 ~60k
    hard1   = _InvertedMNIST(mnist_tr, _HardPairTransform())   # ×4 ~240k
    hard2   = _InvertedMNIST(mnist_tr, _HardPairTransform())
    hard3   = _InvertedMNIST(mnist_tr, _HardPairTransform())
    hard4   = _InvertedMNIST(mnist_tr, _HardPairTransform())
    persp1  = _InvertedMNIST(mnist_tr, _PerspectiveTransform())# ×2 ~120k
    persp2  = _InvertedMNIST(mnist_tr, _PerspectiveTransform())
    noise1  = _InvertedMNIST(mnist_tr, _NoiseTransform())      # ×2 ~120k
    noise2  = _InvertedMNIST(mnist_tr, _NoiseTransform())

    train_ds = ConcatDataset([
        tel_ocr,                                     # PRIMARY  ~720k
        tel_font,                                    # SECONDARY ~360k
        base,                                        # MNIST base ~60k
        hard1, hard2, hard3, hard4,                  # confusion pairs ~240k
        persp1, persp2,                              # perspective ~120k
        noise1, noise2,                              # noise ~120k
    ])                                               # TOTAL  ~1.62M

    test_ds = _InvertedMNIST(
        datasets.MNIST(_DATA, train=False, download=True,
                        transform=transforms.ToTensor()))

    cuda = torch.cuda.is_available()
    nw   = min(4, os.cpu_count() or 1) if cuda else 0
    kw   = dict(batch_size=BATCH_SIZE, num_workers=nw,
                pin_memory=cuda, collate_fn=_collate_skip_zero)
    if nw > 0:
        kw["persistent_workers"] = True

    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(test_ds,  shuffle=False, **kw))


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    print(f"\n{'='*68}")
    print(f"  DigitCNN — Telegraph Sudoku OCR  |  Custom CNN + SE Residual")
    print(f"{'='*68}")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Norm     : mean={_MEAN[0]}  std={_STD[0]}")
    print(f"  Fonts    : {[os.path.basename(f) for f in _FONTS] or 'PIL default'}")
    print(f"{'='*68}\n")

    os.makedirs(_WDIR, exist_ok=True)

    model   = DigitCNN().to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {nparams:,}\n")

    # ── Loss: label smoothing avoids overconfident predictions on ambiguous cells
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)

    # ── Optimiser: AdamW with weight decay for L2 regularisation
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)

    # ── Schedule: linear warm-up → cosine annealing
    def _lr_lambda(epoch: int) -> float:
        if epoch < WARMUP_EP:
            return (epoch + 1) / WARMUP_EP
        t = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Mixed precision (CUDA only)
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = _build_loaders()
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        # ── Training pass ──────────────────────────────────────────────────────
        model.train()
        t_loss = t_correct = t_total = 0

        for imgs, labels in train_loader:
            if imgs.shape[0] == 0:
                continue
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss    += loss.item() * imgs.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        # ── Validation pass ────────────────────────────────────────────────────
        model.eval()
        v_correct = v_total = 0
        cls_ok    = {d: 0 for d in range(1, 10)}
        cls_tot   = {d: 0 for d in range(1, 10)}

        with torch.no_grad():
            for imgs, labels in test_loader:
                if imgs.shape[0] == 0:
                    continue
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                preds  = model(imgs).argmax(1)

                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)

                for d in range(1, 10):
                    mask = (labels == d)
                    cls_ok[d]  += (preds[mask] == d).sum().item()
                    cls_tot[d] += mask.sum().item()

        val_acc   = v_correct / max(v_total, 1) * 100
        train_acc = t_correct / max(t_total, 1) * 100
        cur_lr    = scheduler.get_last_lr()[0]

        print(f"  Ep {epoch:03d}/{EPOCHS}  "
              f"loss={t_loss/max(t_total,1):.4f}  "
              f"train={train_acc:.1f}%  val={val_acc:.2f}%  "
              f"lr={cur_lr:.2e}")

        # Per-class accuracy every 10 epochs — spot weak digits immediately
        if epoch % 10 == 0 or epoch == EPOCHS:
            row = "  Per-digit: " + "  ".join(
                f"{d}={cls_ok[d]/max(cls_tot[d],1)*100:.0f}%"
                for d in range(1, 10))
            print(row)

        scheduler.step()

        # Save last checkpoint every epoch
        torch.save(model.state_dict(), _LAST)

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), _BEST)
            print(f"    ✓ New best  val={val_acc:.2f}%  → saved to {_BEST}")

    print(f"\n{'='*68}")
    print(f"  Training complete.  Best validation accuracy: {best_acc:.2f}%")
    print(f"  Weights saved: {_BEST}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    train()