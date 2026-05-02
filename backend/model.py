"""
backend/model.py
================
DigitCNN — Custom Telegraph Newspaper Sudoku Digit Recogniser.

INPUT  : (B, 1, 28, 28)  grayscale — BLACK digit on WHITE background
OUTPUT : (B, 10)          logits for classes 0-9  (0 = blank cell)

Architecture: Custom Deep CNN with Residual Blocks + SE Attention
─────────────────────────────────────────────────────────────────
  This is an original architecture designed specifically for recognising
  printed digits from Telegraph newspaper sudoku photographs.

  The design draws on the following well-known architectural ideas:
    • Residual (skip) connections  [He et al., 2016 — ResNet]
    • Squeeze-and-Excitation channel attention  [Hu et al., 2018 — SENet]
    • Global Average Pooling instead of Flatten  [Lin et al., 2014 — NIN]
    • Label smoothing + cosine LR schedule  [modern training best practices]

  All source code is original — no pretrained model or pretrained weights
  are used.  The architecture parameters, block structure, channel counts,
  and head design are custom-tuned for this specific 28×28 digit task.

Spatial flow:
  Input  (1, 28, 28)
  Stem   → (32, 28, 28)
  Stage1 → (64, 14, 14)   stride-2 conv + SEResBlock × 1
  Stage2 → (128,  7,  7)  stride-2 conv + SEResBlock × 2
  Stage3 → (256,  4,  4)  stride-2 conv + SEResBlock × 1
  GAP    → (256,)
  Head   → 256 → 512 → 256 → 10

Normalisation (must be IDENTICAL in train.py and ocr.py):
  mean = 0.8693   std = 0.3081   (black digit on white background)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILDING BLOCK — Squeeze-and-Excitation Residual Block
# ═══════════════════════════════════════════════════════════════════════════════

class SEResBlock(nn.Module):
    """
    Custom residual block combining:
      1. Two 3×3 conv layers (same channel count, no spatial downsampling)
      2. Batch normalisation after each conv
      3. Identity skip connection (input added directly to output)
      4. Squeeze-and-Excitation channel recalibration

    SE attention mechanism:
      • Global average pool → scalar per channel (squeeze)
      • Two FC layers with bottleneck (excitation)
      • Sigmoid gate multiplied back onto feature map
      This teaches the block which channels carry digit information
      and suppresses channels dominated by paper texture noise.
    """

    def __init__(self, channels: int, se_ratio: int = 8):
        super().__init__()

        # Two conv layers — same channels in and out (required for skip)
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # SE: squeeze to channels//se_ratio, excite back to channels
        squeezed = max(4, channels // se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(channels, squeezed, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(squeezed, channels, bias=False),
            nn.Sigmoid(),                      # gate in [0, 1]
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out      = self.conv_block(x)
        # SE: reshape gate to (B, C, 1, 1) for broadcast
        gate     = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out      = out * gate
        # Skip connection + activation
        return self.activation(residual + out)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN MODEL — DigitCNN
# ═══════════════════════════════════════════════════════════════════════════════

class DigitCNN(nn.Module):
    """
    Custom deep CNN for Telegraph newspaper sudoku digit recognition.

    Design choices:
    • Stem conv: 3×3, no stride — preserve spatial detail at 28×28
    • 3 downsampling stages via stride-2 conv (not MaxPool) — learnable
    • SEResBlocks after each downsampling — channel recalibration
    • Global Average Pooling — translation-robust, no spatial overfit
    • Wide head (256→512→256→10) with heavy dropout — handles noisy inputs
    • BatchNorm throughout — stable training on mixed real/synthetic data
    """

    def __init__(self):
        super().__init__()

        # ── Stem: initial feature extraction, full spatial resolution ─────────
        # 3×3 conv, 32 channels, no stride → preserves all spatial detail
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Stage 1: 28×28 → 14×14, 32→64 channels ───────────────────────────
        # stride-2 conv for learnable downsampling (vs fixed MaxPool)
        # SEResBlock recalibrates channels at this spatial scale
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEResBlock(64),
            nn.Dropout2d(p=0.10),
        )

        # ── Stage 2: 14×14 → 7×7, 64→128 channels ────────────────────────────
        # Two SEResBlocks: most discriminative scale for digit strokes
        # (~3-8 px wide strokes on a 14px canvas)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEResBlock(128),
            SEResBlock(128),
            nn.Dropout2d(p=0.15),
        )

        # ── Stage 3: 7×7 → 4×4, 128→256 channels ─────────────────────────────
        # High-level semantic features; one SEResBlock is sufficient
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEResBlock(256),
            nn.Dropout2d(p=0.20),
        )

        # ── Global Average Pooling: 4×4 → 1×1 ────────────────────────────────
        # More translation-robust than Flatten for slightly off-centre digits.
        # Output: (B, 256) after squeeze
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # ── Classifier Head ───────────────────────────────────────────────────
        # Wide intermediate layer (512) captures subtle inter-class differences
        # e.g. 1 vs 7 (horizontal cap), 6 vs 9 (open/closed loop)
        # Heavy dropout regularises against synthetic↔real domain gap
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.45),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(256, 10),       # 10 classes: 0=blank, 1-9=digits
        )

        # Weight initialisation — He init for ReLU networks
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.head(x)

    def predict_single(self, x: torch.Tensor):
        """
        Single-image inference helper.
        x: (1, 1, 28, 28) tensor, already normalised
        Returns (digit: int, confidence: float)
        """
        self.eval()
        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()