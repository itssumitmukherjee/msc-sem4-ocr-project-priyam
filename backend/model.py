"""
backend/model.py
================
DigitCNN — Telegraph newspaper sudoku digit recogniser.

Architecture
------------
  Input  : (B, 1, 28, 28)  BLACK digit on WHITE background
  Output : (B, 10)          logits for classes 0..9 (0 = blank)

Three convolutional blocks with Squeeze-and-Excitation channel attention.
SE attention learns to suppress grid-line noise features and amplify
digit-stroke features — critical for thin-stroke newspaper digits.

Classifier head is intentionally wide (512 units) to handle the subtle
visual differences between Telegraph digits (e.g. 1 vs 7, 3 vs 8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: global avg pool → fc down → fc up → sigmoid gate."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * s


class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Block 1 — (1,28,28) → (32,14,14)
        self.b1 = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.se1 = SEBlock(32)

        # Block 2 — (32,14,14) → (64,7,7)
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.se2 = SEBlock(64)

        # Block 3 — (64,7,7) → (128,3,3)
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.se3 = SEBlock(128)

        # Classifier — 128*3*3 = 1152
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512,  256), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256,   10),
        )

    def forward(self, x):
        return self.clf(self.se3(self.b3(self.se2(self.b2(self.se1(self.b1(x)))))))

    def predict_single(self, x):
        self.eval()
        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()
