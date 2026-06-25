"""
model.py — temporal CNN classifier for bottle-flip landing outcome.

Architecture (TSN-style, mobile-friendly):
    MobileNetV3-Large backbone (ImageNet-pretrained) applied per frame
      -> 960-d feature per frame
      -> temporal pooling over T frames (mean ++ max concatenated -> 1920-d)
      -> MLP head -> 2 logits (failed / successful landing)

MobileNetV3 is chosen deliberately: it is the same family that runs efficiently in
the browser (TF.js / ONNX Runtime Web) on phones, which is the deployment target.
"""
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class FlipClassifier(nn.Module):
    def __init__(self, freeze_backbone_layers=True, dropout=0.4):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        bb = mobilenet_v3_large(weights=weights)
        self.features = bb.features          # -> (B, 960, h, w)
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 960, 1, 1)
        feat_dim = 960

        if freeze_backbone_layers:
            # freeze the early blocks, fine-tune the last few + head
            n = len(self.features)
            for i, m in enumerate(self.features):
                if i < n - 4:
                    for p in m.parameters():
                        p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        f = self.features(x)
        f = self.pool(f).flatten(1)          # (B*T, 960)
        f = f.reshape(B, T, -1)              # (B, T, 960)
        mean = f.mean(dim=1)
        mx = f.max(dim=1).values
        pooled = torch.cat([mean, mx], dim=1)  # (B, 1920)
        return self.head(pooled)
