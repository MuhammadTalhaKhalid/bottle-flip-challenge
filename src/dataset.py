"""
dataset.py — loads cached frame tensors for training/eval.

Each item: a clip's cached (T, H, W, 3) uint8 array -> (T, 3, H, W) float tensor,
ImageNet-normalized, plus its binary label and metadata.

Augmentation (train only) is applied CONSISTENTLY across all T frames of a clip so
temporal motion cues are preserved (same flip / same crop / same jitter per clip).
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
SPLITS = os.path.join(ROOT, "data", "splits.csv")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_split(split):
    with open(SPLITS, newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class FlipClips(Dataset):
    def __init__(self, split, augment=False):
        self.rows = read_split(split)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _augment(self, x):
        # x: (T, H, W, 3) uint8.  Same transform for every frame in the clip.
        if np.random.rand() < 0.5:  # horizontal flip
            x = x[:, :, ::-1, :]
        # brightness / contrast jitter
        if np.random.rand() < 0.7:
            b = np.random.uniform(0.85, 1.15)
            c = np.random.uniform(0.85, 1.15)
            xf = x.astype(np.float32)
            mean = xf.mean(axis=(1, 2), keepdims=True)
            xf = (xf - mean) * c + mean * b
            x = np.clip(xf, 0, 255).astype(np.uint8)
        # small random crop+resize-back (zoom) to simulate framing variation
        if np.random.rand() < 0.5:
            T, H, W, _ = x.shape
            s = np.random.uniform(0.85, 1.0)
            ch, cw = int(H * s), int(W * s)
            top = np.random.randint(0, H - ch + 1)
            left = np.random.randint(0, W - cw + 1)
            import cv2
            x = np.stack([
                cv2.resize(f[top:top + ch, left:left + cw], (W, H))
                for f in x
            ])
        return np.ascontiguousarray(x)

    def __getitem__(self, i):
        r = self.rows[i]
        path = os.path.join(CACHE, r["filename"].replace(".mp4", ".npy"))
        x = np.load(path)  # (T,H,W,3) uint8
        if self.augment:
            x = self._augment(x)
        x = torch.from_numpy(x).float().div_(255.0)  # (T,H,W,3)
        x = x.permute(0, 3, 1, 2)  # (T,3,H,W)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        y = int(r["y"])
        return x, y, r["filename"], r["issue_tag"]
