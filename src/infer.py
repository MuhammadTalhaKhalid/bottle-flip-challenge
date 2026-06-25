"""
infer.py — classify a bottle-flip clip as successful / failed landing.

This is the production interface: the app records one flip clip (~3-5s) and asks
"good flip or not?". Point it at a file, a folder, or a glob.

    python src/infer.py --clip path/to/flip.mp4
    python src/infer.py --clip "data/videos/*.mp4" --csv out.csv

Decoding matches training exactly (rotation-aware, 16 frames, 160px, ImageNet norm).
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

from cache_frames import decode_clip
from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import FlipClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["failed_landing", "successful_landing"]


def load_model(ckpt_path, device):
    model = FlipClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def preprocess(path):
    arr = decode_clip(path)  # (T,H,W,3) uint8 or None
    if arr is None:
        return None
    x = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)  # (1,T,3,H,W)


@torch.no_grad()
def classify(model, path, device, threshold=0.5):
    x = preprocess(path)
    if x is None:
        return {"clip": os.path.basename(path), "error": "could not decode"}
    x = x.to(device)
    with torch.autocast(device_type="cuda", enabled=device == "cuda"):
        logit = model(x)
    prob = torch.softmax(logit.float(), 1)[0, 1].item()
    label = "successful_landing" if prob >= threshold else "failed_landing"
    return {"clip": os.path.basename(path), "prediction": label,
            "prob_success": round(prob, 4), "good_flip": int(prob >= threshold)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="file, folder, or glob")
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "best.pt"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if os.path.isdir(args.clip):
        paths = sorted(glob.glob(os.path.join(args.clip, "*.mp4")))
    else:
        paths = sorted(glob.glob(args.clip))
    if not paths:
        print(f"No clips matched: {args.clip}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    results, good = [], 0
    for p in paths:
        r = classify(model, p, device, args.threshold)
        results.append(r)
        if r.get("good_flip"):
            good += 1
        if "error" in r:
            print(f"{r['clip']:24s}  ERROR: {r['error']}")
        else:
            mark = "GOOD" if r["good_flip"] else "BAD "
            print(f"{r['clip']:24s}  [{mark}]  {r['prediction']:18s}  "
                  f"p(success)={r['prob_success']:.3f}")

    if len(paths) > 1:
        print(f"\nSession summary: {good}/{len(paths)} successful flips "
              f"({good/len(paths):.0%})")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["clip", "prediction",
                               "prob_success", "good_flip", "error"])
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
