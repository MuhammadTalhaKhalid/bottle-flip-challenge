"""
cache_frames.py — decode every clip to a fixed-size frame tensor and cache it.

Why this design:
  * The dataset is heterogeneous: h264 + vp9, portrait + landscape-with-rotation,
    and 20/30/60 fps. OpenCV's ffmpeg backend auto-applies rotation metadata, so we
    get display-correct frames. We resize on read, so codec/resolution differences
    vanish.
  * CAP_PROP_FRAME_COUNT is unreliable for vp9 (returns garbage), so we DECODE ALL
    frames and then uniformly sample NUM_FRAMES indices across the real sequence.
    Correctness over cleverness — seeking by index is not trustworthy here.

Output: data/cache/<clip>.npy, each shape (NUM_FRAMES, SIZE, SIZE, 3) uint8 (RGB).
Total cache ~1.2 GB for 982 clips at 16x160x160.

Run:
    python src/cache_frames.py            # all clips, multiprocessed
    python src/cache_frames.py --workers 6
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS = os.path.join(ROOT, "data", "videos")
SPLITS = os.path.join(ROOT, "data", "splits.csv")
CACHE = os.path.join(ROOT, "data", "cache")

NUM_FRAMES = 16
SIZE = 160


def decode_clip(path, num_frames=NUM_FRAMES, size=SIZE):
    """Decode all frames (resized), then uniformly sample `num_frames` of them."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    n = len(frames)
    if n >= num_frames:
        idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    else:
        # pad short clips by repeating the last index
        idx = np.linspace(0, n - 1, num_frames).round().astype(int)
        idx = np.clip(idx, 0, n - 1)
    return np.stack([frames[i] for i in idx]).astype(np.uint8)


def _work(fn):
    src = os.path.join(VIDEOS, fn)
    dst = os.path.join(CACHE, fn.replace(".mp4", ".npy"))
    if os.path.exists(dst):
        return fn, "skip"
    try:
        arr = decode_clip(src)
        if arr is None:
            return fn, "empty"
        np.save(dst, arr)
        return fn, "ok"
    except Exception as e:  # noqa: BLE001
        return fn, f"err:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    with open(SPLITS, newline="") as f:
        files = [r["filename"] for r in csv.DictReader(f)]

    print(f"Caching {len(files)} clips with {args.workers} workers -> {CACHE}")
    done = bad = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_work, fn): fn for fn in files}
        for i, fut in enumerate(as_completed(futs), 1):
            fn, status = fut.result()
            if status not in ("ok", "skip"):
                bad += 1
                print(f"  [{status}] {fn}")
            else:
                done += 1
            if i % 100 == 0:
                print(f"  {i}/{len(files)} processed ({bad} problems)")
    print(f"Done. {done} cached/skipped, {bad} problems.")


if __name__ == "__main__":
    main()
