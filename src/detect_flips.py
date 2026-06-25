"""
detect_flips.py — temporal flip detection with exact time intervals.

The trained model classifies a whole clip as successful / failed landing. To find
*when* a flip happens in a (possibly longer) video, we slide a short window across
the video, classify each window, and report the time interval(s) where a flip
landing is detected — with start/end timestamps and confidence.

Output per video:
  * detected flip interval(s)  [start -> end]  with peak p(success)
  * the single most-confident landing moment (mm:ss.s)
  * a full per-window confidence timeline (saved to CSV)

    python src/detect_flips.py --video test_clips/clip_000062.mp4
    python src/detect_flips.py --video test_clips            # folder / glob
    python src/detect_flips.py --video myvideo.mp4 --win 2.0 --stride 0.25

Saves:
  runs/detections/<video>.json        per-video detected intervals + summary
  runs/detections/<video>_timeline.csv  every window: t_start,t_end,p_success
  runs/detections/all_detections.csv  one row per detected flip across all videos
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import FlipClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = 160
NUM_FRAMES = 16
OUT_DIR = os.path.join(ROOT, "runs", "detections")


def mmss(t):
    """Seconds -> mm:ss.s"""
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:04.1f}"


def decode_all(path, size=SIZE):
    """Decode every frame (resized RGB) and return (frames, fps)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not fps or fps != fps or fps <= 0:  # nan / zero guard
        fps = 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(fr)
    cap.release()
    return frames, fps


def sample_window(frames, lo, hi, num=NUM_FRAMES):
    """Uniformly sample `num` frame indices from [lo, hi) (inclusive of hi-1)."""
    hi = max(hi, lo + 1)
    idx = np.linspace(lo, hi - 1, num).round().astype(int)
    idx = np.clip(idx, 0, len(frames) - 1)
    return np.stack([frames[i] for i in idx]).astype(np.uint8)


def to_tensor(arr):
    x = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x  # (T,3,H,W)


@torch.no_grad()
def scan_video(model, path, device, win_sec, stride_sec, batch=16):
    frames, fps = decode_all(path)
    if not frames:
        return None
    n = len(frames)
    dur = n / fps
    win = max(1, int(round(win_sec * fps)))
    stride = max(1, int(round(stride_sec * fps)))

    # build window start indices; if video shorter than one window, use one window
    starts = list(range(0, max(1, n - win + 1), stride))
    if not starts:
        starts = [0]

    windows, spans = [], []
    for s0 in starts:
        s1 = min(s0 + win, n)
        windows.append(to_tensor(sample_window(frames, s0, s1)))
        spans.append((s0 / fps, s1 / fps))

    probs = []
    for i in range(0, len(windows), batch):
        x = torch.stack(windows[i:i + batch]).to(device)  # (B,T,3,H,W)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logit = model(x)
        p = torch.softmax(logit.float(), 1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())

    return {"fps": fps, "duration": dur, "spans": spans,
            "probs": probs, "n_frames": n}


def find_intervals(spans, probs, threshold, min_len_sec=0.0):
    """Merge contiguous windows with p>=threshold into [start,end,peak] intervals."""
    intervals = []
    cur = None
    for (t0, t1), p in zip(spans, probs):
        if p >= threshold:
            if cur is None:
                cur = [t0, t1, p, t0 + (t1 - t0) / 2]  # start,end,peak,peak_t
            else:
                cur[1] = t1
                if p > cur[2]:
                    cur[2] = p
                    cur[3] = t0 + (t1 - t0) / 2
        else:
            if cur is not None:
                intervals.append(cur)
                cur = None
    if cur is not None:
        intervals.append(cur)
    return [iv for iv in intervals if (iv[1] - iv[0]) >= min_len_sec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="file, folder, or glob")
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "best.pt"))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="p(success) above which a window counts as a landed flip")
    ap.add_argument("--win", type=float, default=2.0, help="window length (s)")
    ap.add_argument("--stride", type=float, default=0.25, help="window stride (s)")
    args = ap.parse_args()

    if os.path.isdir(args.video):
        paths = sorted(glob.glob(os.path.join(args.video, "*.mp4")))
    else:
        paths = sorted(glob.glob(args.video))
    if not paths:
        print(f"No videos matched: {args.video}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlipClassifier().to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        res = scan_video(model, path, device, args.win, args.stride)
        if res is None:
            print(f"{name}: could not decode")
            continue

        intervals = find_intervals(res["spans"], res["probs"], args.threshold)
        overall_peak = float(np.max(res["probs"]))
        peak_i = int(np.argmax(res["probs"]))
        peak_t0, peak_t1 = res["spans"][peak_i]
        peak_mid = peak_t0 + (peak_t1 - peak_t0) / 2

        print(f"\n=== {name}  (duration {mmss(res['duration'])}, "
              f"{res['fps']:.0f} fps) ===")
        if intervals:
            for k, (a, b, pk, pkt) in enumerate(intervals, 1):
                print(f"  Flip #{k} DETECTED  {mmss(a)} -> {mmss(b)}   "
                      f"landing ~{mmss(pkt)}   peak p(success)={pk:.3f}")
        else:
            print(f"  No successful-landing flip detected "
                  f"(max p(success)={overall_peak:.3f} at {mmss(peak_mid)})")

        # save per-window timeline CSV
        tl_path = os.path.join(OUT_DIR, f"{name}_timeline.csv")
        with open(tl_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_start_s", "t_end_s", "t_start", "t_end", "p_success"])
            for (t0, t1), p in zip(res["spans"], res["probs"]):
                w.writerow([f"{t0:.3f}", f"{t1:.3f}", mmss(t0), mmss(t1),
                            f"{p:.4f}"])

        # save per-video JSON
        j = {
            "video": os.path.basename(path),
            "duration_s": round(res["duration"], 3),
            "fps": round(res["fps"], 2),
            "window_s": args.win, "stride_s": args.stride,
            "threshold": args.threshold,
            "overall_verdict": "successful_landing" if overall_peak >= args.threshold
                               else "failed_landing",
            "peak_p_success": round(overall_peak, 4),
            "peak_landing_time": mmss(peak_mid),
            "detected_flips": [
                {"index": k, "start_s": round(a, 3), "end_s": round(b, 3),
                 "start": mmss(a), "end": mmss(b),
                 "landing_time": mmss(pkt), "peak_p_success": round(pk, 4)}
                for k, (a, b, pk, pkt) in enumerate(intervals, 1)
            ],
        }
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(j, f, indent=2)

        if intervals:
            for k, (a, b, pk, pkt) in enumerate(intervals, 1):
                all_rows.append([os.path.basename(path), k, f"{a:.3f}",
                                 f"{b:.3f}", mmss(a), mmss(b), mmss(pkt),
                                 f"{pk:.4f}"])
        else:
            all_rows.append([os.path.basename(path), 0, "", "", "", "",
                             mmss(peak_mid), f"{overall_peak:.4f}"])

    summary_path = os.path.join(OUT_DIR, "all_detections.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "flip_index", "start_s", "end_s", "start", "end",
                    "landing_time", "peak_p_success"])
        w.writerows(all_rows)

    print(f"\nSaved per-video JSON + timelines to {OUT_DIR}")
    print(f"Saved combined results to {summary_path}")


if __name__ == "__main__":
    main()
