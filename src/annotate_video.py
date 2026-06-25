"""
annotate_video.py — render a watch-and-verify annotated video, driven by the
NEW trained classifier.

This produces output like the old `output_v1_m.mp4`, but the success/fail call
is no longer the weak aspect-ratio proxy. Instead:

    * YOLO detects the bottle each frame  -> draws the box + tracks motion.
    * A light state machine segments WHEN a flip happens (rest -> throw -> rest).
    * At each landing, the trained temporal CNN (runs/best.pt) judges
      SUCCESS / FAIL on the buffered clip of that attempt.
    * Overlays: live "Flips: N" count, current state, the bottle box, and a
      verdict banner with confidence + the exact time interval of the attempt.

Every judged attempt is written to an events CSV with its start/end timestamps,
so you can scrub to each interval in the video and verify it yourself.

    python src/annotate_video.py --video test_clips/clip_000062.mp4
    python src/annotate_video.py --video test_clips           # folder/glob
    python src/annotate_video.py --video output_source.mp4 --yolo yolov8x.pt

Outputs (per video) in runs/annotated/:
    <name>_annotated.mp4     the video with overlays
    <name>_events.csv        one row per judged flip attempt + time interval
"""
import argparse
import glob
import csv
import os
from collections import deque

import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import FlipClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
BOTTLE_CLASS = 39  # COCO 'bottle'
SIZE = 160
NUM_FRAMES = 16
OUT_DIR = os.path.join(ROOT, "runs", "annotated")


def mmss(t):
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def pick_bottle(boxes):
    """Highest-confidence bottle -> (center, h/w ratio, conf, xyxy) or None."""
    best = None
    for b in boxes:
        if int(b.cls.item()) != BOTTLE_CLASS:
            continue
        conf = b.conf.item()
        if best is None or conf > best[0]:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
            best = (conf, ((x1 + x2) / 2, (y1 + y2) / 2), h / w, (x1, y1, x2, y2))
    if best is None:
        return None
    conf, center, ratio, xyxy = best
    return center, ratio, conf, xyxy


class FlipSegmenter:
    """Segments rest -> throw -> rest events and reports the attempt window."""

    def __init__(self, fps):
        self.fps = fps
        self.still_thresh = 7.0
        self.settle_frames = int(0.45 * fps)
        self.throw_speed = 28.0
        self.min_blur = 2
        self.lookback = int(1.8 * fps)

        self.prev_center = None
        self.motion_hist = deque(maxlen=5)
        self.win_motion = deque(maxlen=self.lookback)
        self.win_detected = deque(maxlen=self.lookback)
        self.still_run = 0
        self.judged = True            # idle at start
        self.throw_start = None       # frame idx the current attempt began

    def _motion(self, c):
        if self.prev_center is None:
            return 0.0
        d = np.hypot(c[0] - self.prev_center[0], c[1] - self.prev_center[1])
        self.motion_hist.append(d)
        return float(np.mean(self.motion_hist))

    def update(self, frame_idx, pick):
        """Return ('REST'|'ACTIVE'|'LOST', attempt_window_or_None).

        attempt_window is (start_idx, end_idx) only on the frame a landing is judged."""
        if pick is None:
            self.win_motion.append(-1.0)
            self.win_detected.append(0)
            self.still_run = 0
            if not self.judged and self.throw_start is None:
                self.throw_start = frame_idx
            self.judged = False
            self.prev_center = None
            return "LOST", None

        center, ratio, conf, xyxy = pick
        motion = self._motion(center)
        self.win_motion.append(motion)
        self.win_detected.append(1)

        if motion < self.still_thresh:
            self.still_run += 1
        else:
            if self.judged or self.throw_start is None:
                self.throw_start = frame_idx
            self.still_run = 0
            self.judged = False

        window = None
        if self.still_run >= self.settle_frames and not self.judged:
            recent = [m for m in self.win_motion if m >= 0]
            peak = max(recent) if recent else 0.0
            blur = sum(1 for d in self.win_detected if d == 0)
            threw = peak >= self.throw_speed or blur >= self.min_blur
            start = self.throw_start if self.throw_start is not None else \
                max(0, frame_idx - self.lookback)
            if threw:
                window = (start, frame_idx)   # a real attempt -> judge with CNN
            self.judged = True
            self.throw_start = None

        self.prev_center = center
        state = "REST" if self.still_run >= self.settle_frames else "ACTIVE"
        return state, window


@torch.no_grad()
def classify_window(model, small_frames, device):
    """small_frames: list of (160,160,3) RGB uint8 -> p(success)."""
    n = len(small_frames)
    if n == 0:
        return 0.0
    idx = np.clip(np.linspace(0, n - 1, NUM_FRAMES).round().astype(int), 0, n - 1)
    arr = np.stack([small_frames[i] for i in idx]).astype(np.uint8)
    x = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", enabled=device == "cuda"):
        logit = model(x)
    return torch.softmax(logit.float(), 1)[0, 1].item()


def process(video, yolo, clf, device, threshold, out_path, events_path):
    cap = cv2.VideoCapture(video)
    # Honor phone rotation metadata so frames come out display-upright (portrait).
    # Without this, clips stored landscape-with-rotation render sideways.
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    except Exception:
        pass
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine true output dims from the first decoded (already-rotated) frame.
    ok, first = cap.read()
    if not ok:
        cap.release()
        return 0, [], fps, 0
    H, W = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
    except Exception:
        pass

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    # rolling buffer of small frames so we can re-classify a past attempt window
    buf_len = int(4.0 * fps)
    small_buf = deque(maxlen=buf_len)   # (frame_idx, small_rgb)
    seg = FlipSegmenter(fps)

    count = 0
    events = []
    banner = 0          # frames remaining to show the verdict banner
    banner_text = ""
    banner_color = (255, 255, 255)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE),
                             interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        small_buf.append((frame_idx, small))

        res = yolo.predict(frame, classes=[BOTTLE_CLASS], conf=0.20,
                           verbose=False)[0]
        pick = pick_bottle(res.boxes)
        state, window = seg.update(frame_idx, pick)

        if window is not None:
            s_idx, e_idx = window
            frames_win = [sf for (fi, sf) in small_buf if s_idx <= fi <= e_idx]
            p = classify_window(clf, frames_win, device)
            success = p >= threshold
            verdict = "SUCCESS" if success else "FAIL"
            if success:
                count += 1
            t0, t1 = s_idx / fps, e_idx / fps
            events.append({
                "attempt": len(events) + 1, "verdict": verdict,
                "p_success": round(p, 4),
                "start_s": round(t0, 2), "end_s": round(t1, 2),
                "start": mmss(t0), "end": mmss(t1),
                "flip_count_after": count,
            })
            banner = int(1.3 * fps)
            banner_text = f"{verdict}  p={p:.2f}"
            banner_color = (0, 200, 0) if success else (0, 80, 255)

        # ---- draw overlays ----
        if pick is not None:
            center, ratio, conf, xyxy = pick
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            bc = (0, 200, 0) if state == "ACTIVE" else (200, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)
            cv2.putText(frame, f"bottle {conf:.2f}", (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bc, 2)

        def hud(txt, org, scale, color, thick):
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 0), thick + 4)
            cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, thick)

        hud(f"Flips: {count}", (20, 55), 1.4, (255, 255, 255), 3)
        hud(f"state: {state}", (20, 95), 0.7, (230, 230, 230), 2)
        hud(f"t {mmss(frame_idx / fps)}", (20, H - 25), 0.7, (230, 230, 230), 2)
        if banner > 0:
            hud(banner_text, (20, 150), 1.3, banner_color, 3)
            banner -= 1

        writer.write(frame)
        frame_idx += 1
        if total > 0 and frame_idx % 300 == 0:
            print(f"    {frame_idx}/{total} frames, flips={count}")

    cap.release()
    writer.release()

    with open(events_path, "w", newline="") as f:
        cols = ["attempt", "verdict", "p_success", "start_s", "end_s",
                "start", "end", "flip_count_after"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(events)

    return count, events, fps, frame_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="file, folder, or glob")
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "best.pt"))
    ap.add_argument("--yolo", default=os.path.join(HOME, "yolov8m.pt"))
    ap.add_argument("--threshold", type=float, default=0.5)
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

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    clf = FlipClassifier().to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    clf.load_state_dict(ckpt["model"])
    clf.eval()

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT_DIR, f"{name}_annotated.mp4")
        events_path = os.path.join(OUT_DIR, f"{name}_events.csv")
        print(f"\n=== {name} ===")
        count, events, fps, nf = process(path, yolo, clf, device,
                                         args.threshold, out_path, events_path)
        print(f"  flips counted: {count}   ({len(events)} attempts judged)")
        for e in events:
            print(f"    attempt {e['attempt']}: {e['verdict']:7s} "
                  f"p={e['p_success']:.2f}  {e['start']} -> {e['end']}")
        print(f"  video : {out_path}")
        print(f"  events: {events_path}")


if __name__ == "__main__":
    main()
