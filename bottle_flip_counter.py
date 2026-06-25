"""
Bottle Flip Counter — v1 (pretrained detection + state machine)
================================================================

Counts SUCCESSFUL bottle flips in a video.

A "successful flip" is defined as:
    bottle resting upright on a surface
      -> goes airborne / rotates (a throw)
      -> comes to rest again AND is standing upright AND stays stable

This v1 uses the pretrained COCO 'bottle' class (no custom training needed) and
an aspect-ratio (height/width) orientation proxy:
    tall box  -> standing upright
    wide box  -> fallen / lying down
It is the validation baseline. The production version replaces the orientation
proxy with a custom YOLOv8-Pose model (cap + base keypoints) for true
upright-vs-upside-down discrimination.

Usage:
    python bottle_flip_counter.py --video "path/to/video.mp4" --out out.mp4
"""

import argparse
import csv
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

BOTTLE_CLASS = 39  # COCO class id for 'bottle'


class FlipStateMachine:
    """Counts successful flips from a per-frame stream of bottle observations."""

    def __init__(self, fps):
        self.fps = fps
        # --- tunable thresholds ---
        self.upright_ratio = 1.9        # h/w >= this => standing upright
        self.fallen_ratio = 1.4         # h/w <= this => lying / horizontal
        self.still_thresh = 7.0         # px/frame center motion => "still"
        self.settle_frames = int(0.45 * fps)   # consecutive still frames => "at rest"
        self.lookback = int(1.6 * fps)  # window to decide "was there a throw recently"
        self.throw_speed = 33.0         # px/frame peak speed in window => a real throw
        self.min_blur = 2               # lost-detection frames in window => fast flight

        # --- rolling state ---
        self.count = 0
        self.events = []                # (frame_idx, result, ratio, reason)
        self.prev_center = None
        self.motion_hist = deque(maxlen=5)         # for smoothing instantaneous motion
        self.ratio_hist = deque(maxlen=self.settle_frames)
        self.win_motion = deque(maxlen=self.lookback)   # recent smoothed motions (-1 == lost)
        self.win_detected = deque(maxlen=self.lookback) # recent detection flags (1/0)
        self.still_run = 0
        self.judged = True              # already judged the current rest? (start idle)
        self.disturbed_speed = 0.0      # peak speed seen since last rest
        self.last_trace = {}

    def _motion(self, center):
        if self.prev_center is None:
            return 0.0
        d = np.hypot(center[0] - self.prev_center[0], center[1] - self.prev_center[1])
        self.motion_hist.append(d)
        return float(np.mean(self.motion_hist))

    def _judge(self, frame_idx):
        """Bottle just came to rest. Decide if a throw happened in the lookback window."""
        recent_motions = [m for m in self.win_motion if m >= 0]
        peak_speed = max(recent_motions) if recent_motions else 0.0
        blur = sum(1 for d in self.win_detected if d == 0)
        threw = peak_speed >= self.throw_speed or blur >= self.min_blur
        stable_ratio = float(np.median(self.ratio_hist)) if self.ratio_hist else 0.0
        landed_upright = stable_ratio >= self.upright_ratio
        reason = f"peak_spd={peak_speed:.0f} blur={blur}"
        if threw:
            if landed_upright:
                self.count += 1
                self.events.append((frame_idx, "SUCCESS", round(stable_ratio, 2), reason))
            else:
                self.events.append((frame_idx, "FAIL", round(stable_ratio, 2), reason))
        else:
            self.events.append((frame_idx, "IGNORED", round(stable_ratio, 2),
                                reason + " (no throw)"))

    def update(self, frame_idx, center, ratio, detected):
        if not detected:
            # lost detection mid-flight (motion blur) — record it as a flight signature
            self.win_motion.append(-1.0)
            self.win_detected.append(0)
            self.still_run = 0
            self.judged = False          # something is happening => a new rest must be judged
            self.prev_center = None
            self.last_trace = dict(motion=-1, cy=-1, ratio=-1, count=self.count)
            return "LOST"

        motion = self._motion(center)
        self.ratio_hist.append(ratio)
        self.win_motion.append(motion)
        self.win_detected.append(1)

        if motion < self.still_thresh:
            self.still_run += 1
        else:
            self.still_run = 0
            self.judged = False          # bottle disturbed => arm a new judgement
            self.disturbed_speed = max(self.disturbed_speed, motion)

        # bottle has been still long enough => it is at rest; judge once per rest
        if self.still_run >= self.settle_frames and not self.judged:
            self._judge(frame_idx)
            self.judged = True
            self.disturbed_speed = 0.0

        self.prev_center = center
        self.last_trace = dict(motion=round(motion, 1), cy=round(center[1]),
                               ratio=round(ratio, 2), count=self.count)
        return "REST" if self.still_run >= self.settle_frames else "ACTIVE"


def pick_bottle(boxes):
    """Return (center, h/w ratio, conf, xyxy) for the highest-confidence bottle, or None."""
    best = None
    for b in boxes:
        cls = int(b.cls.item())
        if cls != BOTTLE_CLASS:
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


def run(video, out_path, weights, conf_thresh, events_csv,
        start_frame=0, end_frame=None, trace_csv=None):
    model = YOLO(weights)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    trace_rows = []
    sm = FlipStateMachine(fps)
    flash = 0  # frames remaining to flash a SUCCESS banner

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[BOTTLE_CLASS], conf=conf_thresh, verbose=False)[0]
        pick = pick_bottle(res.boxes)

        prev_count = sm.count
        if pick is not None:
            center, ratio, conf, xyxy = pick
            cur_state = sm.update(frame_idx, center, ratio, True)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            up = ratio >= sm.upright_ratio
            color = (0, 200, 0) if up else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{'UP' if up else 'flat'} r={ratio:.1f}",
                        (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cur_state = sm.update(frame_idx, None, None, False)

        if sm.count > prev_count:
            flash = int(fps)  # flash for ~1s

        # HUD
        cv2.putText(frame, f"Flips: {sm.count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 6)
        cv2.putText(frame, f"Flips: {sm.count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
        cv2.putText(frame, f"state:{cur_state}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        if flash > 0:
            cv2.putText(frame, "FLIP!", (w // 2 - 90, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 0), 5)
            flash -= 1

        if trace_csv is not None:
            t = sm.last_trace
            trace_rows.append((frame_idx, round(frame_idx / fps, 2),
                               1 if pick is not None else 0,
                               t.get("cy", -1), t.get("ratio", -1),
                               t.get("motion", -1), cur_state, sm.count))

        if writer:
            writer.write(frame)
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  {frame_idx}/{total} frames, flips so far: {sm.count}")

    cap.release()
    if writer:
        writer.release()

    if trace_csv is not None:
        with open(trace_csv, "w", newline="") as f:
            wc = csv.writer(f)
            wc.writerow(["frame", "time", "detected", "cy", "ratio", "motion", "state", "count"])
            wc.writerows(trace_rows)
        print(f"Trace written to {trace_csv}")

    print(f"\n=== DONE ===\nTotal successful flips counted: {sm.count}")
    print("Events:")
    for fi, result, ratio, reason in sm.events:
        print(f"  frame {fi:5d}  ({fi/fps:6.1f}s)  {result:8s} ratio={ratio:<5} {reason}")

    if events_csv:
        with open(events_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame", "time_sec", "result", "hw_ratio", "reason"])
            for fi, result, ratio, reason in sm.events:
                wcsv.writerow([fi, round(fi / fps, 2), result, ratio, reason])
        print(f"Events written to {events_csv}")

    return sm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="output_annotated.mp4")
    ap.add_argument("--weights", default="C:/Users/LAPTOP LAND/yolov8x.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--events", default="flip_events.csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--trace", default=None)
    args = ap.parse_args()
    run(args.video, args.out, args.weights, args.conf, args.events,
        start_frame=args.start, end_frame=args.end, trace_csv=args.trace)
