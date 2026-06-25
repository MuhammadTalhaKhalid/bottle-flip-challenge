"""
realtime_engine.py — stateful real-time bottle-flip session engine.

This is the core that the production API wraps. You feed it frames one at a time
(as they arrive from the live camera) and it returns, every frame, everything the
front-end needs to draw: setup guidance, current state, flip count, tries left,
and the last verdict.

Design decisions (locked with the client's constraints):
  * SETUP GATE — flips are only judged once the bottle is framed the way the model
    was trained (close enough + centered). This keeps the live input in the SAME
    distribution as the training data, which is what makes the 97.96% hold up.
    Modes: "hard" (block until framed right), "soft" (warn but still count),
    "hard_calibrate" (hard + learn the user's bottle size once at session start).
  * REAL THROW REQUIRED — an attempt only counts if the bottle actually went
    airborne / moved fast then settled. Placing or nudging a standing bottle does
    NOT count (this is the false-positive class we saw on the far video).
  * FIXED N TRIES — the session ends after `max_tries` judged attempts and reports
    a final score. A confidence floor avoids counting coin-flip judgments.

Classification is whole-frame (matching how the model was trained); the gate
guarantees the bottle is large enough in-frame for that to be reliable.
"""
import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import FlipClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
BOTTLE_CLASS = 39
SIZE = 160
NUM_FRAMES = 16


@dataclass
class SessionConfig:
    max_tries: int = 5
    gate_mode: str = "hard"          # "hard" | "soft" | "hard_calibrate"
    require_throw: bool = True
    conf_floor: float = 0.60         # min p to call a SUCCESS
    # framing gate (bottle box height as a fraction of frame height)
    min_box_frac: float = 0.20
    max_box_frac: float = 0.80
    center_tol: float = 0.32         # |bottle_cx - 0.5| must be <= this
    ready_secs: float = 0.8          # steady-in-zone time to pass the gate
    # flip dynamics
    settle_secs: float = 0.55        # stillness that means "at rest" (incl. stayed up)
    throw_speed: float = 28.0        # px/frame peak speed => a real throw
    lookback_secs: float = 1.8
    yolo_conf: float = 0.20


class FlipSession:
    def __init__(self, fps, config: SessionConfig, yolo, clf, device):
        self.fps = fps
        self.cfg = config
        self.yolo = yolo
        self.clf = clf
        self.device = device

        # session score
        self.flips = 0
        self.tries_used = 0
        self.finished = False
        self.events = []                  # one dict per judged attempt
        self.last_verdict = None          # (text, p, ttl_frames)

        # gate
        self.gate_passed = (config.gate_mode == "soft")
        self.ready_run = 0
        self.calib_box_frac = None        # learned reference (calibrate mode)

        # motion / segmentation
        self.prev_center = None
        self.motion_hist = deque(maxlen=5)
        L = max(1, int(config.lookback_secs * fps))
        self.win_motion = deque(maxlen=L)
        self.win_detected = deque(maxlen=L)
        self.settle_frames = max(1, int(config.settle_secs * fps))
        self.still_run = 0
        self.judged = True
        self.throw_start = None

        # rolling small-frame buffer for classification
        self.small_buf = deque(maxlen=int(4.0 * fps))

    # ---- detection ----
    def _pick_bottle(self, boxes, frame_h):
        best = None
        n_bottles = 0
        for b in boxes:
            if int(b.cls.item()) != BOTTLE_CLASS:
                continue
            n_bottles += 1
            conf = b.conf.item()
            if best is None or conf > best[0]:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                cx = (x1 + x2) / 2
                best = (conf, (cx, (y1 + y2) / 2), h / w, (x1, y1, x2, y2),
                        h / frame_h, cx)
        if best is None:
            return None, n_bottles
        return best, n_bottles

    # ---- setup gate ----
    def _check_gate(self, pick, n_bottles, frame_w):
        """Return (in_zone, message). Updates ready_run / gate_passed."""
        if pick is None:
            self.ready_run = 0
            return False, "Show the bottle"
        if n_bottles > 1:
            self.ready_run = 0
            return False, "Only one bottle in frame"

        _, _, _, _, box_frac, cx = pick
        cx_frac = cx / frame_w
        lo, hi = self.cfg.min_box_frac, self.cfg.max_box_frac
        if self.cfg.gate_mode == "hard_calibrate" and self.calib_box_frac:
            lo, hi = self.calib_box_frac * 0.6, self.calib_box_frac * 1.6

        if box_frac < lo:
            self.ready_run = 0
            return False, "Move closer"
        if box_frac > hi:
            self.ready_run = 0
            return False, "Move back"
        if abs(cx_frac - 0.5) > self.cfg.center_tol:
            self.ready_run = 0
            return False, "Center the bottle"

        # framed correctly — require a short steady hold before READY
        self.ready_run += 1
        if self.ready_run >= int(self.cfg.ready_secs * self.fps):
            if self.cfg.gate_mode == "hard_calibrate" and self.calib_box_frac is None:
                self.calib_box_frac = box_frac
            self.gate_passed = True
            return True, "READY"
        return False, "Hold steady..."

    # ---- motion ----
    def _motion(self, center):
        if self.prev_center is None:
            return 0.0
        d = np.hypot(center[0] - self.prev_center[0],
                     center[1] - self.prev_center[1])
        self.motion_hist.append(d)
        return float(np.mean(self.motion_hist))

    @torch.no_grad()
    def _classify(self, s_idx, e_idx):
        frames = [sf for (fi, sf) in self.small_buf if s_idx <= fi <= e_idx]
        if not frames:
            return 0.0
        idx = np.clip(np.linspace(0, len(frames) - 1, NUM_FRAMES).round()
                      .astype(int), 0, len(frames) - 1)
        arr = np.stack([frames[i] for i in idx]).astype(np.uint8)
        x = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(self.device)
        with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
            logit = self.clf(x)
        return torch.softmax(logit.float(), 1)[0, 1].item()

    # ---- main entry ----
    def process_frame(self, frame, frame_idx):
        H, W = frame.shape[:2]
        small = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE),
                             interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.small_buf.append((frame_idx, small))

        res = self.yolo.predict(frame, classes=[BOTTLE_CLASS],
                                conf=self.cfg.yolo_conf, verbose=False)[0]
        pick, n_bottles = self._pick_bottle(res.boxes, H)

        out = {
            "frame": frame_idx, "flips": self.flips,
            "tries_used": self.tries_used, "tries_left":
                max(0, self.cfg.max_tries - self.tries_used),
            "finished": self.finished, "state": "IDLE",
            "gate_ok": self.gate_passed, "message": "",
            "bottle_box": None, "verdict": None,
        }
        if pick is not None:
            out["bottle_box"] = [int(v) for v in pick[3]]
        if self.finished:
            out["state"] = "DONE"
            out["message"] = f"Final score: {self.flips}/{self.cfg.max_tries}"
            return out

        # --- setup gate ---
        if not self.gate_passed or self.cfg.gate_mode != "soft":
            in_zone, msg = self._check_gate(pick, n_bottles, W)
            out["gate_ok"] = self.gate_passed
            out["message"] = msg
            if self.cfg.gate_mode in ("hard", "hard_calibrate") and not self.gate_passed:
                out["state"] = "SETUP"
                return out  # nothing counts until framed correctly

        # --- flip segmentation (gate passed) ---
        if pick is None:
            self.win_motion.append(-1.0)
            self.win_detected.append(0)
            self.still_run = 0
            if not self.judged and self.throw_start is None:
                self.throw_start = frame_idx
            self.judged = False
            self.prev_center = None
            out["state"] = "AIRBORNE"
        else:
            center = pick[1]
            motion = self._motion(center)
            self.win_motion.append(motion)
            self.win_detected.append(1)
            if motion < 7.0:
                self.still_run += 1
            else:
                if self.judged or self.throw_start is None:
                    self.throw_start = frame_idx
                self.still_run = 0
                self.judged = False
            self.prev_center = center
            out["state"] = "REST" if self.still_run >= self.settle_frames else "ACTIVE"

            # landing reached -> judge once
            if self.still_run >= self.settle_frames and not self.judged:
                recent = [m for m in self.win_motion if m >= 0]
                peak = max(recent) if recent else 0.0
                blur = sum(1 for d in self.win_detected if d == 0)
                threw = peak >= self.cfg.throw_speed or blur >= 2
                start = self.throw_start if self.throw_start is not None else \
                    max(0, frame_idx - int(self.cfg.lookback_secs * self.fps))

                if threw or not self.cfg.require_throw:
                    p = self._classify(start, frame_idx)
                    success = p >= self.cfg.conf_floor
                    if success:
                        self.flips += 1
                    self.tries_used += 1
                    verdict = "SUCCESS" if success else "FAIL"
                    t0, t1 = start / self.fps, frame_idx / self.fps
                    ev = {"try": self.tries_used, "verdict": verdict,
                          "p_success": round(p, 4),
                          "start_s": round(t0, 2), "end_s": round(t1, 2)}
                    self.events.append(ev)
                    out["verdict"] = ev
                    self.last_verdict = (verdict, p, int(1.4 * self.fps))
                    if self.tries_used >= self.cfg.max_tries:
                        self.finished = True
                self.judged = True
                self.throw_start = None

        out["flips"] = self.flips
        out["tries_used"] = self.tries_used
        out["tries_left"] = max(0, self.cfg.max_tries - self.tries_used)
        out["finished"] = self.finished
        return out


def load_models(device, weights=None, yolo_path=None):
    from ultralytics import YOLO
    weights = weights or os.path.join(ROOT, "runs", "best.pt")
    yolo_path = yolo_path or os.path.join(HOME, "yolov8m.pt")
    yolo = YOLO(yolo_path)
    clf = FlipClassifier().to(device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    clf.load_state_dict(ckpt["model"])
    clf.eval()
    return yolo, clf
