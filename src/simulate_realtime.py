"""
simulate_realtime.py — run a video through the real-time engine as if it were a
live camera, and render the full session UI so you can verify behavior.

This is the offline proxy for the production API: same engine, same per-frame
output; only the frame source (a file vs a live webcam) differs.

    python src/simulate_realtime.py --video session.mp4 --max-tries 5
    python src/simulate_realtime.py --video session.mp4 --gate-mode hard_calibrate

Writes runs/realtime/<name>_session.mp4 and <name>_session.csv (per-try log).
"""
import argparse
import csv
import glob
import os

import cv2

from realtime_engine import FlipSession, SessionConfig, load_models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "runs", "realtime")


def hud(frame, txt, org, scale, color, thick):
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 4)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw(frame, st, cfg, fps, vstate):
    H, W = frame.shape[:2]

    # setup target zone (center band the bottle must sit in)
    if not st["gate_ok"] or cfg.gate_mode != "soft":
        x0 = int((0.5 - cfg.center_tol) * W)
        x1 = int((0.5 + cfg.center_tol) * W)
        zc = (0, 200, 0) if st["gate_ok"] else (0, 200, 255)
        cv2.rectangle(frame, (x0, int(0.15 * H)), (x1, int(0.85 * H)), zc, 2)

    # bottle box
    if st["bottle_box"]:
        x1b, y1b, x2b, y2b = st["bottle_box"]
        bc = (0, 200, 0) if st["state"] in ("ACTIVE", "REST") else (200, 200, 200)
        cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), bc, 2)

    # top HUD
    hud(frame, f"Flips: {st['flips']}", (20, 55), 1.3, (255, 255, 255), 3)
    hud(frame, f"Tries left: {st['tries_left']}", (20, 95), 0.8,
        (230, 230, 230), 2)
    hud(frame, f"state: {st['state']}", (20, 130), 0.7, (200, 200, 200), 2)

    # big center guidance during setup
    if st["state"] == "SETUP" or (not st["gate_ok"] and st["message"]):
        msg = st["message"]
        col = (0, 200, 0) if msg == "READY" else (0, 200, 255)
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        hud(frame, msg, ((W - tw) // 2, int(0.12 * H)), 1.4, col, 3)

    # verdict banner (persist via vstate)
    if st["verdict"]:
        v = st["verdict"]
        vstate["text"] = f"{v['verdict']}  p={v['p_success']:.2f}"
        vstate["color"] = (0, 200, 0) if v["verdict"] == "SUCCESS" else (0, 80, 255)
        vstate["ttl"] = int(1.4 * fps)
    if vstate["ttl"] > 0:
        hud(frame, vstate["text"], (20, 180), 1.2, vstate["color"], 3)
        vstate["ttl"] -= 1

    # done overlay
    if st["finished"]:
        msg = f"DONE  {st['flips']}/{cfg.max_tries} flips"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        hud(frame, msg, ((W - tw) // 2, H // 2), 1.6, (0, 220, 120), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--max-tries", type=int, default=5)
    ap.add_argument("--gate-mode", default="hard",
                    choices=["hard", "soft", "hard_calibrate"])
    ap.add_argument("--conf-floor", type=float, default=0.60)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--yolo", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.video)) if not os.path.isfile(args.video) \
        else [args.video]
    if not paths:
        print(f"No video matched: {args.video}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    yolo, clf = load_models(device, args.weights, args.yolo)

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
        except Exception:
            pass
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not fps or fps != fps or fps <= 0:
            fps = 30.0
        ok, first = cap.read()
        if not ok:
            print(f"{name}: cannot read")
            continue
        H, W = first.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)
        except Exception:
            pass

        cfg = SessionConfig(max_tries=args.max_tries, gate_mode=args.gate_mode,
                            conf_floor=args.conf_floor)
        sess = FlipSession(fps, cfg, yolo, clf, device)
        out_path = os.path.join(OUT_DIR, f"{name}_session.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W, H))
        vstate = {"text": "", "color": (255, 255, 255), "ttl": 0}

        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            st = sess.process_frame(frame, fi)
            draw(frame, st, cfg, fps, vstate)
            writer.write(frame)
            fi += 1
        cap.release()
        writer.release()

        csv_path = os.path.join(OUT_DIR, f"{name}_session.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["try", "verdict", "p_success",
                                              "start_s", "end_s"])
            w.writeheader()
            w.writerows(sess.events)

        print(f"\n=== {name} ===  gate={args.gate_mode}")
        print(f"  final score: {sess.flips}/{cfg.max_tries} flips "
              f"({sess.tries_used} tries judged)")
        for e in sess.events:
            print(f"    try {e['try']}: {e['verdict']:7s} p={e['p_success']:.2f}"
                  f"  {e['start_s']:.1f}-{e['end_s']:.1f}s")
        print(f"  video: {out_path}")
        print(f"  log  : {csv_path}")


if __name__ == "__main__":
    main()
