# Bottle-Flip Landing Classifier

Classifies a recorded bottle-flip clip as a **successful landing** (bottle lands
upright and stays standing) or a **failed landing** — the core AI for the flip-
challenge app. Trained on the client's 982 labeled clips.

## Results

Trained 30 epochs; best checkpoint chosen by validation macro-F1 (epoch 7).

| Metric | Validation (147) | Held-out test (147) |
|---|---|---|
| **Accuracy** | **99.32%** | **97.96%** |
| Macro-F1 | 0.9931 | 0.9793 |
| ROC-AUC | — | 0.9979 |
| Errors | 1 / 147 | 3 / 147 |

Best at the natural threshold of **0.50** (no tuning needed). The 3 test errors are
genuinely ambiguous clips (e.g. bottle ends up held in-hand at the cap).

Confusion matrix (rows = truth):

```
          pred_fail  pred_succ
  fail        81          1
  succ         2         63
```

Every edge-case clip in the test set was classified correctly
(`body_contact`, `bottle_left_frame`, `poor_camera_angle`).

**Session-counter check** — on a 20-clip run the model counted **16/20** good
flips; ground truth was exactly **16/20**. That is the live use case (count good
flips during a timed challenge) working end to end.

Full numbers: `runs/test_report.txt`. Per-clip predictions: `runs/test_predictions.csv`.

## Approach

The earlier prototype used a pretrained COCO bottle detector + a hand-tuned
aspect-ratio state machine. It hit a ceiling: a "tall" box can't tell upright from
upside-down, and detection drops out during fast flight.

This version **learns the outcome from the labeled clips** instead of hand-tuning
thresholds:

- **Per clip:** decode 16 frames evenly across the clip (rotation-aware; handles the
  mixed h264/vp9, portrait/landscape, 20/30/60 fps recordings), resized to 160×160.
- **Model:** MobileNetV3-Large backbone (ImageNet-pretrained, last blocks fine-tuned)
  applied per frame → temporal mean+max pooling → small MLP → success/fail.
  MobileNetV3 is the same family that runs efficiently in a phone browser.
- Trained with class weights, light augmentation, mixed precision over 30 epochs with
  cosine LR decay. Best epoch chosen by validation macro-F1 (~45 s/epoch on an RTX 3070).

## Apps — play / test it live

**Primary deliverable — the in-browser PWA (`docs/`).** Real-time bottle-flip
counting that runs **entirely on the phone**: YOLOv8n bottle detection + the
MobileNetV3 landing classifier execute in the browser via **ONNX Runtime Web**
(WebGPU, with a WASM fallback). No server, no upload — the camera feed never
leaves the device. It's a installable PWA (offline after first load) and is the
exact match for the client's requirement (live, on-device, phone-first).

Deployed via GitHub Pages from `docs/` →
**https://muhammadtalhakhalid.github.io/bottle-flip-challenge/**
(open on a phone, allow the camera, and flip). `docs/engine.js` is a faithful
on-device port of `src/realtime_engine.py` (same setup gate, throw detection, and
16-frame temporal judgement), so the browser counts flips with identical logic to
the Python system.

**Optional server API (`api/` + `Dockerfile`).** A FastAPI + WebSocket service
wrapping the same engine, for hosts that prefer server-side inference (e.g. a GPU
backend). Containerised and self-contained:

```bash
docker build -t bottle-flip-api .
docker run -p 8000:8000 bottle-flip-api      # -> http://localhost:8000  (/ , /health, /ws)
```

## Layout

```
data/
  videos/            982 clips (git-ignored)
  cache/             16-frame tensors per clip (git-ignored)
  labels.csv         client labels
  splits.csv         stratified train/val/test (70/15/15)
src/
  make_splits.py     build stratified splits
  cache_frames.py    decode + cache frame tensors
  dataset.py         data loading + augmentation
  model.py           MobileNetV3 temporal classifier
  train.py           training loop
  evaluate.py        held-out test metrics + report
  infer.py           classify a clip / folder (+ session count)
  export_onnx.py     export to ONNX for the browser
  realtime_engine.py stateful live session engine (gate + flip segmentation + judge)
docs/                in-browser PWA (the deliverable) — served by GitHub Pages
  index.html, styles.css, app.js   camera + HUD + UI wiring
  engine.js          on-device port of realtime_engine.py (FlipSession)
  yolo.js            YOLOv8n bottle detection in onnxruntime-web
  models/            yolov8n.onnx + flip_classifier.onnx (fp16, ~8 MB)
  sw.js, manifest.webmanifest, icons/   PWA install + offline cache
api/
  server.py          FastAPI + WebSocket service wrapping realtime_engine
  requirements.txt   API service deps
runs/
  best.pt            trained checkpoint
  flip_classifier.onnx  full-precision ONNX export
  test_report.txt, test_predictions.csv, train_log.csv
```

## Reproduce

```bash
pip install -r requirements.txt
python src/make_splits.py
python src/cache_frames.py
python src/train.py --epochs 30 --batch 8
python src/evaluate.py
```

## Use

```bash
# single clip
python src/infer.py --clip path/to/flip.mp4

# a whole session folder (prints per-clip result + count of good flips)
python src/infer.py --clip "session_folder/" --csv results.csv
```

## Deploying to the phone browser

`runs/flip_classifier.onnx` runs in-browser with **onnxruntime-web** (WASM / WebGPU).
The app already records a flip clip and lets the user verify it — so the integration
is: sample 16 frames from the recorded clip → build a `(1,16,3,160,160)` tensor
(ImageNet-normalized) → run the model → `p(success)`. Output parity with PyTorch is
verified to 1e-6. The model is ~15 MB and loads once per session.

## Notes / data quality

- One clip (`clip_000490.mp4`) arrived corrupt (truncated, ~8 KB, no video stream) and
  was excluded. Worth re-exporting from the source if it's wanted.
- The handful of test errors are genuinely ambiguous clips (e.g. the bottle ends up
  held in-hand at the cap). More clips of those specific situations would push accuracy
  higher still.
