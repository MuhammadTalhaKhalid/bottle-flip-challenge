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

The earlier prototype (`bottle_flip_counter.py`) used a pretrained COCO bottle
detector + a hand-tuned aspect-ratio state machine. It hit a ceiling: a "tall" box
can't tell upright from upside-down, and detection drops out during fast flight.

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

Two front-ends serve the same trained model (`runs/best.pt`):

```bash
pip install -r requirements-app.txt        # (install torch/torchvision first)

# 1) Interactive web app — webcam challenge + upload-a-clip
python run_app.py streamlit                 # -> http://localhost:8501

# 2) FastAPI + WebSocket service (for the Lovable / browser front-end)
python run_app.py api                       # -> http://localhost:8000  (/ , /health, /ws)
```

**`streamlit_app.py`** is the player-facing app:
- **🎥 Live Challenge** — opens the webcam (`streamlit-webrtc`), runs YOLOv8 bottle
  detection + the landing classifier on every frame, and draws a live HUD (flip
  count, tries left, READY gate, SUCCESS/FAIL flash, final score) straight onto the
  video. Reuses `src/realtime_engine.py` — identical logic to the API.
- **📤 Upload a Flip** — upload a recorded clip, get an animated verdict + confidence.
  Works on hosts where live webcam streaming is blocked (e.g. Streamlit Cloud).

**Deploying:** the Streamlit app runs anywhere Python does; the live (webcam) mode
needs a STUN/TURN reachable network (a public STUN is preconfigured). On CPU hosts it
auto-prefers `yolov8n.pt` for a smoother feed. The FastAPI service is containerised
(`Dockerfile`) and is what the Lovable phone front-end connects to over WebSocket.

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
streamlit_app.py     interactive web app (live webcam challenge + upload-a-clip)
run_app.py           launcher: `python run_app.py streamlit | api`
api/
  server.py          FastAPI + WebSocket service wrapping realtime_engine
  requirements.txt   API service deps
requirements-app.txt deps for both web apps
runs/
  best.pt            trained checkpoint
  flip_classifier.onnx  browser-deployable model (~15 MB)
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
