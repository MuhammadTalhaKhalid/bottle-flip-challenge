"""
streamlit_app.py — Bottle-Flip Challenge, mobile-first web app.

Designed to be opened in a phone browser. Two ways to test the trained system:

  📲 Record / Upload  — (PRIMARY, works on any phone) tap to record a flip with
                        your phone camera or pick a clip, and get an animated
                        SUCCESS / FAIL verdict with the model's confidence.
  🎥 Live Challenge   — (experimental on free hosting) open the camera and the
                        app counts good flips live. Detection (YOLOv8) + the
                        trained landing classifier run per frame with an on-screen
                        scoreboard. Needs a strong connection; the free server is
                        CPU-only, so the live feed can lag — use Record/Upload if
                        it doesn't connect.

Run locally:
    streamlit run streamlit_app.py

The live mode reuses the exact same engine the FastAPI service wraps
(src/realtime_engine.py), so what you see here is what a deployed player gets.
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from model import FlipClassifier  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from cache_frames import decode_clip  # noqa: E402

# Mobile-first: single centered column, sidebar hidden by default (phones don't
# show the sidebar well — settings live in an on-page expander instead).
st.set_page_config(page_title="Bottle Flip Challenge", page_icon="🍾",
                   layout="centered", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------- #
#  Styling — game feel + touch-friendly sizing on small screens.
# ----------------------------------------------------------------------------- #
st.markdown("""
<style>
  .stApp { background: radial-gradient(900px 500px at 50% -10%, #1b2440 0%, #0d1020 60%); }
  /* tighten the default page padding so the phone gets more usable height */
  .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 720px; }
  .hero { text-align:center; padding: 2px 0; }
  .hero h1 { font-size: 2.1rem; margin: 0; font-weight: 800;
             background: linear-gradient(90deg,#6ee7ff,#a78bfa,#fef08a);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .hero p { color:#9aa4c4; margin:.25rem 0 0; font-size:.98rem; }
  .tips { color:#9aa4c4; font-size:.92rem; line-height:1.6; }
  .verdict-good { color:#34d399; font-weight:800; font-size:2.2rem; margin:.2rem 0; }
  .verdict-bad  { color:#fb7185; font-weight:800; font-size:2.2rem; margin:.2rem 0; }
  div[data-testid="stMetricValue"] { font-size: 2.1rem; }
  /* big, thumb-friendly tap targets */
  .stButton > button { padding:.7rem 1rem; font-size:1.05rem; border-radius:14px; }
  .stTabs [data-baseweb="tab"] { font-size:1rem; padding:.4rem .7rem; }
  /* the file-uploader dropzone, sized for a thumb */
  section[data-testid="stFileUploaderDropzone"] { padding: 1.1rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- #
#  Model loading (cached once per server process).
# ----------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading detector + landing classifier…")
def get_engine_models():
    from realtime_engine import load_models  # lazy: only needed for live mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # prefer the lighter detector on CPU deployments for a smoother live feed
    yolo_path = None
    if device == "cpu":
        cand = os.path.join(os.path.expanduser("~"), "yolov8n.pt")
        if os.path.exists(cand):
            yolo_path = cand
    yolo, clf = load_models(device, weights=os.path.join(ROOT, "runs", "best.pt"),
                            yolo_path=yolo_path)
    return device, yolo, clf


@st.cache_resource(show_spinner="Loading landing classifier…")
def get_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = FlipClassifier().to(device)
    ckpt = torch.load(os.path.join(ROOT, "runs", "best.pt"),
                      map_location=device, weights_only=False)
    clf.load_state_dict(ckpt["model"])
    clf.eval()
    return device, clf


# ----------------------------------------------------------------------------- #
#  HUD — draw the game state straight onto the video frame (live mode).
# ----------------------------------------------------------------------------- #
GREEN = (80, 220, 120)
AMBER = (60, 200, 250)
RED = (90, 90, 240)
WHITE = (240, 240, 240)


def _bar(img, y0, y1, alpha=0.45):
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (img.shape[1], y1), (15, 12, 8), -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _text(img, s, org, scale, color, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def draw_hud(frame, st_obj, cfg, verdict_flash):
    H, W = frame.shape[:2]
    # bottle box
    if st_obj.get("bottle_box"):
        x1, y1, x2, y2 = st_obj["bottle_box"]
        c = GREEN if st_obj.get("gate_ok") else AMBER
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)

    # top scoreboard bar
    _bar(frame, 0, int(0.13 * H))
    _text(frame, f"FLIPS {st_obj.get('flips', 0)}", (16, int(0.09 * H)),
          1.1 * H / 480, GREEN, 2)
    tl = st_obj.get("tries_left", "-")
    _text(frame, f"Tries left: {tl}", (int(W * 0.52), int(0.06 * H)),
          0.7 * H / 480, WHITE, 2)
    _text(frame, f"state: {st_obj.get('state', '-')}", (int(W * 0.52), int(0.11 * H)),
          0.6 * H / 480, (200, 200, 200), 1)

    # setup guidance (only when the gate isn't satisfied)
    msg = st_obj.get("message", "")
    if msg and not st_obj.get("finished"):
        ready = msg == "READY"
        col = GREEN if ready else AMBER
        sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * H / 480, 3)[0]
        _text(frame, msg, ((W - sz[0]) // 2, int(0.30 * H)), 1.0 * H / 480, col, 3)

    # verdict flash banner
    if verdict_flash is not None:
        text, p = verdict_flash
        good = text == "SUCCESS"
        col = GREEN if good else RED
        big = f"{'✓' if good else '✗'} {text}"
        sz = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 1.7 * H / 480, 4)[0]
        _bar(frame, int(0.40 * H), int(0.60 * H), 0.5)
        _text(frame, big, ((W - sz[0]) // 2, int(0.52 * H)), 1.7 * H / 480, col, 4)
        sub = f"confidence {p:.0%}"
        ss = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * H / 480, 2)[0]
        _text(frame, sub, ((W - ss[0]) // 2, int(0.58 * H)), 0.8 * H / 480, WHITE, 2)

    # finished overlay
    if st_obj.get("finished"):
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W, H), (10, 8, 20), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        f = f"FINAL  {st_obj.get('flips', 0)} / {cfg.max_tries}"
        sz = cv2.getTextSize(f, cv2.FONT_HERSHEY_SIMPLEX, 1.8 * H / 480, 4)[0]
        _text(frame, f, ((W - sz[0]) // 2, int(0.48 * H)), 1.8 * H / 480, (120, 240, 255), 4)
        sub = "Tap  New Game  to play again"
        ss = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * H / 480, 2)[0]
        _text(frame, sub, ((W - ss[0]) // 2, int(0.56 * H)), 0.7 * H / 480, WHITE, 2)
    return frame


# ----------------------------------------------------------------------------- #
#  Header
# ----------------------------------------------------------------------------- #
st.markdown("<div class='hero'><h1>🍾 Bottle Flip Challenge</h1>"
            "<p>Flip the bottle. The AI judges the landing.</p></div>",
            unsafe_allow_html=True)


# ----------------------------------------------------------------------------- #
#  Settings — in an on-page expander so phone users can reach them (no sidebar).
# ----------------------------------------------------------------------------- #
with st.expander("⚙️  Settings & how to play"):
    max_tries = st.slider("Tries per game (live mode)", 1, 15, 5)
    gate_mode = st.selectbox(
        "Setup gate (live mode)", ["soft", "hard", "hard_calibrate"], index=0,
        help="soft = always count (most fun) · hard = only judge once the bottle "
             "is framed like the training data (most accurate) · hard_calibrate = "
             "hard + learns your bottle size once.")
    conf_floor = st.slider("Success confidence floor", 0.40, 0.90, 0.60, 0.05,
                           help="Minimum model confidence to count a landing as good.")
    fps_hint = st.slider("Engine FPS hint (live mode)", 10, 30, 20,
                         help="Match your camera's effective frame rate for best timing.")
    st.markdown("<div class='tips'><b>How to play</b><br>"
                "1. Hold the phone low, bottle in view (low angle = most accurate).<br>"
                "2. Record a flip (or go live) — a good landing = bottle upright &amp; still.<br>"
                "3. The AI calls SUCCESS or FAIL with its confidence.<br>"
                "4. Beat your high score 🏆</div>", unsafe_allow_html=True)


def make_config():
    from realtime_engine import SessionConfig
    return SessionConfig(max_tries=max_tries, gate_mode=gate_mode,
                         conf_floor=conf_floor)


# Record/Upload first — it's the reliable path on a phone.
tab_upload, tab_live = st.tabs(["📲  Record / Upload", "🎥  Live Challenge"])


# ----------------------------------------------------------------------------- #
#  RECORD / UPLOAD  (primary mobile flow)
# ----------------------------------------------------------------------------- #
with tab_upload:
    st.markdown("**Tap below to record a flip** with your phone camera (or pick a "
                "saved clip). The AI judges the landing in a couple of seconds.")
    up = st.file_uploader("Record or choose a flip clip",
                          type=["mp4", "mov", "webm", "avi"],
                          label_visibility="collapsed")
    if up is not None:
        suffix = os.path.splitext(up.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(up.read())
            tmp_path = tf.name
        st.video(tmp_path)
        with st.spinner("Analysing the flip…"):
            device, clf = get_classifier()
            arr = decode_clip(tmp_path)  # (T,H,W,3) uint8, rotation-aware
            if arr is None:
                st.error("Couldn't read that video. Try another file.")
            else:
                x = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)
                x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
                with torch.no_grad(), torch.autocast(
                        device_type="cuda", enabled=device == "cuda"):
                    logit = clf(x)
                p = torch.softmax(logit.float(), 1)[0, 1].item()
                good = p >= conf_floor
                if good:
                    st.markdown("<div class='verdict-good'>✓ SUCCESS</div>",
                                unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown("<div class='verdict-bad'>✗ FAILED LANDING</div>",
                                unsafe_allow_html=True)
                st.metric("Model confidence (good landing)", f"{p:.1%}")
                st.progress(float(p))
                st.caption(f"Decision threshold: {conf_floor:.0%}  ·  "
                           f"device: {device.upper()}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ----------------------------------------------------------------------------- #
#  LIVE CHALLENGE  (experimental on free hosting)
# ----------------------------------------------------------------------------- #
with tab_live:
    st.caption("⚠️ Experimental on the free server (CPU-only, shared). If the live "
               "feed won't start or lags, use **Record / Upload** — it's just as "
               "accurate and works on every phone.")
    try:
        import av  # noqa: F401
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        WEBRTC_OK = True
    except Exception as e:  # noqa: BLE001
        WEBRTC_OK = False
        st.error(f"Live mode needs streamlit-webrtc (`pip install streamlit-webrtc av`). {e}")

    if WEBRTC_OK:
        device, yolo, clf = get_engine_models()
        cfg = make_config()
        from realtime_engine import FlipSession

        if "game_id" not in st.session_state:
            st.session_state.game_id = 0
        if st.button("🔄  New Game", use_container_width=True, type="primary"):
            st.session_state.game_id += 1
        st.caption("Tap **START**, allow the camera, and flip. The scoreboard is "
                   "drawn right on the video.")

        class FlipProcessor:
            def __init__(self):
                self.session = FlipSession(float(fps_hint), cfg, yolo, clf, device)
                self.fidx = 0
                self._flash = None      # (text, p)
                self._flash_left = 0
                self._flash_ttl = int(1.5 * fps_hint)

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                st_obj = self.session.process_frame(img, self.fidx)
                self.fidx += 1
                # latch a new verdict for a short on-screen flash
                if st_obj.get("verdict"):
                    v = st_obj["verdict"]
                    self._flash = (v["verdict"], v["p_success"])
                    self._flash_left = self._flash_ttl
                flash = None
                if self._flash_left > 0:
                    flash = self._flash
                    self._flash_left -= 1
                img = draw_hud(img, st_obj, cfg, flash)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key=f"flip-{st.session_state.game_id}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FlipProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        st.info("Tip: a low camera angle (near floor level) matches the training "
                "data best — that's where the model is most accurate.", icon="💡")


st.markdown("<hr style='border-color:#222'>", unsafe_allow_html=True)
st.caption("Same model as the FastAPI service (runs/best.pt) · 99.3% validation "
           "accuracy · YOLOv8 bottle detection + MobileNetV3 temporal landing classifier.")
