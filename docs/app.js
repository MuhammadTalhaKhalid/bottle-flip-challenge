/* app.js — wires the camera, the two ONNX models, and the FlipSession engine
 * into a real-time on-device bottle-flip counter. Everything runs on the phone;
 * no server, no upload.
 */
import { FlipSession, DEFAULT_CONFIG, SIZE, NUM_FRAMES } from "./engine.js";
import { detectBottles } from "./yolo.js";

const els = {
  video: document.getElementById("cam"),
  hud: document.getElementById("hud"),
  start: document.getElementById("startBtn"),
  newgame: document.getElementById("newGameBtn"),
  status: document.getElementById("status"),
  gate: document.getElementById("gateMode"),
  conf: document.getElementById("confFloor"),
  confVal: document.getElementById("confVal"),
  timerSel: document.getElementById("timerSel"),
  fps: document.getElementById("fpsTag"),
};
const hctx = els.hud.getContext("2d");

// ImageNet normalization (matches src/dataset.py)
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const ASSUMED_FPS = 12;

// small-frame capture canvas (160x160 RGB for the classifier)
const smallCanvas = document.createElement("canvas");
smallCanvas.width = SIZE; smallCanvas.height = SIZE;
const sctx = smallCanvas.getContext("2d", { willReadFrequently: true });

let yoloSession = null, clfSession = null;
let session = null;          // FlipSession
let running = false, frameIdx = 0, busy = false;
let flash = null, flashLeft = 0;
let lastT = 0, fpsEMA = ASSUMED_FPS;
// timed-challenge state: a countdown round that ends and shows a final score.
let challenge = { active: false, finished: false, endsAt: 0, remaining: 30, duration: 30, finalFlips: 0 };

function setStatus(msg) { els.status.textContent = msg; }

function currentConfig() {
  return {
    ...DEFAULT_CONFIG,
    gateMode: els.gate.value,
    confFloor: parseFloat(els.conf.value),
    maxTries: 9999,   // timed mode: the countdown ends the round, not a tries cap
  };
}

// Build the (1,16,3,160,160) tensor from 16 Uint8 RGB frames and classify.
async function classifyLanding(frames) {
  const fb = NUM_FRAMES, ch = 3, hw = SIZE * SIZE;
  const data = new Float32Array(fb * ch * hw);
  for (let f = 0; f < fb; f++) {
    const rgb = frames[f];
    const base = f * ch * hw;
    for (let p = 0; p < hw; p++) {
      data[base + p] = (rgb[p * 3] / 255 - MEAN[0]) / STD[0];
      data[base + hw + p] = (rgb[p * 3 + 1] / 255 - MEAN[1]) / STD[1];
      data[base + 2 * hw + p] = (rgb[p * 3 + 2] / 255 - MEAN[2]) / STD[2];
    }
  }
  const inName = clfSession.inputNames[0];
  const outName = clfSession.outputNames[0];
  const t = new ort.Tensor("float32", data, [1, fb, ch, SIZE, SIZE]);
  const res = await clfSession.run({ [inName]: t });
  const logits = res[outName].data;             // [fail, success]
  const m = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - m), e1 = Math.exp(logits[1] - m);
  return e1 / (e0 + e1);
}

function captureSmall(vw, vh) {
  // center-crop-free resize to 160x160 RGB (matches cv2.resize behavior closely)
  sctx.drawImage(els.video, 0, 0, vw, vh, 0, 0, SIZE, SIZE);
  const { data } = sctx.getImageData(0, 0, SIZE, SIZE);
  const rgb = new Uint8Array(SIZE * SIZE * 3);
  for (let p = 0; p < SIZE * SIZE; p++) {
    rgb[p * 3] = data[p * 4];
    rgb[p * 3 + 1] = data[p * 4 + 1];
    rgb[p * 3 + 2] = data[p * 4 + 2];
  }
  return rgb;
}

// ---- HUD drawing (overlay canvas sized to the displayed video) ----
function fitHud() {
  const r = els.video.getBoundingClientRect();
  els.hud.width = r.width; els.hud.height = r.height;
}
function fmtTime(s) {
  s = Math.max(0, Math.ceil(s));
  const m = Math.floor(s / 60), ss = s % 60;
  return m > 0 ? `${m}:${ss.toString().padStart(2, "0")}` : `${ss}s`;
}

function drawHud(out) {
  const W = els.hud.width, H = els.hud.height;
  hctx.clearRect(0, 0, W, H);
  const sx = W / (els.video.videoWidth || W);
  const sy = H / (els.video.videoHeight || H);

  // bottle box
  if (out.bottleBox) {
    const [x1, y1, x2, y2] = out.bottleBox;
    hctx.lineWidth = 3;
    hctx.strokeStyle = out.gateOk ? "#50dc78" : "#fac83c";
    hctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
  }

  // top scoreboard bar: FLIPS (left) + countdown (right)
  hctx.fillStyle = "rgba(10,8,15,0.45)";
  hctx.fillRect(0, 0, W, 0.13 * H);
  hctx.textBaseline = "alphabetic";
  hctx.textAlign = "left";
  hctx.fillStyle = "#7df9c0";
  hctx.font = `800 ${Math.round(0.075 * H)}px system-ui, sans-serif`;
  hctx.fillText(`FLIPS ${out.flips}`, 14, 0.092 * H);

  const rem = challenge.active ? challenge.remaining : challenge.duration;
  const low = challenge.active && rem <= 5;
  hctx.save();
  if (low) hctx.globalAlpha = 0.45 + 0.55 * Math.abs(Math.sin(performance.now() / 140));
  hctx.fillStyle = low ? "#fb5b6e" : "#ffffff";
  hctx.font = `800 ${Math.round(0.07 * H)}px system-ui, sans-serif`;
  hctx.textAlign = "right";
  hctx.fillText(`⏱ ${fmtTime(rem)}`, W - 14, 0.092 * H);
  hctx.restore();
  hctx.textAlign = "left";

  // setup / preview guidance
  if (out.message && !challenge.finished) {
    hctx.fillStyle = out.gateOk ? "#50dc78" : "#fac83c";
    hctx.font = `800 ${Math.round(0.055 * H)}px system-ui, sans-serif`;
    hctx.textAlign = "center";
    hctx.fillText(out.message, W / 2, 0.30 * H);
    hctx.textAlign = "left";
  }

  // verdict flash
  if (flash && flashLeft > 0 && !challenge.finished) {
    const [text, p] = flash;
    const good = text === "SUCCESS";
    hctx.fillStyle = "rgba(10,8,15,0.5)";
    hctx.fillRect(0, 0.40 * H, W, 0.20 * H);
    hctx.fillStyle = good ? "#34d399" : "#fb5b6e";
    hctx.font = `800 ${Math.round(0.11 * H)}px system-ui, sans-serif`;
    hctx.textAlign = "center";
    hctx.fillText(`${good ? "✓" : "✗"} ${text}`, W / 2, 0.52 * H);
    hctx.fillStyle = "#fff";
    hctx.font = `600 ${Math.round(0.045 * H)}px system-ui, sans-serif`;
    hctx.fillText(`confidence ${(p * 100).toFixed(0)}%`, W / 2, 0.575 * H);
    hctx.textAlign = "left";
    flashLeft -= 1;
  }

  // time-up final overlay
  if (challenge.finished) {
    hctx.fillStyle = "rgba(8,6,16,0.62)";
    hctx.fillRect(0, 0, W, H);
    hctx.textAlign = "center";
    hctx.fillStyle = "#fde68a";
    hctx.font = `800 ${Math.round(0.058 * H)}px system-ui, sans-serif`;
    hctx.fillText("⏱ TIME'S UP", W / 2, 0.34 * H);
    hctx.fillStyle = "#78f0ff";
    hctx.font = `800 ${Math.round(0.17 * H)}px system-ui, sans-serif`;
    hctx.fillText(`${challenge.finalFlips}`, W / 2, 0.55 * H);
    hctx.fillStyle = "#fff";
    hctx.font = `700 ${Math.round(0.05 * H)}px system-ui, sans-serif`;
    hctx.fillText(challenge.finalFlips === 1 ? "flip" : "flips", W / 2, 0.63 * H);
    hctx.fillStyle = "#cbd5ff";
    hctx.font = `600 ${Math.round(0.042 * H)}px system-ui, sans-serif`;
    hctx.fillText("Tap 🏁 Play Again", W / 2, 0.75 * H);
    hctx.textAlign = "left";
  }
}

// ---- main loop ----
async function tick() {
  if (!running) return;
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  if (!vw || busy) { requestAnimationFrame(tick); return; }
  busy = true;
  try {
    // advance the countdown; end the round at zero
    if (challenge.active) {
      challenge.remaining = Math.max(0, (challenge.endsAt - performance.now()) / 1000);
      if (challenge.remaining <= 0) finishChallenge();
    }

    if (challenge.finished) {
      drawHud({ flips: challenge.finalFlips, state: "DONE", message: "",
                gateOk: false, bottleBox: null });
    } else {
      const { bottles, nBottles } = await detectBottles(
        yoloSession, els.video, vw, vh, session.cfg.yoloConf);
      const pick = bottles.length ? bottles[0] : null;
      const box = pick ? pick.box.map((v) => Math.round(v)) : null;

      if (challenge.active) {
        session.pushSmall(frameIdx, captureSmall(vw, vh));
        const out = await session.processFrame(pick, nBottles, vw, vh, frameIdx);
        if (out.verdict) {
          flash = [out.verdict.verdict, out.verdict.pSuccess];
          flashLeft = Math.round(1.2 * fpsEMA);
        }
        drawHud(out);
        frameIdx += 1;
      } else {
        // camera live, round not started yet — show preview box + prompt
        drawHud({ flips: 0, state: "READY", gateOk: !!pick, bottleBox: box,
                  message: pick ? "Tap 🏁 Start Challenge" : "Show the bottle" });
      }
    }

    // fps estimate
    const now = performance.now();
    if (lastT) { const inst = 1000 / Math.max(now - lastT, 1); fpsEMA = 0.9 * fpsEMA + 0.1 * inst; }
    lastT = now;
    els.fps.textContent = `${fpsEMA.toFixed(1)} FPS`;
  } catch (e) {
    console.error(e);
  } finally {
    busy = false;
    requestAnimationFrame(tick);
  }
}

function newGame() {
  // Use the MEASURED processing rate, not a fixed guess. The engine converts its
  // "seconds" constants (settleSecs, lookbackSecs, readySecs) to frame counts via
  // this fps. Hard-coding 12 while the phone really runs ~3-5 fps inflated every
  // wait ~3x — that's the "flip → ~5s → verdict" latency the client reported, and
  // it also stretched the classify window. fpsEMA is warm by Start Challenge
  // (the preview loop runs detection every frame). Clamp to a sane band so a cold
  // or spiky estimate can't produce degenerate frame counts.
  const fps = Math.min(30, Math.max(3, Math.round(fpsEMA)));
  session = new FlipSession(fps, currentConfig(), classifyLanding);
  frameIdx = 0; flash = null; flashLeft = 0;
  hctx.clearRect(0, 0, els.hud.width, els.hud.height);
}

// Begin (or restart) a timed round: fresh score + a countdown that ends it.
function startChallenge() {
  if (!running) { setStatus("Start the camera first."); return; }
  newGame();
  challenge.duration = parseInt(els.timerSel.value, 10);
  challenge.remaining = challenge.duration;
  challenge.endsAt = performance.now() + challenge.duration * 1000;
  challenge.finalFlips = 0;
  challenge.active = true;
  challenge.finished = false;
  els.newgame.textContent = "🔄 Restart";
  setStatus(`Go! ${challenge.duration}s — flip as many as you can!`);
}

function finishChallenge() {
  challenge.active = false;
  challenge.finished = true;
  challenge.finalFlips = session ? session.flips : 0;
  els.newgame.textContent = "🏁 Play Again";
  setStatus(`Time! Final score: ${challenge.finalFlips} flip${challenge.finalFlips === 1 ? "" : "s"}.`);
}

// Front (selfie) camera is what the client wants. Try to force it with `exact`;
// if a browser refuses that constraint, fall back to `ideal` so the camera still
// opens (front-preferred) instead of hard-failing.
async function getFrontStream() {
  const size = { width: { ideal: 480, max: 640 }, height: { ideal: 360, max: 480 } };
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: "user" }, ...size }, audio: false,
    });
  } catch (e) {
    if (e && (e.name === "OverconstrainedError" || e.name === "NotFoundError")) {
      return await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "user" }, ...size }, audio: false,
      });
    }
    throw e;
  }
}

async function startCamera() {
  setStatus("Requesting camera…");
  try {
    const stream = await getFrontStream();
    els.video.srcObject = stream;
    await els.video.play();
    fitHud();
    newGame();
    challenge.active = false; challenge.finished = false;
    challenge.duration = parseInt(els.timerSel.value, 10);
    challenge.remaining = challenge.duration;
    running = true;
    els.start.textContent = "⏹ Stop";
    els.newgame.textContent = "🏁 Start Challenge";
    setStatus("Front camera live — frame the bottle, then tap Start Challenge.");
    requestAnimationFrame(tick);
  } catch (e) {
    setStatus("Camera blocked. Allow camera access and reload. " + e.message);
  }
}

function stopCamera() {
  running = false;
  challenge.active = false; challenge.finished = false;
  const s = els.video.srcObject;
  if (s) s.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;
  els.start.textContent = "▶ Start camera";
  els.newgame.textContent = "🏁 Start Challenge";
  hctx.clearRect(0, 0, els.hud.width, els.hud.height);
  setStatus("Stopped.");
}

async function loadModels() {
  setStatus("Loading AI models (one-time, ~27 MB)…");
  ort.env.wasm.simd = true;
  // GitHub Pages can't set COOP/COEP, so wasm threads are unavailable; webgpu
  // (when present) is far faster, otherwise single-thread wasm still works.
  const ep = ["webgpu", "wasm"];
  try {
    yoloSession = await ort.InferenceSession.create("./models/yolov8n.onnx",
      { executionProviders: ep, graphOptimizationLevel: "all" });
    clfSession = await ort.InferenceSession.create("./models/flip_classifier.onnx",
      { executionProviders: ep, graphOptimizationLevel: "all" });
    setStatus("Models ready. Tap Start camera.");
    els.start.disabled = false;
  } catch (e) {
    setStatus("Failed to load models: " + e.message);
    console.error(e);
  }
}

// ---- UI wiring ----
els.start.addEventListener("click", () => (running ? stopCamera() : startCamera()));
els.newgame.addEventListener("click", startChallenge);
els.conf.addEventListener("input", () => {
  els.confVal.textContent = parseFloat(els.conf.value).toFixed(2);
  if (session) session.cfg.confFloor = parseFloat(els.conf.value);
});
els.timerSel.addEventListener("change", () => {
  challenge.duration = parseInt(els.timerSel.value, 10);
  if (!challenge.active) challenge.remaining = challenge.duration;
});
els.gate.addEventListener("change", () => { if (session && !challenge.active) newGame(); });
window.addEventListener("resize", () => { if (running) fitHud(); });

els.start.disabled = true;
loadModels();

// register service worker for installability / offline caching
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}