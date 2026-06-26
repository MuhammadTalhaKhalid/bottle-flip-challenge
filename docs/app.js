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
  tries: document.getElementById("tries"),
  triesVal: document.getElementById("triesVal"),
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

function setStatus(msg) { els.status.textContent = msg; }

function currentConfig() {
  return {
    ...DEFAULT_CONFIG,
    gateMode: els.gate.value,
    confFloor: parseFloat(els.conf.value),
    maxTries: parseInt(els.tries.value, 10),
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
  // top scoreboard bar
  hctx.fillStyle = "rgba(10,8,15,0.45)";
  hctx.fillRect(0, 0, W, 0.13 * H);
  hctx.textBaseline = "alphabetic";
  hctx.fillStyle = "#7df9c0";
  hctx.font = `800 ${Math.round(0.075 * H)}px system-ui, sans-serif`;
  hctx.fillText(`FLIPS ${out.flips}`, 14, 0.092 * H);
  hctx.fillStyle = "#fff";
  hctx.font = `600 ${Math.round(0.04 * H)}px system-ui, sans-serif`;
  hctx.fillText(`Tries left: ${out.triesLeft}`, W * 0.52, 0.06 * H);
  hctx.fillStyle = "#c8c8c8";
  hctx.font = `500 ${Math.round(0.033 * H)}px system-ui, sans-serif`;
  hctx.fillText(`state: ${out.state}`, W * 0.52, 0.108 * H);

  // setup guidance
  if (out.message && !out.finished) {
    const ready = out.message === "READY";
    hctx.fillStyle = ready ? "#50dc78" : "#fac83c";
    hctx.font = `800 ${Math.round(0.06 * H)}px system-ui, sans-serif`;
    hctx.textAlign = "center";
    hctx.fillText(out.message, W / 2, 0.32 * H);
    hctx.textAlign = "left";
  }

  // verdict flash
  if (flash && flashLeft > 0) {
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

  // finished overlay
  if (out.finished) {
    hctx.fillStyle = "rgba(8,6,16,0.55)";
    hctx.fillRect(0, 0, W, H);
    hctx.fillStyle = "#78f0ff";
    hctx.font = `800 ${Math.round(0.1 * H)}px system-ui, sans-serif`;
    hctx.textAlign = "center";
    hctx.fillText(`FINAL  ${out.flips} / ${session.cfg.maxTries}`, W / 2, 0.48 * H);
    hctx.fillStyle = "#fff";
    hctx.font = `600 ${Math.round(0.04 * H)}px system-ui, sans-serif`;
    hctx.fillText("Tap New Game to play again", W / 2, 0.56 * H);
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
    const { bottles, nBottles } = await detectBottles(
      yoloSession, els.video, vw, vh, session.cfg.yoloConf);
    const pick = bottles.length ? bottles[0] : null;
    session.pushSmall(frameIdx, captureSmall(vw, vh));
    const out = await session.processFrame(pick, nBottles, vw, vh, frameIdx);
    if (out.verdict) { flash = [out.verdict.verdict, out.verdict.pSuccess]; flashLeft = Math.round(1.5 * fpsEMA); }
    drawHud(out);
    frameIdx += 1;

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
  session = new FlipSession(ASSUMED_FPS, currentConfig(), classifyLanding);
  frameIdx = 0; flash = null; flashLeft = 0;
  hctx.clearRect(0, 0, els.hud.width, els.hud.height);
}

async function startCamera() {
  setStatus("Requesting camera…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user",
               width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    fitHud();
    newGame();
    running = true;
    els.start.textContent = "⏹ Stop";
    setStatus("Live — point at the bottle and flip!");
    requestAnimationFrame(tick);
  } catch (e) {
    setStatus("Camera blocked. Allow camera access and reload. " + e.message);
  }
}

function stopCamera() {
  running = false;
  const s = els.video.srcObject;
  if (s) s.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;
  els.start.textContent = "▶ Start camera";
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
els.newgame.addEventListener("click", () => { if (session) newGame(); });
els.conf.addEventListener("input", () => {
  els.confVal.textContent = parseFloat(els.conf.value).toFixed(2);
  if (session) session.cfg.confFloor = parseFloat(els.conf.value);
});
els.tries.addEventListener("input", () => {
  els.triesVal.textContent = els.tries.value;
  if (session) session.cfg.maxTries = parseInt(els.tries.value, 10);
});
els.gate.addEventListener("change", () => { if (session) newGame(); });
window.addEventListener("resize", () => { if (running) fitHud(); });

els.start.disabled = true;
loadModels();

// register service worker for installability / offline caching
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("./sw.js").catch(() => {});
}
