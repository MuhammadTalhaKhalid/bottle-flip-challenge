/* engine.js — on-device port of src/realtime_engine.py (FlipSession).
 *
 * Same state machine as the trained Python system: a setup gate keeps the bottle
 * in-distribution, a throw must actually happen (rejects placements/pickups), and
 * once the bottle settles the 16-frame temporal classifier judges the landing.
 *
 * Inference is injected so this file stays framework-agnostic:
 *   onClassify(frames)  -> Promise<number>  // p(success); frames = [Uint8 RGB 160x160]
 */
export const BOTTLE_CLASS = 39;
export const SIZE = 160;
export const NUM_FRAMES = 16;  // MUST stay 16: the ONNX classifier's temporal axis
                               // is fixed at 16 (only 'batch' is dynamic). Feeding 8
                               // throws a shape mismatch and no landing ever classifies.

export const DEFAULT_CONFIG = {
  maxTries: 10,
  gateMode: "soft",        // "hard" | "soft" | "hard_calibrate"
  requireThrow: true,
  confFloor: 0.60,         // success threshold (accuracy, not speed). Live app
                           // overrides this from the UI slider; keep at the
                           // validated 0.60 — lower = more false SUCCESS.
  // The values below are RESTORED to the validated realtime_engine.py config
  // (97.96% acc). DeepSeek's "speed optimization" loosened them, which shrank
  // the [throw→landing] window fed to the temporal classifier so it saw mostly
  // mid-air/tumbling frames and read every successful landing as FAIL. These
  // constants define WHICH frames get classified — they are accuracy knobs, not
  // free speed knobs. Do not loosen them again to chase latency.
  minBoxFrac: 0.20,        // framing gate (bottle box height / frame height)
  maxBoxFrac: 0.80,
  centerTol: 0.32,         // |bottle_cx - 0.5| must be <= this
  readySecs: 0.8,          // steady-in-zone hold to pass the setup gate
  settleSecs: 0.55,        // stillness that means "at rest" — MUST stay ~0.55 so
                           // the classify window keeps enough settled-upright
                           // frames; shortening this is what caused all-FAIL.
  throwSpeed: 28.0,        // px/frame peak speed => a real throw
  lookbackSecs: 1.8,       // window collected for the landing classifier
  yoloConf: 0.20,          // bottle-detection confidence
  // Stillness threshold is size-aware. 7 px/frame fits the trained framing (bottle
  // small/far), but a bottle CLOSE to the camera is large on-screen, so the same
  // hand jitter spans more pixels and never drops below 7 -> it never registers as
  // "settled" -> the landing is never judged ("close flips aren't detected"). Scale
  // the threshold up only once boxFrac passes stillBoxRef, so normal/far framing is
  // UNCHANGED (== stillMotion) and only close bottles loosen.
  stillMotion: 7.0,        // base stillness threshold (px/frame)
  stillBoxRef: 0.50,       // boxFrac at/below which threshold == stillMotion
  // A thrown attempt that never comes to rest within this long did NOT stay
  // standing: judge it anyway so failed flips get a verdict instead of silently
  // showing nothing ("failed flips show no result"). The classifier still labels it.
  // 3.5s (vs 2.5) lets a slow-settling SUCCESS reach the normal settle-judge first,
  // cutting false-FAILs on real successes (A/B: 3 -> 1 of 65) for ~1s more latency on
  // fails that never settle (which don't affect the score anyway).
  maxAttemptSecs: 3.5,
};

function pushCapped(arr, v, max) {
  arr.push(v);
  if (arr.length > max) arr.shift();
}

export class FlipSession {
  constructor(fps, config, onClassify) {
    this.fps = fps;
    this.cfg = config;
    this.onClassify = onClassify;       // async (frames) => p

    this.flips = 0;
    this.triesUsed = 0;
    this.finished = false;
    this.events = [];
    this.lastVerdict = null;

    this.gatePassed = config.gateMode === "soft";
    this.readyRun = 0;
    this.calibBoxFrac = null;

    this.prevCenter = null;
    this.motionHist = [];               // maxlen 5
    this.L = Math.max(1, Math.round(config.lookbackSecs * fps));
    this.winMotion = [];                // maxlen L
    this.winDetected = [];              // maxlen L
    this.settleFrames = Math.max(1, Math.round(config.settleSecs * fps));
    this.maxAttemptFrames = Math.max(1, Math.round(config.maxAttemptSecs * fps));
    this.stillRun = 0;
    this.judged = true;
    this.throwStart = null;

    this.smallBuf = [];                 // {idx, rgb:Uint8Array(160*160*3)}
    this.smallMax = Math.max(1, Math.round(4.0 * fps));
  }

  pushSmall(idx, rgb) {
    this.smallBuf.push({ idx, rgb });
    if (this.smallBuf.length > this.smallMax) this.smallBuf.shift();
  }

  _motion(center) {
    if (this.prevCenter === null) return 0.0;
    const d = Math.hypot(center[0] - this.prevCenter[0],
                         center[1] - this.prevCenter[1]);
    pushCapped(this.motionHist, d, 5);
    return this.motionHist.reduce((a, b) => a + b, 0) / this.motionHist.length;
  }

  // pick: {conf, center:[cx,cy], ratio, box:[x1,y1,x2,y2], boxFrac, cx} | null
  _checkGate(pick, nBottles, frameW) {
    if (pick === null) { this.readyRun = 0; return [false, "Show the bottle"]; }
    if (nBottles > 1) { this.readyRun = 0; return [false, "Only one bottle in frame"]; }

    const boxFrac = pick.boxFrac;
    const cxFrac = pick.cx / frameW;
    let lo = this.cfg.minBoxFrac, hi = this.cfg.maxBoxFrac;
    if (this.cfg.gateMode === "hard_calibrate" && this.calibBoxFrac) {
      lo = this.calibBoxFrac * 0.6; hi = this.calibBoxFrac * 1.6;
    }
    if (boxFrac < lo) { this.readyRun = 0; return [false, "Move closer"]; }
    if (boxFrac > hi) { this.readyRun = 0; return [false, "Move back"]; }
    if (Math.abs(cxFrac - 0.5) > this.cfg.centerTol) {
      this.readyRun = 0; return [false, "Center the bottle"];
    }
    this.readyRun += 1;
    if (this.readyRun >= Math.round(this.cfg.readySecs * this.fps)) {
      if (this.cfg.gateMode === "hard_calibrate" && this.calibBoxFrac === null) {
        this.calibBoxFrac = boxFrac;
      }
      this.gatePassed = true;
      return [true, "READY"];
    }
    return [false, "Hold steady..."];
  }

  // Classify the [sIdx..eIdx] window and record one judged attempt. Returns the
  // event, or null if no real throw happened. Shared by the settle path and the
  // never-settled finalize path.
  async _judge(start, frameIdx) {
    const recent = this.winMotion.filter((m) => m >= 0);
    const peak = recent.length ? Math.max(...recent) : 0.0;
    const blur = this.winDetected.filter((d) => d === 0).length;
    const threw = peak >= this.cfg.throwSpeed || blur >= 2;
    if (!(threw || !this.cfg.requireThrow)) return null;
    const p = await this._classify(start, frameIdx);
    const success = p >= this.cfg.confFloor;
    if (success) this.flips += 1;
    this.triesUsed += 1;
    const verdict = success ? "SUCCESS" : "FAIL";
    const ev = {
      try: this.triesUsed, verdict, pSuccess: Math.round(p * 1e4) / 1e4,
      startS: Math.round((start / this.fps) * 100) / 100,
      endS: Math.round((frameIdx / this.fps) * 100) / 100,
    };
    this.events.push(ev);
    this.lastVerdict = [verdict, p, Math.round(1.4 * this.fps)];
    if (this.triesUsed >= this.cfg.maxTries) this.finished = true;
    return ev;
  }

  // Judge a still-pending attempt when a round ends. Without this, a failed flip the
  // round ends on (bottle never came to rest) would show no verdict. Returns ev|null.
  async finalize(frameIdx) {
    if (this.finished || this.judged || this.throwStart === null) return null;
    const ev = await this._judge(this.throwStart, frameIdx);
    this.judged = true;
    this.throwStart = null;
    this.stillRun = 0;
    return ev;
  }

  async _classify(sIdx, eIdx) {
    const frames = this.smallBuf
      .filter((s) => s.idx >= sIdx && s.idx <= eIdx)
      .map((s) => s.rgb);
    if (frames.length === 0) return 0.0;
    const sel = [];
    for (let i = 0; i < NUM_FRAMES; i++) {
      const t = NUM_FRAMES === 1 ? 0 : i / (NUM_FRAMES - 1);
      let j = Math.round(t * (frames.length - 1));
      j = Math.min(Math.max(j, 0), frames.length - 1);
      sel.push(frames[j]);
    }
    return await this.onClassify(sel);
  }

  // pick may be null. Returns the same dict shape as the Python engine.
  async processFrame(pick, nBottles, frameW, frameH, frameIdx) {
    const out = {
      frame: frameIdx, flips: this.flips, triesUsed: this.triesUsed,
      triesLeft: Math.max(0, this.cfg.maxTries - this.triesUsed),
      finished: this.finished, state: "IDLE", gateOk: this.gatePassed,
      message: "", bottleBox: pick ? pick.box.map((v) => Math.round(v)) : null,
      verdict: null,
    };
    if (this.finished) {
      out.state = "DONE";
      out.message = `Final score: ${this.flips}/${this.cfg.maxTries}`;
      return out;
    }

    // --- setup gate ---
    if (!this.gatePassed || this.cfg.gateMode !== "soft") {
      const [, msg] = this._checkGate(pick, nBottles, frameW);
      out.gateOk = this.gatePassed;
      out.message = msg;
      if ((this.cfg.gateMode === "hard" || this.cfg.gateMode === "hard_calibrate")
          && !this.gatePassed) {
        out.state = "SETUP";
        return out;
      }
    }

    // --- flip segmentation (gate passed) ---
    if (pick === null) {
      pushCapped(this.winMotion, -1.0, this.L);
      pushCapped(this.winDetected, 0, this.L);
      this.stillRun = 0;
      if (!this.judged && this.throwStart === null) this.throwStart = frameIdx;
      this.judged = false;
      this.prevCenter = null;
      out.state = "AIRBORNE";
    } else {
      const center = pick.center;
      const motion = this._motion(center);
      pushCapped(this.winMotion, motion, this.L);
      pushCapped(this.winDetected, 1, this.L);
      // validated stillness threshold (realtime_engine.py), made size-aware: it is
      // exactly stillMotion (7.0) for normal/far framing (boxFrac <= stillBoxRef)
      // and only scales up for a close, large bottle so it can still settle.
      const stillThresh = this.cfg.stillMotion *
        Math.max(1.0, pick.boxFrac / this.cfg.stillBoxRef);
      if (motion < stillThresh) {
        this.stillRun += 1;
      } else {
        if (this.judged || this.throwStart === null) this.throwStart = frameIdx;
        this.stillRun = 0;
        this.judged = false;
      }
      this.prevCenter = center;
      out.state = this.stillRun >= this.settleFrames ? "REST" : "ACTIVE";

      // landing reached -> judge once
      if (this.stillRun >= this.settleFrames && !this.judged) {
        const start = this.throwStart !== null ? this.throwStart
          : Math.max(0, frameIdx - Math.round(this.cfg.lookbackSecs * this.fps));
        const ev = await this._judge(start, frameIdx);
        if (ev) out.verdict = ev;
        this.judged = true;
        this.throwStart = null;
      }
    }

    // A thrown attempt that never comes to rest = a failed flip that would otherwise
    // show nothing. Once it has been in flight longer than a real flip could take,
    // judge it now so the user still gets a verdict. Runs whether the bottle is
    // rolling in-frame or was thrown out of frame.
    if (!this.judged && this.throwStart !== null
        && this.stillRun < this.settleFrames
        && frameIdx - this.throwStart >= this.maxAttemptFrames) {
      const ev = await this._judge(this.throwStart, frameIdx);
      if (ev && !out.verdict) out.verdict = ev;
      this.judged = true;
      this.throwStart = null;
      this.stillRun = 0;
    }

    out.flips = this.flips;
    out.triesUsed = this.triesUsed;
    out.triesLeft = Math.max(0, this.cfg.maxTries - this.triesUsed);
    out.finished = this.finished;
    return out;
  }
}