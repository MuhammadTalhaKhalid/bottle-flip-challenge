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
  minBoxFrac: 0.15,        // ← LOWER from 0.20 (easier to detect)
  maxBoxFrac: 0.85,        // ← HIGHER from 0.80 (more tolerant)
  centerTol: 0.40,         // ← HIGHER from 0.32 (easier to center)
  readySecs: 0.3,          // ← REDUCED from 0.8 (faster setup)
  settleSecs: 0.25,        // ← REDUCED from 0.55 (faster judgement)
  throwSpeed: 20.0,        // ← REDUCED from 28.0 (easier to detect throw)
  lookbackSecs: 1.2,       // ← REDUCED from 1.8 (less frames to collect)
  yoloConf: 0.15,          // ← REDUCED from 0.20 (faster detection)
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
      if (motion < 5.0) {   // ← LOWER from 7.0 (faster stillness detection)
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
        const recent = this.winMotion.filter((m) => m >= 0);
        const peak = recent.length ? Math.max(...recent) : 0.0;
        const blur = this.winDetected.filter((d) => d === 0).length;
        const threw = peak >= this.cfg.throwSpeed || blur >= 2;
        const start = this.throwStart !== null ? this.throwStart
          : Math.max(0, frameIdx - Math.round(this.cfg.lookbackSecs * this.fps));

        if (threw || !this.cfg.requireThrow) {
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
          out.verdict = ev;
          this.lastVerdict = [verdict, p, Math.round(1.4 * this.fps)];
          if (this.triesUsed >= this.cfg.maxTries) this.finished = true;
        }
        this.judged = true;
        this.throwStart = null;
      }
    }

    out.flips = this.flips;
    out.triesUsed = this.triesUsed;
    out.triesLeft = Math.max(0, this.cfg.maxTries - this.triesUsed);
    out.finished = this.finished;
    return out;
  }
}