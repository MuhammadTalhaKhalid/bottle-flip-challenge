/* yolo.js — YOLOv8n bottle detection in the browser via onnxruntime-web.
 *
 * Mirrors how src/realtime_engine.py uses YOLO: detect COCO class 39 (bottle),
 * pick the highest-confidence one, return its box in original video coordinates
 * plus the framing stats the gate needs.
 */
import { BOTTLE_CLASS } from "./engine.js";

const YOLO_SIZE = 480;          // must match the exported onnx input
const IOU_THRES = 0.45;

// reusable letterbox canvas
const lbCanvas = document.createElement("canvas");
lbCanvas.width = YOLO_SIZE; lbCanvas.height = YOLO_SIZE;
const lbCtx = lbCanvas.getContext("2d", { willReadFrequently: true });

let _input = new Float32Array(3 * YOLO_SIZE * YOLO_SIZE);

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
  const inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function nms(dets) {
  dets.sort((p, q) => q.score - p.score);
  const keep = [];
  for (const d of dets) {
    if (keep.every((k) => iou(d, k) < IOU_THRES)) keep.push(d);
  }
  return keep;
}

/* Run detection on a <video> (or canvas) element of size vw x vh.
 * Returns { bottles: [...sorted by score], nBottles } with boxes in video coords.
 */
export async function detectBottles(session, source, vw, vh, confThres) {
  // letterbox into YOLO_SIZE square (gray pad), preserving aspect ratio
  const scale = Math.min(YOLO_SIZE / vw, YOLO_SIZE / vh);
  const nw = Math.round(vw * scale), nh = Math.round(vh * scale);
  const padX = Math.floor((YOLO_SIZE - nw) / 2);
  const padY = Math.floor((YOLO_SIZE - nh) / 2);
  lbCtx.fillStyle = "rgb(114,114,114)";
  lbCtx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);
  lbCtx.drawImage(source, 0, 0, vw, vh, padX, padY, nw, nh);
  const { data } = lbCtx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE);

  // HWC RGBA uint8 -> CHW RGB float /255
  const area = YOLO_SIZE * YOLO_SIZE;
  for (let i = 0; i < area; i++) {
    _input[i] = data[i * 4] / 255;                 // R
    _input[area + i] = data[i * 4 + 1] / 255;       // G
    _input[2 * area + i] = data[i * 4 + 2] / 255;   // B
  }
  const inName = session.inputNames[0];
  const outName = session.outputNames[0];
  const tensor = new ort.Tensor("float32", _input, [1, 3, YOLO_SIZE, YOLO_SIZE]);
  const result = await session.run({ [inName]: tensor });
  const out = result[outName];                      // [1, 84, N]
  const dims = out.dims;
  const N = dims[2];                                // anchors
  const d = out.data;

  // row layout: [cx, cy, w, h, cls0..cls79] each as a length-N stripe
  const clsOff = (4 + BOTTLE_CLASS) * N;
  const dets = [];
  for (let a = 0; a < N; a++) {
    const score = d[clsOff + a];
    if (score < confThres) continue;
    const cx = d[a], cy = d[N + a], w = d[2 * N + a], h = d[3 * N + a];
    dets.push({
      score,
      x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
    });
  }
  const kept = nms(dets);

  // map letterbox coords -> original video coords
  const bottles = kept.map((b) => {
    const x1 = Math.min(Math.max((b.x1 - padX) / scale, 0), vw);
    const y1 = Math.min(Math.max((b.y1 - padY) / scale, 0), vh);
    const x2 = Math.min(Math.max((b.x2 - padX) / scale, 0), vw);
    const y2 = Math.min(Math.max((b.y2 - padY) / scale, 0), vh);
    const w = Math.max(x2 - x1, 1e-6), h = Math.max(y2 - y1, 1e-6);
    const cx = (x1 + x2) / 2;
    return {
      conf: b.score, center: [cx, (y1 + y2) / 2], ratio: h / w,
      box: [x1, y1, x2, y2], boxFrac: h / vh, cx,
    };
  }).sort((p, q) => q.conf - p.conf);

  return { bottles, nBottles: bottles.length };
}
