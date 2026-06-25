"""
server.py — FastAPI + WebSocket service that wraps the real-time flip engine.

The browser (e.g. the Lovable front-end) opens a WebSocket, streams camera
frames as JPEG bytes, and receives one JSON state object per frame:

    {
      "gate_ok": bool, "message": "Move closer" | "READY" | ...,
      "state": "SETUP" | "ACTIVE" | "AIRBORNE" | "REST" | "DONE",
      "flips": int, "tries_left": int, "finished": bool,
      "bottle_box": [x1,y1,x2,y2] | null,
      "verdict": {"try":1,"verdict":"SUCCESS","p_success":0.96,...} | null
    }

Endpoints:
    GET  /health                          liveness + device + model info
    GET  /                                a self-contained demo camera client
    WS   /ws?max_tries=5&gate_mode=hard&fps=30&conf_floor=0.6
          - send: binary JPEG frame  ->  recv: JSON state (above)
          - send: text "reset"       ->  restarts the session, recv {"reset":true}

One WebSocket connection == one play session. Models load once at startup and
are shared across connections.

Run locally:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
import os
import sys

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from realtime_engine import FlipSession, SessionConfig, load_models  # noqa: E402

app = FastAPI(title="Bottle Flip Detection API", version="1.0")

STATE = {"device": None, "yolo": None, "clf": None}


@app.on_event("startup")
def _startup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = os.environ.get("FLIP_WEIGHTS")           # default: runs/best.pt
    yolo_path = os.environ.get("YOLO_WEIGHTS")          # default: ~/yolov8m.pt
    yolo, clf = load_models(device, weights, yolo_path)
    STATE.update(device=device, yolo=yolo, clf=clf)
    print(f"[startup] models loaded on {device}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": STATE["device"],
        "models_loaded": STATE["clf"] is not None,
    }


def _make_session(params):
    def _f(name, cast, default):
        try:
            return cast(params.get(name, default))
        except Exception:
            return default
    cfg = SessionConfig(
        max_tries=_f("max_tries", int, 5),
        gate_mode=str(params.get("gate_mode", "hard")),
        conf_floor=_f("conf_floor", float, 0.60),
    )
    fps = _f("fps", float, 30.0) or 30.0
    return FlipSession(fps, cfg, STATE["yolo"], STATE["clf"], STATE["device"]), cfg


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    params = dict(ws.query_params)
    sess, cfg = _make_session(params)
    frame_idx = 0
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                if msg["text"].strip().lower() == "reset":
                    sess, cfg = _make_session(params)
                    frame_idx = 0
                    await ws.send_json({"reset": True})
                continue
            data = msg.get("bytes")
            if not data:
                continue
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({"error": "could not decode frame"})
                continue
            st = sess.process_frame(frame, frame_idx)
            frame_idx += 1
            await ws.send_json(st)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def demo():
    return HTMLResponse(_DEMO_HTML)


_DEMO_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Bottle Flip — Live Demo</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
 #wrap{position:relative;display:inline-block;margin-top:10px}
 video{max-height:80vh;border-radius:8px}
 #hud{position:absolute;top:10px;left:12px;text-align:left;text-shadow:0 0 4px #000}
 #flips{font-size:34px;font-weight:700}
 #msg{font-size:24px;color:#ffd24d}
 #verdict{font-size:28px;font-weight:700}
 button{font-size:16px;padding:8px 18px;margin:8px;border:0;border-radius:6px;cursor:pointer}
 #box{position:absolute;border:3px solid #0c0;border-radius:4px;display:none;pointer-events:none}
</style></head><body>
<h3>Bottle Flip Detection — live demo</h3>
<div>
  Tries <input id="tries" type="number" value="5" min="1" style="width:48px">
  Gate
  <select id="gate"><option value="hard">hard</option>
   <option value="soft">soft</option><option value="hard_calibrate">hard+calibrate</option></select>
  <button onclick="start()">Start camera</button>
  <button onclick="resetSession()">Reset</button>
</div>
<div id="wrap">
  <video id="v" autoplay playsinline muted></video>
  <div id="box"></div>
  <div id="hud">
    <div id="flips">Flips: 0</div>
    <div id="tl">Tries left: -</div>
    <div id="st">state: -</div>
    <div id="msg"></div>
    <div id="verdict"></div>
  </div>
</div>
<canvas id="c" style="display:none"></canvas>
<script>
let ws, stream, sending=false;
const v=document.getElementById('v'), c=document.getElementById('c'), box=document.getElementById('box');
async function start(){
  if(!stream){
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:720,height:1280}});
    v.srcObject=stream; await v.play();
  }
  const tries=document.getElementById('tries').value, gate=document.getElementById('gate').value;
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws?max_tries=${tries}&gate_mode=${gate}&fps=30`);
  ws.binaryType='arraybuffer';
  ws.onmessage=e=>{ if(typeof e.data==='string') render(JSON.parse(e.data)); };
  ws.onopen=()=>{ sending=true; loop(); };
  ws.onclose=()=>{ sending=false; };
}
function resetSession(){ if(ws&&ws.readyState===1) ws.send('reset'); }
function loop(){
  if(!sending||!ws||ws.readyState!==1) return;
  const w=v.videoWidth, h=v.videoHeight;
  if(w&&h){ c.width=w; c.height=h; c.getContext('2d').drawImage(v,0,0,w,h);
    c.toBlob(b=>{ if(b&&ws.readyState===1) b.arrayBuffer().then(a=>ws.send(a)); }, 'image/jpeg', 0.6);
  }
  setTimeout(loop, 80); // ~12 fps upload
}
function render(s){
  if(s.reset){ document.getElementById('verdict').textContent=''; return; }
  if(s.error) return;
  document.getElementById('flips').textContent='Flips: '+(s.flips??0);
  document.getElementById('tl').textContent='Tries left: '+(s.tries_left??'-');
  document.getElementById('st').textContent='state: '+(s.state??'-');
  document.getElementById('msg').textContent=s.gate_ok?'':(s.message||'');
  if(s.finished){ document.getElementById('verdict').style.color='#4dff88';
    document.getElementById('verdict').textContent='DONE '+s.flips+' flips'; }
  else if(s.verdict){ const ok=s.verdict.verdict==='SUCCESS';
    document.getElementById('verdict').style.color=ok?'#4dff88':'#ff5b5b';
    document.getElementById('verdict').textContent=s.verdict.verdict+' p='+s.verdict.p_success.toFixed(2);
    setTimeout(()=>{document.getElementById('verdict').textContent='';},1400); }
  // bottle box overlay (scaled from video px to displayed px)
  if(s.bottle_box){ const r=v.getBoundingClientRect(), wr=document.getElementById('wrap').getBoundingClientRect();
    const sx=v.clientWidth/(v.videoWidth||1), sy=v.clientHeight/(v.videoHeight||1);
    const [x1,y1,x2,y2]=s.bottle_box;
    box.style.display='block';
    box.style.left=(r.left-wr.left+x1*sx)+'px'; box.style.top=(r.top-wr.top+y1*sy)+'px';
    box.style.width=((x2-x1)*sx)+'px'; box.style.height=((y2-y1)*sy)+'px';
    box.style.borderColor=s.gate_ok?'#0c0':'#fc0';
  } else box.style.display='none';
}
</script></body></html>"""
