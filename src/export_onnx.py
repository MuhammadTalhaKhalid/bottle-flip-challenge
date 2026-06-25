"""
export_onnx.py — export the trained classifier to ONNX for in-browser inference.

The deployment target is a phone browser. MobileNetV3 + this head exports cleanly to
ONNX, which runs in the browser via onnxruntime-web (WASM/WebGL/WebGPU). The app
samples 16 frames from a recorded flip clip, builds a (1,16,3,160,160) tensor, and
runs the model to get p(success).

    python src/export_onnx.py

Produces runs/flip_classifier.onnx and verifies parity with the PyTorch model.
"""
import os

import numpy as np
import torch

from model import FlipClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
T, SIZE = 16, 160


def main():
    device = "cpu"
    model = FlipClassifier().to(device)
    ckpt = torch.load(os.path.join(RUNS, "best.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, T, 3, SIZE, SIZE)
    onnx_path = os.path.join(RUNS, "flip_classifier.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["frames"], output_names=["logits"],
        dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {onnx_path}")

    # parity check against onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pt = model(dummy).numpy()
        ox = sess.run(None, {"frames": dummy.numpy()})[0]
        diff = np.abs(pt - ox).max()
        print(f"PyTorch vs ONNX max abs diff: {diff:.2e}  "
              f"({'OK' if diff < 1e-3 else 'CHECK'})")
    except ImportError:
        print("onnxruntime not installed; skipped parity check "
              "(pip install onnxruntime to verify).")


if __name__ == "__main__":
    main()
