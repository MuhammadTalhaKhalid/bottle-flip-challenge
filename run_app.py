"""
run_app.py — one-command launcher for the Bottle-Flip apps.

    python run_app.py streamlit     # interactive web app (default)
    python run_app.py api           # FastAPI + WebSocket service

Both serve the same trained model (runs/best.pt).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "streamlit"
    if target == "streamlit":
        cmd = [sys.executable, "-m", "streamlit", "run",
               os.path.join(ROOT, "streamlit_app.py")]
    elif target == "api":
        cmd = [sys.executable, "-m", "uvicorn", "api.server:app",
               "--host", "0.0.0.0", "--port", "8000"]
    else:
        print(f"unknown target {target!r} (use 'streamlit' or 'api')")
        sys.exit(1)
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
