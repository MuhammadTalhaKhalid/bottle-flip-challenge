# Bottle Flip Detection — real-time API
# CPU image for portability (deploy on any host). For GPU, see api/README.md.
FROM python:3.11-slim

# OpenCV (headless) + ultralytics runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first (smaller, no CUDA), then the rest of the deps
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# App code + trained classifier weights
COPY src/ /app/src/
COPY api/ /app/api/
COPY runs/best.pt /app/runs/best.pt

# Bake the YOLO detector into the image so startup needs no network
# (ultralytics downloads yolov8m.pt into the working dir /app)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
ENV YOLO_WEIGHTS=/app/yolov8m.pt
ENV FLIP_WEIGHTS=/app/runs/best.pt
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
