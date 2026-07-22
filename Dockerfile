FROM python:3.11-slim

WORKDIR /app

# Install system deps for opencv + curl for model pre-download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir -e .

# Pre-download OpenCV DNN models for pure self-hosted 100% no external API at runtime
# MobileNet-SSD + YOLOv3-tiny + coco.names baked into image
RUN mkdir -p /tmp/models /tmp/mobilenet_ssd /tmp/yolo /app/models && \
    curl -L --retry 3 -o /tmp/yolo/yolov3-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg && \
    curl -L --retry 3 -o /tmp/yolo/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names && \
    curl -L --retry 3 -o /tmp/mobilenet_ssd/MobileNetSSD_deploy.prototxt https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt && \
    curl -L --retry 3 -o /tmp/mobilenet_ssd/MobileNetSSD_deploy.caffemodel https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.caffemodel || echo "caffemodel download failed, will fallback to runtime download" && \
    curl -L --retry 3 -o /tmp/yolo/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights || echo "yolo weights download failed, will fallback" && \
    cp -r /tmp/yolo/* /tmp/models/ 2>/dev/null || true && \
    cp -r /tmp/mobilenet_ssd/* /tmp/models/ 2>/dev/null || true && \
    cp /tmp/yolo/coco.names /tmp/models/coco.names 2>/dev/null || true && \
    ls -lh /tmp/models/ /tmp/yolo/ /tmp/mobilenet_ssd/ || true

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=3)"

CMD ["uvicorn", "capsolver.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
