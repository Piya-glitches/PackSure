FROM python:3.11-slim

# libgl1/libglib2.0-0: OpenCV. libgomp1: Paddle inference runtime.
# (libzbar0 is gone: zxing-cpp ships self-contained wheels.)
# fonts-dejavu-core: sample-label generator.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # paddleocr drags in the GUI opencv wheels; keep ONE cv2 (headless) on disk
    && pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless || true \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.10.0.84

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
