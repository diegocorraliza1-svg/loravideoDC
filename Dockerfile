# ── Wan2.2 Image-to-Video Worker ─────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Numpy first
RUN pip install --no-cache-dir "numpy==1.26.4"

# PyTorch
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Python deps
RUN pip install --no-cache-dir \
    "diffusers[torch]>=0.32.0" \
    "transformers>=4.46.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.2" \
    "huggingface-hub>=0.25.0" \
    "imageio[ffmpeg]>=2.34.0" \
    "imageio-ffmpeg>=0.5.1" \
    "ftfy>=6.2.3" \
    Pillow \
    requests \
    runpod

# Verify imports
RUN python -c "\
import numpy; print(f'numpy {numpy.__version__}'); \
import torch; print(f'torch {torch.__version__}'); \
import diffusers; print(f'diffusers {diffusers.__version__}'); \
import imageio; print(f'imageio {imageio.__version__}'); \
import ftfy; print(f'ftfy {ftfy.__version__}'); \
import runpod; print(f'runpod {runpod.__version__}'); \
print('ALL IMPORTS OK')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
