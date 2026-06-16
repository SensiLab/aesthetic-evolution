FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SKETCH_DIR=/app/Harmonograph

# System dependencies: Python 3.13, Java (required by Processing), xvfb (headless sketch rendering)
RUN apt-get update && apt-get install -y \
        software-properties-common wget curl unzip git \
        xvfb libxi6 libxrender1 libxtst6 libxext6 libgl1 libglu1-mesa \
        openjdk-17-jre \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.13 python3.13-dev python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.13 -m ensurepip && python3.13 -m pip install --upgrade pip setuptools wheel

# Install Processing 4.5.3 and expose processing-java on PATH
RUN wget -q \
    "https://github.com/processing/processing4/releases/download/processing-1314-4.5.3/processing-4.5.3-linux-x64-portable.zip" \
    -O /tmp/processing.zip \
    && unzip -q /tmp/processing.zip -d /opt/ \
    && rm /tmp/processing.zip \
    && ln -s /opt/processing-4.5.3/processing-java /usr/local/bin/processing-java

# Install ControlP5 library into the Processing sketchbook
RUN mkdir -p /root/sketchbook/libraries \
    && wget -q "https://github.com/sojamo/controlp5/releases/download/v2.2.6/controlP5-2.2.6.zip" \
    -O /tmp/controlP5.zip \
    && unzip -q /tmp/controlP5.zip -d /root/sketchbook/libraries/ \
    && rm /tmp/controlP5.zip

WORKDIR /app

# PyTorch with CUDA 12.8 — installed before other deps so the CUDA headers are present
# when flash-attn compiles against them in the next step
RUN python3.13 -m pip install --no-cache-dir \
    torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# flash-attn build dependencies must be pre-installed because --no-build-isolation
# prevents pip from pulling them in automatically
RUN python3.13 -m pip install --no-cache-dir psutil packaging ninja

# flash-attn must be compiled from source against the CUDA devel headers in this image.
# This step takes ~10-20 minutes on first build but is cached on subsequent builds.
RUN python3.13 -m pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Remaining production dependencies
# blinker must be force-reinstalled first — the Ubuntu base image ships a distutils-installed
# blinker 1.4 that pip cannot cleanly upgrade when Flask pulls in a newer version
COPY requirements.txt .
RUN python3.13 -m pip install --no-cache-dir --ignore-installed blinker && \
    python3.13 -m pip install --no-cache-dir -r requirements.txt

# Copy application code (Harmonograph is mounted at runtime, not baked in)
COPY . .

EXPOSE 5000 5050
