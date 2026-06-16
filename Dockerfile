# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG SECAI_MODEL_REPO=MaxHaufe/LASR-SECAI-DEMO
ARG SECAI_MODEL_REVISION=main
ARG SECAI_DOWNLOAD_MODELS=false

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    SECAI_DEMO_MODEL=mobilevit_s \
    SECAI_MODEL_REPO=${SECAI_MODEL_REPO} \
    SECAI_MODEL_REVISION=${SECAI_MODEL_REVISION} \
    SECAI_MODELS_DIR=/app/models \
    XDG_RUNTIME_DIR=/tmp/runtime-secai

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libfreetype6 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libice6 \
        libsm6 \
        libv4l-0 \
        libx11-6 \
        libx11-xcb1 \
        libxcb-cursor0 \
        libxcb-glx0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-util1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" \
        "torchvision>=0.17" \
    && python -m pip install -r requirements.txt \
        "huggingface_hub>=0.23,<1"

COPY config ./config
COPY src ./src
COPY docker ./docker

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/models /opt/huggingface /tmp/runtime-secai \
    && chmod 700 /tmp/runtime-secai \
    && if [ "${SECAI_DOWNLOAD_MODELS}" = "true" ]; then \
        python /app/docker/download_models.py \
          --required-model efficientnet_b0 \
          --required-model mobilevit_s \
          --required-model mobilevitv2_100; \
    fi

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["model=mobilevit_s"]
