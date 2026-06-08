# meetily diarize WhisperX RunPod Serverless worker.
#
# Base: PyTorch 2.8 + CUDA 12.8 + cuDNN 9 runtime. If this tag becomes
# unavailable, any recent torch 2.x / CUDA 12.x runtime image works as a
# drop-in fallback (e.g. pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime),
# provided the ctranslate2 pin in requirements.txt stays CUDA-12 compatible.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.title="meetily-diarize-whisperx-worker"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Prefer the attached network volume for model caches so warm boots and
# subsequent workers reuse downloaded weights. These paths live under
# /runpod-volume/models (the volume mounts at /runpod-volume). If the volume is
# not attached, HuggingFace simply downloads into these paths on the container
# filesystem instead — WhisperX still works, just without cross-worker reuse.
ENV HF_HOME=/runpod-volume/models/hf
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/models/hf/hub
ENV TORCH_HOME=/runpod-volume/models/torch
# pyannote (diarization) honours PYANNOTE_CACHE; point it at the same place.
ENV PYANNOTE_CACHE=/runpod-volume/models/hf/hub

# System dependencies: ffmpeg for audio decoding, git for pip VCS installs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (torch is already provided by the base image).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: pre-download the gated diarization model at build time. This is
# guarded so the build succeeds WITHOUT the arg — when HF_TOKEN is unset the
# model downloads on first request instead. Requires the HF account to have
# accepted the user agreements for both:
#   pyannote/speaker-diarization-community-1
#   pyannote/segmentation-3.0
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
        python -c "import os; os.environ['HF_TOKEN']='${HF_TOKEN}'; \
from whisperx.diarize import DiarizationPipeline; \
DiarizationPipeline(token='${HF_TOKEN}', device='cpu'); \
print('Diarization model pre-downloaded.')"; \
    else \
        echo 'HF_TOKEN build-arg not provided; diarization model will download at runtime.'; \
    fi

COPY handler.py .

CMD ["python", "-u", "handler.py"]
