# Local orchestrator — runs on Proxmox. Needs ffmpeg + a little Python, plus a
# CPU speaker-embedding model (resemblyzer) for the optional speaker-id tagging.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="meetily-diarize-orchestrator"

# runtime deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch first so resemblyzer doesn't drag in the multi-GB CUDA build;
# build-essential/python3-dev only compile webrtcvad, then get purged to stay lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY *.py /app/

ENV PYTHONUNBUFFERED=1
VOLUME ["/data"]

CMD ["python", "-u", "watcher.py"]
