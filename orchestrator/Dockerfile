# Local orchestrator — runs on Proxmox. Only needs ffmpeg + a little Python.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="meetily-diarize-orchestrator"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py /app/

ENV PYTHONUNBUFFERED=1
VOLUME ["/data"]

CMD ["python", "-u", "watcher.py"]
