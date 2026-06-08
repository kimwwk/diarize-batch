"""Configuration for the diarize-batch orchestrator, all via environment variables.

Nothing here talks to the Meetily app or any existing RunPod template — this is a
standalone, drop-and-forget batch pipeline. Audio goes to a RunPod serverless
endpoint for GPU transcription+diarization; the resulting transcript is written
only to the local OUTBOX_DIR (your Proxmox box). The uploaded audio is deleted
from RunPod after each job unless KEEP_REMOTE_AUDIO is set.
"""
import os


def _bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int_or_none(name):
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


# --- RunPod serverless endpoint (the NEW endpoint you create for the worker) ---
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
RUNPOD_BASE_URL = os.environ.get("RUNPOD_BASE_URL", "https://api.runpod.ai/v2").rstrip("/")

# --- HuggingFace token (passed to the pod for the diarization models) ---
HF_TOKEN = (os.environ.get("HF_TOKEN", "") or os.environ.get("HF_AUTH_TOKEN", "")).strip()

# --- On-demand POD config (the orchestrator creates/destroys this) ---
POD_NAME = os.environ.get("POD_NAME", "diarize-batch-pod").strip()
POD_IMAGE = os.environ.get("POD_IMAGE", "kimwwk/meetily-diarize-whisperx-worker:pod").strip()
POD_DC = os.environ.get("POD_DC", "EUR-IS-3").strip()
POD_VOLUME_ID = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "").strip()  # warm model cache; blank = none
POD_DISK_GB = int(os.environ.get("POD_DISK_GB", "30"))
POD_GPU_IDS = [g.strip() for g in os.environ.get(
    "POD_GPU_IDS",
    "NVIDIA RTX A5000,NVIDIA GeForce RTX 4090,NVIDIA L4,NVIDIA A40,NVIDIA L40,"
    "NVIDIA L40S,NVIDIA GeForce RTX 5090,NVIDIA A100-SXM4-80GB,NVIDIA H100 80GB HBM3"
).split(",") if g.strip()]
POD_BOOT_TIMEOUT = int(os.environ.get("POD_BOOT_TIMEOUT", "600"))      # secs to boot + /health
POD_IDLE_SECONDS = int(os.environ.get("POD_IDLE_MINUTES", "5")) * 60    # tear down after this idle

# --- SSH tunnel to the pod (key-gated; the API stays on the pod's localhost) ---
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "/secrets/pod_ed25519").strip()
SSH_PUBKEY_PATH = SSH_KEY_PATH + ".pub"
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", "8000"))   # local end of the tunnel

# --- Local folders (inside the container; map /data to a host dir via compose) ---
INBOX_DIR = os.environ.get("INBOX_DIR", "/data/inbox")
OUTBOX_DIR = os.environ.get("OUTBOX_DIR", "/data/outbox")
DONE_DIR = os.environ.get("DONE_DIR", "/data/done")
FAILED_DIR = os.environ.get("FAILED_DIR", "/data/failed")
WORK_DIR = os.environ.get("WORK_DIR", "/data/work")

# --- Transcription / diarization options (passed to the worker) ---
LANGUAGE = os.environ.get("LANGUAGE", "").strip() or None   # None => autodetect
DIARIZE = _bool("DIARIZE", True)
MODEL = os.environ.get("MODEL", "large-v3").strip()
MIN_SPEAKERS = _int_or_none("MIN_SPEAKERS")
MAX_SPEAKERS = _int_or_none("MAX_SPEAKERS")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16").strip()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
INITIAL_PROMPT = os.environ.get("INITIAL_PROMPT", "").strip() or None  # seed Whisper w/ domain vocab/names

# --- Orchestrator behaviour ---
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "15"))     # seconds between status checks
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "7200"))       # give up on a job after this many seconds
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "5"))      # seconds between inbox scans
STABLE_SECONDS = int(os.environ.get("STABLE_SECONDS", "10"))   # file must be untouched this long before pickup
KEEP_REMOTE_AUDIO = _bool("KEEP_REMOTE_AUDIO", False)          # keep the uploaded audio on the volume
DELETE_INPUT_AFTER = _bool("DELETE_INPUT_AFTER", False)        # delete input instead of moving to done/
AUDIO_EXTS = {
    e.strip().lower()
    for e in os.environ.get(
        "AUDIO_EXTS", ".mp4,.m4a,.wav,.flac,.mp3,.aac,.ogg,.webm,.opus,.mkv"
    ).split(",")
    if e.strip()
}

_REQUIRED = {
    "RUNPOD_API_KEY": RUNPOD_API_KEY,
    "HF_TOKEN": HF_TOKEN,
}


def validate():
    """Return the list of required config that is missing/empty."""
    missing = [name for name, value in _REQUIRED.items() if not value]
    if not os.path.exists(SSH_PUBKEY_PATH):
        missing.append(f"ssh key file ({SSH_PUBKEY_PATH})")
    return missing
