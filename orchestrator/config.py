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

# --- RunPod S3-compatible network volume (used to ship the audio file) ---
# The volume id IS the S3 bucket name. A file uploaded under key "foo/bar.flac"
# is visible to the worker at "/runpod-volume/foo/bar.flac".
S3_ENDPOINT = os.environ.get("RUNPOD_S3_ENDPOINT", "").strip()   # e.g. https://s3api-us-il-1.runpod.io/
S3_REGION = os.environ.get("RUNPOD_S3_REGION", "").strip()       # e.g. us-il-1
S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY", "").strip()
S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY", "").strip()
VOLUME_ID = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "").strip()  # = S3 bucket name
REMOTE_PREFIX = os.environ.get("REMOTE_PREFIX", "diarize-inbox").strip("/")

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
    "RUNPOD_ENDPOINT_ID": RUNPOD_ENDPOINT_ID,
    "RUNPOD_S3_ENDPOINT": S3_ENDPOINT,
    "RUNPOD_S3_REGION": S3_REGION,
    "RUNPOD_S3_ACCESS_KEY": S3_ACCESS_KEY,
    "RUNPOD_S3_SECRET_KEY": S3_SECRET_KEY,
    "RUNPOD_NETWORK_VOLUME_ID": VOLUME_ID,
}


def validate():
    """Return the list of required env vars that are missing/empty."""
    return [name for name, value in _REQUIRED.items() if not value]
