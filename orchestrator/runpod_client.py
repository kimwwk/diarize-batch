"""Thin RunPod client: upload audio to the network volume (S3 API) and run a
serverless job, polling until it completes.

We deliberately use plain requests + boto3 rather than the runpod SDK so the
control flow (submit -> poll -> fetch output) is fully visible and easy to debug.
"""
import time

import boto3
import requests
from botocore.config import Config as BotoConfig

import config


# --------------------------------------------------------------------------- #
# Audio transfer via the RunPod S3-compatible network volume
# --------------------------------------------------------------------------- #
def _s3():
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        region_name=config.S3_REGION,
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def upload_audio(local_path, key):
    """Upload a local file to the network volume. Returns the path the worker sees."""
    _s3().upload_file(local_path, config.VOLUME_ID, key)
    return f"/runpod-volume/{key}"


def delete_audio(key):
    """Remove the uploaded audio from the volume (best-effort)."""
    try:
        _s3().delete_object(Bucket=config.VOLUME_ID, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001 - cleanup must never crash the run
        print(f"[runpod] warning: could not delete remote {key}: {exc}", flush=True)
        return False


# --------------------------------------------------------------------------- #
# Serverless job submission + polling
# --------------------------------------------------------------------------- #
def _headers():
    return {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def submit(input_payload):
    """Submit an async job; return the RunPod job id."""
    url = f"{config.RUNPOD_BASE_URL}/{config.RUNPOD_ENDPOINT_ID}/run"
    resp = requests.post(url, json={"input": input_payload}, headers=_headers(), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("id")
    if not job_id:
        raise RuntimeError(f"no job id in RunPod response: {data}")
    return job_id


def wait(job_id, poll=None, timeout=None, on_tick=None):
    """Poll /status until COMPLETED; return the job output dict.

    Raises on FAILED/CANCELLED/TIMED_OUT or local timeout.
    """
    poll = poll or config.POLL_INTERVAL
    timeout = timeout or config.JOB_TIMEOUT
    url = f"{config.RUNPOD_BASE_URL}/{config.RUNPOD_ENDPOINT_ID}/status/{job_id}"
    waited = 0
    while True:
        resp = requests.get(url, headers=_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data.get("output") or {}
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"RunPod job {job_id} {status}: {data.get('error') or data}")
        if waited >= timeout:
            raise TimeoutError(
                f"RunPod job {job_id} still {status} after {timeout}s — giving up"
            )
        if on_tick:
            on_tick(status, waited)
        time.sleep(poll)
        waited += poll
