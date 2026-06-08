"""Drop-and-forget batch orchestrator.

Loop: watch INBOX_DIR, pick the oldest stable audio file, compress it to 16 kHz
mono FLAC, upload to the RunPod network volume, run the serverless diarization
job, write the transcript to OUTBOX_DIR, then delete the remote audio and archive
the input. Exactly one file is processed at a time (FIFO), so a cold RunPod
worker simply means the queue waits — nothing is lost.
"""
import os
import shutil
import subprocess
import sys
import time
import uuid

import config
import render
import runpod_client


def log(msg):
    print(f"[orchestrator] {msg}", flush=True)


def ensure_dirs():
    for d in (config.INBOX_DIR, config.OUTBOX_DIR, config.DONE_DIR,
              config.FAILED_DIR, config.WORK_DIR):
        os.makedirs(d, exist_ok=True)


def _is_audio(path):
    return os.path.splitext(path)[1].lower() in config.AUDIO_EXTS


def stable_files():
    """Audio files in the inbox that haven't been modified for STABLE_SECONDS
    (so we don't grab a file mid-copy), oldest first."""
    now = time.time()
    found = []
    for name in os.listdir(config.INBOX_DIR):
        if name.startswith("."):
            continue
        path = os.path.join(config.INBOX_DIR, name)
        if not os.path.isfile(path) or not _is_audio(path):
            continue
        st = os.stat(path)
        if now - st.st_mtime >= config.STABLE_SECONDS:
            found.append((st.st_mtime, path))
    found.sort()
    return [p for _, p in found]


def to_flac(src, dst):
    """Downmix to 16 kHz mono FLAC — lossless for ASR, ~10x smaller to upload."""
    cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", dst]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[-500:]}")


def process(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    job_id = uuid.uuid4().hex[:12]
    flac = os.path.join(config.WORK_DIR, f"{job_id}.flac")
    key = f"{config.REMOTE_PREFIX}/{job_id}.flac"

    log(f"processing '{name}' (job {job_id})")
    to_flac(path, flac)
    worker_path = runpod_client.upload_audio(flac, key)
    log(f"uploaded -> {worker_path}")

    try:
        payload = {
            "audio_path": worker_path,
            "diarize": config.DIARIZE,
            "language": config.LANGUAGE,
            "model": config.MODEL,
            "compute_type": config.COMPUTE_TYPE,
            "batch_size": config.BATCH_SIZE,
        }
        if config.MIN_SPEAKERS is not None:
            payload["min_speakers"] = config.MIN_SPEAKERS
        if config.MAX_SPEAKERS is not None:
            payload["max_speakers"] = config.MAX_SPEAKERS
        if config.INITIAL_PROMPT:
            payload["initial_prompt"] = config.INITIAL_PROMPT

        rp_job = runpod_client.submit(payload)
        log(f"submitted RunPod job {rp_job}; waiting for completion...")
        result = runpod_client.wait(
            rp_job,
            on_tick=lambda status, waited: log(f"  job {rp_job}: {status} ({waited}s)"),
        )
        if "error" in result:
            raise RuntimeError(f"worker error: {result['error']}")
        if not result.get("segments"):
            raise RuntimeError(f"worker returned no segments: {result}")

        out_stem = os.path.join(config.OUTBOX_DIR, stem)
        files = render.write_outputs(result, out_stem, name)
        log(f"done: {result.get('num_speakers', '?')} speakers, "
            f"{len(result['segments'])} segments -> {os.path.basename(files[0])} (+{len(files) - 1} more)")
    finally:
        if not config.KEEP_REMOTE_AUDIO:
            runpod_client.delete_audio(key)
        if os.path.exists(flac):
            os.remove(flac)

    if config.DELETE_INPUT_AFTER:
        os.remove(path)
    else:
        shutil.move(path, os.path.join(config.DONE_DIR, name))


def main():
    ensure_dirs()
    missing = config.validate()
    if missing:
        log(f"FATAL: missing required config: {', '.join(missing)}")
        sys.exit(1)

    log(f"watching {config.INBOX_DIR} | diarize={config.DIARIZE} model={config.MODEL} "
        f"language={config.LANGUAGE or 'auto'}")
    while True:
        try:
            queue = stable_files()
            if not queue:
                time.sleep(config.SCAN_INTERVAL)
                continue
            path = queue[0]
            try:
                process(path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the loop
                name = os.path.basename(path)
                log(f"FAILED '{name}': {exc}")
                dest = os.path.join(config.FAILED_DIR, name)
                try:
                    shutil.move(path, dest)
                except Exception:
                    dest = path
                with open(dest + ".error.txt", "w", encoding="utf-8") as fh:
                    fh.write(str(exc) + "\n")
        except KeyboardInterrupt:
            log("shutting down")
            break
        except Exception as exc:  # noqa: BLE001 - keep the watcher alive
            log(f"loop error: {exc}")
            time.sleep(config.SCAN_INTERVAL)


if __name__ == "__main__":
    main()
