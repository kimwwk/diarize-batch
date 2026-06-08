"""Drop-and-forget orchestrator (pod mode).

Watch INBOX_DIR. When a file lands: ensure an on-demand RunPod pod is up, downmix
to 16 kHz mono FLAC, POST it to the pod's FastAPI over an SSH tunnel (no base64,
no public endpoint), render the transcript to OUTBOX_DIR, and archive the input.
When the inbox has been idle for POD_IDLE_MINUTES, tear the pod down ($0 idle).
"""
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime

import config
import pod_manager
import render


def log(msg):
    print(f"[orchestrator] {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


def ensure_dirs():
    for d in (config.INBOX_DIR, config.OUTBOX_DIR, config.DONE_DIR,
              config.FAILED_DIR, config.WORK_DIR):
        os.makedirs(d, exist_ok=True)


def _is_audio(path):
    return os.path.splitext(path)[1].lower() in config.AUDIO_EXTS


def stable_files():
    """Audio files untouched for STABLE_SECONDS (so we don't grab a mid-copy file)."""
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
    cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", dst]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[-400:]}")


def process(path, t_detect):
    """Transcribe one file via the (already-up) pod and write outputs."""
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    job_id = uuid.uuid4().hex[:8]
    flac = os.path.join(config.WORK_DIR, f"{job_id}.flac")
    to_flac(path, flac)
    try:
        log(f"uploading '{name}' to pod ...")
        result = pod_manager.infer(
            flac, language=config.LANGUAGE,
            min_speakers=config.MIN_SPEAKERS, max_speakers=config.MAX_SPEAKERS,
            initial_prompt=config.INITIAL_PROMPT, compute_type=config.COMPUTE_TYPE)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"pod error: {str(result['error'])[:300]}")
        if not result.get("segments"):
            raise RuntimeError(f"pod returned no segments: {str(result)[:200]}")

        out_stem = os.path.join(config.OUTBOX_DIR, stem)
        files = render.write_outputs(result, out_stem, name)
    finally:
        if os.path.exists(flac):
            os.remove(flac)

    t_done = time.time()
    log(f"DONE '{name}': {result.get('num_speakers', '?')} speakers, "
        f"{len(result['segments'])} segments -> {[os.path.basename(f) for f in files]}")
    log(f"TIMING '{name}': detected {datetime.fromtimestamp(t_detect).strftime('%H:%M:%S')} "
        f"-> transcript {datetime.fromtimestamp(t_done).strftime('%H:%M:%S')} "
        f"= {int(t_done - t_detect)}s  (pod pipeline {result.get('timings', {}).get('total_s', '?')}s)")

    if config.DELETE_INPUT_AFTER:
        os.remove(path)
    else:
        shutil.move(path, os.path.join(config.DONE_DIR, name))


def fail_file(path, exc):
    name = os.path.basename(path)
    log(f"FAILED '{name}': {exc}")
    dest = os.path.join(config.FAILED_DIR, name)
    try:
        shutil.move(path, dest)
    except Exception:
        dest = path
    with open(dest + ".error.txt", "w", encoding="utf-8") as fh:
        fh.write(str(exc) + "\n")


def main():
    ensure_dirs()
    missing = config.validate()
    if missing:
        log(f"FATAL: missing config: {', '.join(missing)}")
        sys.exit(1)

    def _shutdown(*_):
        log("shutting down — terminating pod")
        pod_manager.terminate()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log(f"watching {config.INBOX_DIR} | pod={config.POD_NAME} dc={','.join(config.POD_DCS) or 'any'} "
        f"idle-down={config.POD_IDLE_SECONDS // 60}min | model={config.MODEL} "
        f"compute={config.COMPUTE_TYPE} lang={config.LANGUAGE or 'auto'}")
    last_activity = time.time()
    heartbeat = os.path.join(os.path.dirname(config.INBOX_DIR), ".heartbeat")
    while True:
        try:  # heartbeat so the watchdog knows the orchestrator is alive
            with open(heartbeat, "w") as fh:
                fh.write(str(int(time.time())))
        except Exception:
            pass
        queue = stable_files()
        if not queue:
            if pod_manager.is_up() and time.time() - last_activity > config.POD_IDLE_SECONDS:
                log(f"inbox idle >{config.POD_IDLE_SECONDS // 60}min — tearing pod down")
                pod_manager.terminate()
            time.sleep(config.SCAN_INTERVAL)
            continue

        path = queue[0]
        t_detect = time.time()
        log(f"DETECTED '{os.path.basename(path)}' at {datetime.fromtimestamp(t_detect).strftime('%H:%M:%S')}")
        try:
            pod_manager.ensure_up()  # boot the pod (transient failure -> retry, file stays)
        except Exception as exc:  # noqa: BLE001
            log(f"pod boot failed (will retry): {exc}")
            pod_manager.terminate()
            time.sleep(config.SCAN_INTERVAL)
            continue
        try:
            process(path, t_detect)
            last_activity = time.time()
        except KeyboardInterrupt:
            _shutdown()
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the loop
            fail_file(path, exc)
            last_activity = time.time()


if __name__ == "__main__":
    main()
