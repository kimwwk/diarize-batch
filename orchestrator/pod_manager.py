"""RunPod pod lifecycle + SSH tunnel for the orchestrator.

The pod runs a FastAPI server bound to its localhost; we reach it ONLY through an
SSH local-forward (ssh -N -L LOCAL_PORT:localhost:8000), so the transcription API
is never public. On-demand: ensure_up() creates+boots the pod when work arrives;
terminate() deletes it when idle so it stops billing.
"""
import os
import subprocess
import time

import requests

import config

API = "https://rest.runpod.io/v1"
_pod_id = None
_tunnel = None  # subprocess.Popen of the `ssh -N -L` tunnel


def log(m):
    print(f"[pod] {m}", flush=True)


def _hdr():
    return {"Authorization": f"Bearer {config.RUNPOD_API_KEY}", "Content-Type": "application/json"}


def _rest(method, path, body=None):
    r = requests.request(method, API + path, headers=_hdr(), json=body, timeout=90)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:300]}")
    return r.json() if r.text.strip() else {}


def _find_running():
    data = _rest("GET", "/pods")
    pods = data if isinstance(data, list) else data.get("data", [])
    for p in pods:
        if p.get("name") == config.POD_NAME and p.get("desiredStatus") == "RUNNING":
            return p["id"]
    return None


def _create():
    pub = open(config.SSH_PUBKEY_PATH).read().strip()
    body = {
        "name": config.POD_NAME, "imageName": config.POD_IMAGE,
        "gpuTypeIds": config.POD_GPU_IDS, "gpuCount": 1,
        "ports": ["22/tcp"], "containerDiskInGb": config.POD_DISK_GB,
        "env": {"PUBLIC_KEY": pub, "HF_TOKEN": config.HF_TOKEN},
    }
    if config.POD_DCS:
        body["dataCenterIds"] = config.POD_DCS
    if config.POD_VOLUME_ID:
        body["networkVolumeId"] = config.POD_VOLUME_ID
        body["volumeMountPath"] = "/runpod-volume"
    d = _rest("POST", "/pods", body)
    pid = d.get("id")
    if not pid:
        raise RuntimeError(f"pod create failed: {d}")
    log(f"created pod {pid} (gpu pool {len(config.POD_GPU_IDS)} types, dc {','.join(config.POD_DCS) or 'any'})")
    return pid


def _ssh_endpoint(pid):
    d = _rest("GET", f"/pods/{pid}")
    ip = d.get("publicIp") or ""
    port = ""
    pm = d.get("portMappings")
    if isinstance(pm, dict):
        port = str(pm.get("22") or "")
    for p in (d.get("runtime") or {}).get("ports") or []:
        if str(p.get("privatePort")) == "22":
            ip = ip or p.get("ip") or ""
            port = port or str(p.get("publicPort") or "")
    return (ip, port) if ip and port else (None, None)


def _ssh_opts():
    return ["-i", config.SSH_KEY_PATH, "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]


def _healthy():
    try:
        r = requests.get(f"http://localhost:{config.LOCAL_PORT}/health", timeout=5)
        return r.ok and r.json().get("ready") is True
    except Exception:
        return False


def is_up():
    return _pod_id is not None


def ensure_up():
    """Make sure a pod is running, booted, and reachable on localhost:LOCAL_PORT."""
    global _pod_id, _tunnel
    if _pod_id and _tunnel and _tunnel.poll() is None and _healthy():
        return
    _pod_id = _find_running() or _create()

    # wait for the SSH port mapping to appear
    ip = port = None
    deadline = time.time() + config.POD_BOOT_TIMEOUT
    while time.time() < deadline:
        ip, port = _ssh_endpoint(_pod_id)
        if ip:
            break
        log("waiting for ssh endpoint...")
        time.sleep(15)
    if not ip:
        raise RuntimeError("pod ssh endpoint never appeared")
    log(f"ssh {ip}:{port}")

    # wait for sshd to accept the key
    for _ in range(18):
        out = subprocess.run(["ssh", *_ssh_opts(), "-p", port, f"root@{ip}", "echo ok"],
                             capture_output=True, text=True).stdout.strip()
        if out == "ok":
            break
        time.sleep(10)

    # open the tunnel
    if _tunnel:
        _tunnel.terminate()
    _tunnel = subprocess.Popen(
        ["ssh", *_ssh_opts(), "-p", port, "-N",
         "-L", f"{config.LOCAL_PORT}:localhost:8000", f"root@{ip}"])
    time.sleep(3)

    # wait for /health ready (server warms models from the volume)
    while time.time() < deadline:
        if _healthy():
            log("pod healthy — models loaded")
            return
        log("waiting for /health...")
        time.sleep(15)
    raise RuntimeError("pod /health never became ready")


def infer(flac_path, language=None, min_speakers=None, max_speakers=None,
          initial_prompt=None, compute_type="float16"):
    """POST the file (multipart) through the tunnel; return the result dict."""
    data = {"language": language or "", "compute_type": compute_type}
    if min_speakers is not None:
        data["min_speakers"] = str(min_speakers)
    if max_speakers is not None:
        data["max_speakers"] = str(max_speakers)
    if initial_prompt:
        data["initial_prompt"] = initial_prompt
    with open(flac_path, "rb") as fh:
        r = requests.post(
            f"http://localhost:{config.LOCAL_PORT}/inference",
            files={"file": (os.path.basename(flac_path), fh)},
            data=data, timeout=config.JOB_TIMEOUT)
    r.raise_for_status()
    return r.json()


def terminate():
    """Close the tunnel and delete the pod (stop billing)."""
    global _pod_id, _tunnel
    if _tunnel:
        try:
            _tunnel.terminate()
        except Exception:
            pass
        _tunnel = None
    if _pod_id:
        try:
            _rest("DELETE", f"/pods/{_pod_id}")
            log(f"terminated pod {_pod_id} (idle)")
        except Exception as e:  # noqa: BLE001
            log(f"terminate warning: {e}")
        _pod_id = None
