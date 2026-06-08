"""Cost watchdog — a separate compose service that guards against a forgotten pod.

The orchestrator already tears pods down when idle; this is the backstop for when
the orchestrator itself crashes or hangs. It kills any pod named POD_NAME if the
orchestrator's heartbeat goes stale (it stopped looping) or a pod outlives a hard
cap. No GPU, no cost — just an API poller.
"""
import datetime
import os
import time

import requests

import config

HEARTBEAT = os.path.join(os.path.dirname(config.INBOX_DIR), ".heartbeat")
CHECK = int(os.environ.get("WATCHDOG_INTERVAL", "60"))
STALE = int(os.environ.get("WATCHDOG_STALE", "600"))            # orch heartbeat age -> orphan
HARD_CAP_MIN = int(os.environ.get("WATCHDOG_MAX_POD_MIN", "60"))  # absolute pod-age backstop
API = "https://rest.runpod.io/v1"


def hdr():
    return {"Authorization": f"Bearer {config.RUNPOD_API_KEY}"}


def log(m):
    print(f"[watchdog] {time.strftime('%H:%M:%S')} {m}", flush=True)


def our_pods():
    r = requests.get(API + "/pods", headers=hdr(), timeout=30)
    r.raise_for_status()
    d = r.json()
    pods = d if isinstance(d, list) else d.get("data", [])
    return [p for p in pods if p.get("name") == config.POD_NAME]


def kill(pid, why):
    log(f"KILLING pod {pid}: {why}")
    try:
        requests.delete(f"{API}/pods/{pid}", headers=hdr(), timeout=30)
    except Exception as e:  # noqa: BLE001
        log(f"  kill error: {e}")


def pod_age_min(p):
    try:
        dt = datetime.datetime.strptime((p.get("createdAt") or "")[:19], "%Y-%m-%d %H:%M:%S")
        return (datetime.datetime.utcnow() - dt).total_seconds() / 60
    except Exception:
        return 0.0


def main():
    log(f"up | check={CHECK}s stale={STALE}s hard-cap={HARD_CAP_MIN}min pod={config.POD_NAME}")
    while True:
        try:
            hb_age = (time.time() - os.path.getmtime(HEARTBEAT)) if os.path.exists(HEARTBEAT) else 1e9
            pods = our_pods()
            for p in pods:
                pid = p["id"]
                age = pod_age_min(p)
                if hb_age > STALE:
                    kill(pid, f"orchestrator heartbeat stale {int(hb_age)}s (crashed/hung) — orphan")
                elif age > HARD_CAP_MIN:
                    kill(pid, f"pod alive {int(age)}min > hard cap {HARD_CAP_MIN}min")
            if pods and hb_age <= STALE:
                log(f"ok: {len(pods)} pod(s) up, orchestrator alive (hb {int(hb_age)}s, oldest {int(max(pod_age_min(p) for p in pods))}min)")
        except Exception as e:  # noqa: BLE001
            log(f"loop error: {e}")
        time.sleep(CHECK)


if __name__ == "__main__":
    main()
