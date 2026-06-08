#!/usr/bin/env bash
# Pod entrypoint: start key-gated sshd (public, port 22) + the FastAPI server
# (private, 127.0.0.1 only). The orchestrator reaches the API via `ssh -L`.
set -e

# 1) SSH — inject the orchestrator's public key, then start sshd.
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "$PUBLIC_KEY" ]; then
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  echo "[entrypoint] injected PUBLIC_KEY into authorized_keys"
fi
ssh-keygen -A 2>/dev/null || true   # generate host keys if missing
mkdir -p /run/sshd
/usr/sbin/sshd
echo "[entrypoint] sshd started on :22 (key-gated, AllowTcpForwarding=yes)"

# 2) FastAPI transcription server — localhost only, reached via the SSH tunnel.
echo "[entrypoint] starting transcription server on 127.0.0.1:${PORT:-8000}"
exec python /app/server.py
