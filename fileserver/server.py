#!/usr/bin/env python3
"""Upload + serve frontend for diarize-batch.

A drop-in replacement for the old `python -m http.server` one-liner. Same job
(serve transcripts from OUTBOX) plus a drag-and-drop upload box that writes
straight into INBOX, so you never need shell access to the box to add a meeting.

Routes
  GET  /                 -> HTML page: drop zone + list of transcripts
  GET  /<name>           -> serve a transcript from OUTBOX (read-only)
  PUT  /upload/<name>    -> stream the request body into INBOX, atomically:
                            writes INBOX/.<name>.part, then os.replace() to
                            INBOX/<name>. The orchestrator skips dotfiles, so it
                            never sees a half-uploaded file; the rename is the
                            "file is complete" signal and it gets picked up on
                            the next 5s scan.

Stdlib only — runs on the stock python:3.12-slim image with no build step.
"""
import html
import json
import os
import re
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

OUTBOX = os.environ.get("OUTBOX_DIR", "/data/outbox")
INBOX = os.environ.get("INBOX_DIR", "/data/inbox")
PORT = int(os.environ.get("PORT", "8080"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
# Optional shared secret. If set, PUT /upload requires ?token=... (or an
# X-Upload-Token header). Leave unset to keep the endpoint open (LAN use).
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "").strip()
# Accepted upload extensions — mirror the orchestrator's AUDIO_EXTS default.
AUDIO_EXTS = {
    e.strip().lower()
    for e in os.environ.get(
        "AUDIO_EXTS", ".mp4,.m4a,.wav,.flac,.mp3,.aac,.ogg,.webm,.opus,.mkv"
    ).split(",")
    if e.strip()
}

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>diarize-batch</title>
<style>
  :root { color-scheme: dark light; }
  body { font: 15px/1.5 system-ui, sans-serif; max-width: 720px; margin: 2rem auto;
         padding: 0 1rem; }
  h1 { font: 600 1.4rem ui-monospace, monospace; }
  h2 { font-size: 1rem; margin-top: 2rem; color: #888; text-transform: uppercase;
       letter-spacing: .05em; }
  #drop { border: 2px dashed #888; border-radius: 12px; padding: 2.2rem 1rem;
          text-align: center; color: #888; cursor: pointer; transition: .15s; }
  #drop.hot { border-color: #4a9; background: rgba(74,170,153,.08); color: #4a9; }
  #drop label { color: #4a9; text-decoration: underline; cursor: pointer; }
  #status div { font: 13px ui-monospace, monospace; padding: .25rem 0; }
  ul { list-style: none; padding: 0; }
  li { padding: .3rem 0; border-bottom: 1px solid #8884; }
  li a { text-decoration: none; }
  li a:hover { text-decoration: underline; }
  .fmt { font: 11px ui-monospace, monospace; color: #888; margin-left: .5rem; }
</style></head><body>
<h1>diarize-batch</h1>
<div id="drop">Drop audio here, or <label>browse<input id="file" type="file" multiple hidden></label></div>
<div id="status"></div>
<h2>Transcripts</h2>
<ul id="list">{{ROWS}}</ul>
<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const statusEl = document.getElementById('status');
const TOKEN = new URLSearchParams(location.search).get('token');

function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const line = document.createElement('div');
    line.textContent = file.name + ' — 0%';
    statusEl.appendChild(line);
    let url = '/upload/' + encodeURIComponent(file.name);
    if (TOKEN) url += '?token=' + encodeURIComponent(TOKEN);
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable)
        line.textContent = file.name + ' — ' + Math.round(e.loaded / e.total * 100) + '%';
    };
    xhr.onload = () => {
      if (xhr.status === 200) { line.textContent = file.name + ' — queued ✓'; resolve(); }
      else {
        let msg; try { msg = JSON.parse(xhr.responseText).error; } catch { msg = xhr.statusText; }
        line.textContent = file.name + ' — error: ' + msg; reject();
      }
    };
    xhr.onerror = () => { line.textContent = file.name + ' — network error'; reject(); };
    xhr.send(file);
  });
}

async function handleFiles(files) {
  for (const f of files) { try { await uploadFile(f); } catch (e) {} }
  setTimeout(() => location.reload(), 2000);
}

['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add('hot');
}));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove('hot');
}));
drop.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
drop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFiles(e.target.files));
</script></body></html>"""


def safe_name(raw):
    """Reduce a client-supplied name to a bare, safe filename or None."""
    name = os.path.basename(raw.replace("\\", "/")).strip()
    if not name or name.startswith("."):
        return None
    return name


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=OUTBOX, **kw)

    # --- serving ---------------------------------------------------------
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            return self._index()
        return super().do_GET()

    def _index(self):
        groups = {}
        try:
            entries = os.listdir(OUTBOX)
        except FileNotFoundError:
            entries = []
        for n in entries:
            if n.startswith(".") or not os.path.isfile(os.path.join(OUTBOX, n)):
                continue
            stem, ext = os.path.splitext(n)
            groups.setdefault(stem, {})[ext.lower()] = n

        rows = []
        for stem in sorted(groups, reverse=True):
            fmts = groups[stem]
            primary = fmts.get(".md") or sorted(fmts.values())[0]
            extras = " ".join(
                f'<a class="fmt" href="{urllib.parse.quote(fmts[e])}">{e[1:]}</a>'
                for e in (".json", ".srt", ".txt") if e in fmts
            )
            rows.append(
                f'<li><a href="{urllib.parse.quote(primary)}">{html.escape(stem)}</a>{extras}</li>'
            )
        body = PAGE.replace("{{ROWS}}", "\n".join(rows) or "<li><em>none yet</em></li>")
        self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")

    # --- uploading -------------------------------------------------------
    def do_PUT(self):
        m = re.match(r"^/upload/([^?]+)", self.path)
        if not m:
            return self._json(404, {"error": "not found"})
        if UPLOAD_TOKEN:
            q = urllib.parse.urlparse(self.path).query
            tok = urllib.parse.parse_qs(q).get("token", [""])[0] or \
                self.headers.get("X-Upload-Token", "")
            if tok != UPLOAD_TOKEN:
                return self._json(403, {"error": "bad or missing token"})

        name = safe_name(urllib.parse.unquote(m.group(1)))
        if not name:
            return self._json(400, {"error": "bad filename"})
        ext = os.path.splitext(name)[1].lower()
        if ext not in AUDIO_EXTS:
            return self._json(415, {"error": f"unsupported type '{ext}'"})

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return self._json(411, {"error": "Content-Length required"})
        if length > MAX_BYTES:
            return self._json(413, {"error": f"too large (max {MAX_BYTES // 1024 // 1024} MB)"})

        os.makedirs(INBOX, exist_ok=True)
        final = os.path.join(INBOX, name)
        if os.path.exists(final):
            return self._json(409, {"error": "a file with that name is already queued"})
        part = os.path.join(INBOX, f".{name}.part")

        remaining = length
        try:
            with open(part, "wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            if remaining != 0:
                raise IOError("connection closed before upload finished")
            os.replace(part, final)  # atomic -> orchestrator now sees it
        except Exception as exc:  # noqa: BLE001
            if os.path.exists(part):
                os.remove(part)
            return self._json(500, {"error": str(exc)})

        self.log_message("queued upload %s (%d bytes)", name, length)
        return self._json(200, {"ok": True, "name": name})

    # --- helpers ---------------------------------------------------------
    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def log_message(self, fmt, *args):
        print("[fileserver] " + (fmt % args), flush=True)


if __name__ == "__main__":
    print(f"[fileserver] serving {OUTBOX} + upload->{INBOX} on :{PORT}"
          f"{' (token required)' if UPLOAD_TOKEN else ''}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
