#!/usr/bin/env python3
"""Upload + serve frontend for diarize-batch.

A drop-in replacement for the old `python -m http.server` one-liner. Same job
(serve transcripts from OUTBOX) plus a drag-and-drop upload box that writes
straight into INBOX, so you never need shell access to the box to add a meeting.

Routes
  GET  /                 -> HTML page: drop zone + list of transcripts
  GET  /<name>           -> serve a transcript from OUTBOX (read-only)
  GET  /view/<stem>      -> interactive transcript viewer: speaker turns with
                            clickable timestamps/sentences that seek an inline
                            audio player (Fireflies-style), plus find-in-
                            transcript with match count and prev/next.
  GET  /audio/<stem>     -> the archived source audio from DONE_DIR (the
                            orchestrator moves processed inputs there), served
                            with HTTP Range support so the player can seek.
  GET  /names/<stem>     -> JSON {raw_speaker_label: name} of the HUMAN name map
                            for one meeting, read from the SQLite DB.
  POST /names/<stem>     -> upsert one mapping (body {"speaker_raw","name"}); an
                            empty/whitespace name deletes the row. This is the
                            additive override layer the viewer shows on top of
                            the auto voice-match — it never edits the transcript.
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
import sqlite3
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

OUTBOX = os.environ.get("OUTBOX_DIR", "/data/outbox")
INBOX = os.environ.get("INBOX_DIR", "/data/inbox")
DONE = os.environ.get("DONE_DIR", "/data/done")
# SQLite name map: one (meeting, speaker_raw) -> human name. Additive override
# layer the viewer renders on top of the auto voice-match; never touches the
# transcript. Shared with tools/names.py + tools/enroll.py over the data volume.
DB_DIR = os.environ.get("DB_DIR", "/data/db")
DB_PATH = os.environ.get("DB_PATH", os.path.join(DB_DIR, "speakers.db"))
PORT = int(os.environ.get("PORT", "8080"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024
# Optional shared secret. If set, PUT /upload requires ?token=... (or an
# X-Upload-Token header). Leave unset to keep the endpoint open (LAN use).
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "").strip()
# Accepted upload extensions — mirror the orchestrator's AUDIO_EXTS default.
# Use `or _DEFAULT_EXTS` so an *empty* env var still falls back to the default:
# compose passes AUDIO_EXTS="" when it isn't set in .env, and os.environ.get
# returns that "" (not the default), which would otherwise accept nothing.
_DEFAULT_EXTS = ".mp4,.m4a,.wav,.flac,.mp3,.aac,.ogg,.webm,.opus,.mkv"
AUDIO_EXTS = {
    e.strip().lower()
    for e in (os.environ.get("AUDIO_EXTS", "").strip() or _DEFAULT_EXTS).split(",")
    if e.strip()
}
AUDIO_CTYPES = {
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
    ".wav": "audio/wav", ".flac": "audio/flac", ".aac": "audio/aac",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".webm": "audio/webm",
    ".mkv": "video/x-matroska",
}


# --- name-map database ----------------------------------------------------
# Tiny SQLite store, schema shared with tools/names.py + tools/enroll.py.
# Keyed by the raw diarizer label (SPEAKER_07) — stable across re-renders,
# unlike the cosmetic "Speaker 8". WAL mode so a UI write and an ssh/CLI read
# don't block each other.
SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_names (
    meeting      TEXT NOT NULL,
    speaker_raw  TEXT NOT NULL,
    name         TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (meeting, speaker_raw)
);
"""


def _db():
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def db_init():
    os.makedirs(DB_DIR, exist_ok=True)
    with _db() as conn:
        conn.executescript(SCHEMA)


def names_for(meeting):
    """{speaker_raw: name} for one meeting."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT speaker_raw, name FROM speaker_names WHERE meeting=?",
            (meeting,)).fetchall()
    return {r[0]: r[1] for r in rows}


def set_name(meeting, speaker_raw, name):
    """Upsert, or delete the row when name is blank. Returns the new name ('' if
    cleared)."""
    name = (name or "").strip()
    with _db() as conn:
        if name:
            conn.execute(
                "INSERT INTO speaker_names (meeting, speaker_raw, name, updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(meeting, speaker_raw) "
                "DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
                (meeting, speaker_raw, name,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        else:
            conn.execute(
                "DELETE FROM speaker_names WHERE meeting=? AND speaker_raw=?",
                (meeting, speaker_raw))
        conn.commit()
    return name

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

# Transcript viewer. Server-side it only needs the stem; everything else is
# fetched client-side from the existing static routes (/<stem>.json and the
# additive /<stem>.speakers.json) plus /audio/<stem> for playback.
VIEWER = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>transcript — diarize-batch</title>
<style>
  :root { color-scheme: dark light; }
  body { font: 15px/1.6 system-ui, sans-serif; max-width: 760px; margin: 0 auto;
         padding: 0 1rem 4rem; }
  #bar { position: sticky; top: 0; background: Canvas; padding: .7rem 0 .6rem;
         border-bottom: 1px solid #8884; z-index: 2; }
  #bar .head { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; }
  #bar .head .back { color: #4a9; text-decoration: none; font-size: 13px; }
  #title { font: 600 1.05rem ui-monospace, monospace; margin: 0; }
  #meta { color: #888; font-size: 12.5px; }
  #player { width: 100%; height: 36px; margin-top: .55rem; }
  .note { color: #c77; font: 13px ui-monospace, monospace; margin-top: .55rem; }
  #find { display: flex; align-items: center; gap: .35rem; margin-top: .55rem; }
  #q { flex: 1; min-width: 0; font: inherit; padding: .3rem .6rem;
       border: 1px solid #8886; border-radius: 8px; background: transparent; }
  #count { font: 12px ui-monospace, monospace; color: #888; min-width: 3.5em;
           text-align: right; }
  #find button { font: 12px ui-monospace, monospace; background: transparent;
                 border: 1px solid #8886; border-radius: 6px; color: inherit;
                 cursor: pointer; padding: .25rem .55rem; }
  #find button:hover { border-color: #4a9; color: #4a9; }
  .turn { margin: 1.1rem 0; }
  .turn .who { display: flex; align-items: center; gap: .5rem; margin-bottom: .15rem; }
  .chip { width: 22px; height: 22px; border-radius: 6px; color: #fff; flex: none;
          font: 600 12px/22px system-ui; text-align: center; }
  .who b { font-size: 13.5px; }
  .who b.named { color: #2a8; }
  .edit { font-size: 12px; color: #888; cursor: pointer; border: 0;
          background: none; padding: 0 .15rem; opacity: .55; }
  .edit:hover { opacity: 1; color: #4a9; }
  .who input { font: inherit; font-size: 13.5px; padding: .05rem .35rem;
               border: 1px solid #4a9; border-radius: 6px; background: transparent;
               color: inherit; width: 12rem; max-width: 50vw; }
  .who .save, .who .cancel { font: 12px ui-monospace, monospace; cursor: pointer;
        border: 1px solid #8886; border-radius: 6px; background: transparent;
        color: inherit; padding: .15rem .45rem; }
  .who .save:hover { border-color: #4a9; color: #4a9; }
  .ts { font: 12px ui-monospace, monospace; color: #4a9; cursor: pointer;
        text-decoration: underline; }
  .turn p { margin: 0 0 0 30px; }
  .seg { cursor: pointer; border-radius: 4px; }
  .seg:hover { background: rgba(74,170,153,.13); }
  .seg.playing { background: rgba(74,170,153,.25); }
  .hide { display: none; }
  mark { background: #ffd23f; color: #000; border-radius: 2px; }
  mark.cur { background: #ff9914; }
  #state { color: #888; margin-top: 2rem; font: 13px ui-monospace, monospace; }
</style></head><body>
<div id="bar">
  <div class="head"><a class="back" href="/">&larr; transcripts</a>
    <h1 id="title">&hellip;</h1><span id="meta"></span></div>
  <div id="audiowrap"><audio id="player" controls preload="metadata"></audio></div>
  <div id="find">
    <input id="q" type="search" placeholder="Search transcript&hellip;" autocomplete="off">
    <span id="count"></span>
    <button id="prev" title="previous match (Shift+Enter)">&#9650;</button>
    <button id="next" title="next match (Enter)">&#9660;</button>
  </div>
</div>
<div id="state">loading&hellip;</div>
<div id="list"></div>
<script>
const STEM = {{STEM}};
const audio = document.getElementById('player');
const list = document.getElementById('list');
const state = document.getElementById('state');
const q = document.getElementById('q');
const countEl = document.getElementById('count');
const PALETTE = ['#3aa087', '#5b8def', '#c98a4b', '#a06ee0',
                 '#7da33c', '#d4647c', '#4aa0b5', '#b08f3e'];
let segs = [], marks = [], cur = -1, canPlay = true, lastSeg = -1;
// name layers, both keyed by raw diarizer label (SPEAKER_07):
//   auto   = voice-match from <stem>.speakers.json (read-only)
//   manual = human edits stored in the SQLite name map (editable here)
// display precedence: manual > auto > "Speaker N".
let auto = {}, manual = {}, colors = {};

const esc = s => s.replace(/[&<>"]/g,
  c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const fmtTs = t => {
  t = Math.max(0, Math.floor(t));
  const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60;
  const mm = String(m).padStart(2, '0'), ss = String(s).padStart(2, '0');
  return h ? h + ':' + mm + ':' + ss : mm + ':' + ss;
};
const label = raw => {
  if (!raw || raw === 'UNKNOWN') return 'Unknown';
  const m = /^SPEAKER_(\\d+)$/.exec(raw);
  return m ? 'Speaker ' + (Number(m[1]) + 1) : raw;
};

audio.addEventListener('error', () => {
  canPlay = false;
  document.getElementById('audiowrap').innerHTML =
    '<div class="note">source audio is not on the server &mdash; click-to-play disabled</div>';
});
audio.src = '/audio/' + encodeURIComponent(STEM);

// what to show for a raw speaker label, and whether it's a real (human/auto) name
const displayName = raw => manual[raw] || auto[raw] || label(raw);
const isNamed = raw => Boolean(manual[raw] || auto[raw]);

function whoInner(raw, t) {
  const name = displayName(raw);
  return '<span class="chip" style="background:' + (colors[raw] || '#888') + '">' +
    esc(name[0].toUpperCase()) + '</span>' +
    '<b class="name' + (isNamed(raw) ? ' named' : '') + '">' + esc(name) + '</b>' +
    '<button class="edit" title="rename speaker">&#9998;</button>' +
    (t != null ? '<span class="ts" data-t="' + t + '">' + fmtTs(t) + '</span>' : '');
}

function render() {
  let ci = 0, html = '', lastSpk = null;
  segs.forEach((s, i) => {
    const spk = s.speaker || 'UNKNOWN';
    if (spk !== lastSpk) {
      if (lastSpk !== null) html += '</p></div>';
      if (!(spk in colors)) colors[spk] = PALETTE[ci++ % PALETTE.length];
      html += '<div class="turn" data-spk="' + esc(spk) + '" data-t="' + s.start +
        '"><div class="who">' + whoInner(spk, s.start) + '</div><p>';
      lastSpk = spk;
    }
    html += '<span class="seg" data-i="' + i + '">' + esc(s.text.trim()) + '</span> ';
  });
  if (lastSpk !== null) html += '</p></div>';
  list.innerHTML = html;
}

// repaint every turn header for one speaker after its name changes
function refreshSpeaker(raw) {
  list.querySelectorAll('.turn[data-spk="' + CSS.escape(raw) + '"]').forEach(turn => {
    turn.querySelector('.who').innerHTML = whoInner(raw, Number(turn.dataset.t));
  });
}

async function load() {
  const resp = await fetch('/' + encodeURIComponent(STEM) + '.json');
  if (!resp.ok) { state.textContent = 'transcript JSON not found'; return; }
  const data = await resp.json();
  document.getElementById('title').textContent = data.title || STEM;
  document.title = (data.title || STEM) + ' — diarize-batch';
  document.getElementById('meta').textContent =
    [data.date, data.time, data.duration,
     data.speakers && data.speakers + ' speakers'].filter(Boolean).join(' · ');
  try {  // auto layer: voice-match side file -> name by raw label where matched
    const sr = await fetch('/' + encodeURIComponent(STEM) + '.speakers.json');
    if (sr.ok)
      for (const info of Object.values((await sr.json()).speakers || {}))
        if (info.matched && info.raw) auto[info.raw] = info.name;
  } catch (e) {}
  try {  // manual layer: human name map from the SQLite DB (keyed by raw)
    const nr = await fetch('/names/' + encodeURIComponent(STEM));
    if (nr.ok) manual = (await nr.json()).names || {};
  } catch (e) {}
  segs = (data.segments || []).filter(s => (s.text || '').trim());
  if (!segs.length) { state.textContent = 'empty transcript'; return; }
  render();
  state.remove();
}

function seek(t) {
  if (!canPlay) return;
  audio.currentTime = t;
  audio.play();
}
list.addEventListener('click', e => {
  if (e.target.closest('.edit')) { startEdit(e.target.closest('.turn')); return; }
  if (e.target.closest('.save')) { commitEdit(e.target.closest('.turn')); return; }
  if (e.target.closest('.cancel')) {
    refreshSpeaker(e.target.closest('.turn').dataset.spk); return;
  }
  if (e.target.closest('.who')) return;  // don't seek when fiddling with the header
  const seg = e.target.closest('.seg'), ts = e.target.closest('.ts');
  if (seg) seek(segs[Number(seg.dataset.i)].start);
  else if (ts) seek(Number(ts.dataset.t));
});

// inline rename: swap this speaker's header for an input; saving writes the
// SQLite name map via POST /names/<stem> and repaints every turn for the speaker.
function startEdit(turn) {
  const raw = turn.dataset.spk, who = turn.querySelector('.who');
  who.innerHTML =
    '<span class="chip" style="background:' + (colors[raw] || '#888') + '"></span>' +
    '<input type="text" maxlength="80"> ' +
    '<button class="save">save</button> <button class="cancel">cancel</button>';
  const inp = who.querySelector('input');
  inp.value = manual[raw] || '';
  inp.placeholder = auto[raw] || label(raw);
  inp.focus();
  inp.select();
  inp.addEventListener('keydown', ev => {
    if (ev.key === 'Enter') { ev.preventDefault(); commitEdit(turn); }
    else if (ev.key === 'Escape') { ev.preventDefault(); refreshSpeaker(raw); }
  });
}

async function commitEdit(turn) {
  const raw = turn.dataset.spk, inp = turn.querySelector('.who input');
  if (!inp) return;
  const name = inp.value.trim();
  inp.disabled = true;
  try {
    const r = await fetch('/names/' + encodeURIComponent(STEM), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speaker_raw: raw, name }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    if (data.name) manual[raw] = data.name; else delete manual[raw];
  } catch (err) {
    alert('Could not save name: ' + err);
  }
  refreshSpeaker(raw);  // repaint from the (updated) manual map
}

audio.addEventListener('timeupdate', () => {
  const t = audio.currentTime;
  let idx = -1;
  for (let i = 0; i < segs.length; i++) {
    if (segs[i].start > t) break;
    if (t < segs[i].end + 0.3) { idx = i; break; }
  }
  if (idx === lastSeg) return;
  const old = list.querySelector('.seg.playing');
  if (old) old.classList.remove('playing');
  if (idx >= 0) {
    const el = list.querySelector('.seg[data-i="' + idx + '"]');
    if (el) el.classList.add('playing');
  }
  lastSeg = idx;
});

function markText(text, term) {
  const lower = text.toLowerCase();
  let out = '', pos = 0;
  for (let i = lower.indexOf(term); i !== -1; i = lower.indexOf(term, pos)) {
    out += esc(text.slice(pos, i)) +
      '<mark>' + esc(text.slice(i, i + term.length)) + '</mark>';
    pos = i + term.length;
  }
  return out + esc(text.slice(pos));
}
function setCount() {
  countEl.textContent = q.value.trim()
    ? (marks.length ? (cur + 1) + '/' + marks.length : '0/0') : '';
}
function activate() {
  marks.forEach((m, i) => m.classList.toggle('cur', i === cur));
  marks[cur].scrollIntoView({block: 'center'});
}
function step(d) {
  if (!marks.length) return;
  cur = (cur + d + marks.length) % marks.length;
  activate();
  setCount();
}
function search() {
  const term = q.value.trim().toLowerCase();
  marks = []; cur = -1;
  list.querySelectorAll('.turn').forEach(turn => {
    let hit = false;
    turn.querySelectorAll('.seg').forEach(el => {
      const text = segs[Number(el.dataset.i)].text.trim();
      if (term && text.toLowerCase().includes(term)) {
        hit = true;
        el.innerHTML = markText(text, term);
      } else el.textContent = text;
    });
    turn.classList.toggle('hide', Boolean(term) && !hit);
  });
  if (term) {
    marks = Array.from(list.querySelectorAll('mark'));
    if (marks.length) { cur = 0; activate(); }
  }
  setCount();
}
q.addEventListener('input', search);
q.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); step(e.shiftKey ? -1 : 1); }
  else if (e.key === 'Escape') { q.value = ''; search(); }
});
document.getElementById('prev').addEventListener('click', () => step(-1));
document.getElementById('next').addEventListener('click', () => step(1));

load().catch(err => { state.textContent = 'failed to load: ' + err; });
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
        path = urllib.parse.urlparse(self.path).path
        m = re.match(r"^/view/([^/]+)$", path)
        if m:
            return self._viewer(urllib.parse.unquote(m.group(1)))
        m = re.match(r"^/audio/([^/]+)$", path)
        if m:
            return self._audio(urllib.parse.unquote(m.group(1)))
        m = re.match(r"^/names/([^/]+)$", path)
        if m:
            return self._names_get(urllib.parse.unquote(m.group(1)))
        return super().do_GET()

    def do_HEAD(self):
        m = re.match(r"^/audio/([^/]+)$", urllib.parse.urlparse(self.path).path)
        if m:
            return self._audio(urllib.parse.unquote(m.group(1)), head=True)
        return super().do_HEAD()

    def do_POST(self):
        m = re.match(r"^/names/([^/]+)$", urllib.parse.urlparse(self.path).path)
        if m:
            return self._names_post(urllib.parse.unquote(m.group(1)))
        return self._json(404, {"error": "not found"})

    def _viewer(self, raw):
        stem = safe_name(raw)
        if not stem or not os.path.isfile(os.path.join(OUTBOX, stem + ".json")):
            return self._json(404, {"error": "no transcript JSON for that name"})
        # json.dumps -> a valid JS string literal; escape '<' so a hostile
        # filename can't close the <script> tag.
        body = VIEWER.replace("{{STEM}}", json.dumps(stem).replace("<", "\\u003c"))
        self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")

    # --- name map --------------------------------------------------------
    def _names_get(self, raw):
        stem = safe_name(raw)
        if not stem:
            return self._json(400, {"error": "bad name"})
        try:
            return self._json(200, {"names": names_for(stem)})
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": str(exc)})

    def _names_post(self, raw):
        stem = safe_name(raw)
        if not stem:
            return self._json(400, {"error": "bad name"})
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1 << 16:
            return self._json(400, {"error": "missing or oversized body"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            speaker = str(payload["speaker_raw"]).strip()
            if not speaker:
                raise ValueError("speaker_raw required")
        except Exception as exc:  # noqa: BLE001
            return self._json(400, {"error": f"bad request: {exc}"})
        try:
            name = set_name(stem, speaker, payload.get("name", ""))
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": str(exc)})
        self.log_message("name %s/%s -> %r", stem, speaker, name)
        return self._json(200, {"ok": True, "speaker_raw": speaker, "name": name})

    def _audio(self, raw, head=False):
        """Serve DONE_DIR/<stem>.<any audio ext> with HTTP Range support, so the
        viewer's <audio> element can seek straight to a clicked sentence."""
        stem = safe_name(raw)
        path = None
        try:
            names = sorted(os.listdir(DONE))
        except OSError:
            names = []
        if stem:
            for n in names:
                s, ext = os.path.splitext(n)
                if s == stem and ext.lower() in AUDIO_EXTS \
                        and os.path.isfile(os.path.join(DONE, n)):
                    path = os.path.join(DONE, n)
                    break
        if not path:
            return self._json(404, {"error": "no archived audio for that name"})

        size = os.path.getsize(path)
        ctype = AUDIO_CTYPES.get(os.path.splitext(path)[1].lower(),
                                 "application/octet-stream")
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:  # suffix form: bytes=-N (the last N bytes)
                start = max(0, size - int(m.group(2)))
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head:
            return
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = fh.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # player seeked/closed mid-stream — routine, not an error

    def _index(self):
        groups = {}
        try:
            entries = os.listdir(OUTBOX)
        except FileNotFoundError:
            entries = []
        for n in entries:
            if n.startswith(".") or not os.path.isfile(os.path.join(OUTBOX, n)):
                continue
            if n.endswith(".speakers.json"):  # additive side file, not a format
                continue
            stem, ext = os.path.splitext(n)
            groups.setdefault(stem, {})[ext.lower()] = n

        rows = []
        for stem in sorted(groups, reverse=True):
            fmts = groups[stem]
            if ".json" in fmts:  # JSON powers the interactive viewer
                primary = f"/view/{urllib.parse.quote(stem)}"
                raw_exts = (".md", ".json", ".srt", ".txt")
            else:
                raw = fmts.get(".md") or sorted(fmts.values())[0]
                primary = urllib.parse.quote(raw)
                raw_exts = tuple(e for e in (".md", ".srt", ".txt") if fmts.get(e) != raw)
            extras = " ".join(
                f'<a class="fmt" href="{urllib.parse.quote(fmts[e])}">{e[1:]}</a>'
                for e in raw_exts if e in fmts
            )
            rows.append(
                f'<li><a href="{primary}">{html.escape(stem)}</a>{extras}</li>'
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
    db_init()
    print(f"[fileserver] serving {OUTBOX} + upload->{INBOX} on :{PORT}"
          f" | name-db {DB_PATH}"
          f"{' (token required)' if UPLOAD_TOKEN else ''}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
