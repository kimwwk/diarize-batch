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
  GET  /roster/<stem>    -> JSON list of every speaker in a meeting with its
                            resolved display name (manual > voice-match > "Speaker
                            N"), source, and talk-time. The "see all speakers"
                            API; also what the viewer's mapping panel shows.
  GET  /note/<stem>      -> JSON {"exists","markdown","html"} for the editable,
                            hand-authored note attached to a transcript by stem
                            (NOTES/<stem>.note.md). Always 200 — no note just
                            means exists=false, so the viewer shows an empty
                            (still editable) panel.
  POST /note/<stem>      -> save the note (body {"markdown":"..."}); a blank body
                            clears it. Writes to the read-write NOTES dir; OUTBOX
                            stays read-only. Requires the transcript to exist.
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
# Hand-authored, editable notes — one per transcript, by stem
# (NOTES/<stem>.note.md). Unlike OUTBOX (immutable pipeline output, mounted
# read-only), NOTES is mounted read-WRITE so the viewer's "Save" can write here.
# Absence is normal and never an error.
NOTES = os.environ.get("NOTES_DIR", "/data/notes")
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


def _label(raw):
    """Cosmetic 'Speaker N' from a raw 'SPEAKER_xx' label (mirror render.py)."""
    if not raw or raw == "UNKNOWN":
        return "Unknown"
    m = re.fullmatch(r"SPEAKER_(\d+)", raw)
    return f"Speaker {int(m.group(1)) + 1}" if m else raw


def roster_for(stem):
    """Every speaker in a meeting with its resolved display name. Merges the
    three layers the viewer uses — manual DB name > auto voice-match > Speaker N
    — plus talk-time, so one call answers 'who is in this meeting and what are
    they called'. Returns None if the transcript JSON is missing."""
    jpath = os.path.join(OUTBOX, stem + ".json")
    if not os.path.isfile(jpath):
        return None
    with open(jpath, encoding="utf-8") as fh:
        data = json.load(fh)
    order, secs, nseg = [], {}, {}
    for s in data.get("segments", []):
        spk = s.get("speaker") or "UNKNOWN"
        if spk not in secs:
            order.append(spk)
            secs[spk] = 0.0
            nseg[spk] = 0
        secs[spk] += max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
        nseg[spk] += 1

    auto = {}
    spath = os.path.join(OUTBOX, stem + ".speakers.json")
    if os.path.isfile(spath):
        try:
            with open(spath, encoding="utf-8") as fh:
                for info in (json.load(fh).get("speakers") or {}).values():
                    if info.get("matched") and info.get("raw"):
                        auto[info["raw"]] = info["name"]
        except Exception:  # noqa: BLE001
            pass

    manual = names_for(stem)
    roster = []
    for spk in order:
        name = manual.get(spk) or auto.get(spk) or _label(spk)
        source = "manual" if spk in manual else ("voice" if spk in auto else "none")
        roster.append({
            "raw": spk, "label": _label(spk), "name": name, "source": source,
            "seconds": round(secs[spk], 1), "segments": nseg[spk],
        })
    roster.sort(key=lambda r: -r["seconds"])
    return roster


# --- notes ----------------------------------------------------------------
# A note is a hand-authored Markdown file attached to one transcript by stem and
# editable from the viewer: NOTES/<stem>.note.md. NOTES is mounted read-WRITE
# (OUTBOX is read-only), so the viewer's "Save" writes here. The file on disk is
# the source of truth, kept verbatim; we only render a copy to HTML for display.
# Absence is normal — note_html() returns None and the viewer shows an empty
# (but still editable) panel.
def note_path(stem):
    return os.path.join(NOTES, stem + ".note.md")


def _md_inline(text):
    """Inline Markdown -> HTML on an already HTML-escaped string. Order matters:
    code spans first (so their contents aren't touched), then links, then
    bold/italic. Stdlib only — no external markdown dependency."""
    # `code` spans -> <code>…</code> (contents left as-is, already escaped)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # [label](url) -> anchor; only http/https/relative, no javascript: etc.
    def _link(m):
        label, url = m.group(1), m.group(2).strip()
        if not re.match(r"^(https?:|/|\.|#|mailto:)", url, re.I):
            return m.group(0)
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", _link, text)
    # **bold** then *italic* / _italic_
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<em>\1</em>", text)
    return text


def md_to_html(text):
    """Minimal, safe Markdown -> HTML. Everything is HTML-escaped first, so the
    output can't inject markup; we then re-introduce only a known set of tags
    (headings, lists, blockquotes, code, paragraphs, inline emphasis/links).
    Good enough for hand-written notes; keeps the raw .md verbatim."""
    out, para, list_stack, in_code = [], [], [], False
    code_buf = []

    def flush_para():
        if para:
            out.append("<p>" + _md_inline("<br>".join(para)) + "</p>")
            para.clear()

    def close_lists(to_depth=0):
        while len(list_stack) > to_depth:
            out.append("</" + list_stack.pop() + ">")

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = html.escape(raw_line)

        # fenced code blocks ```
        if raw_line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + "\n".join(code_buf) + "</code></pre>")
                code_buf, in_code = [], False
            else:
                flush_para(); close_lists()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue

        stripped = line.strip()
        if not stripped:                         # blank line ends a block
            flush_para(); close_lists()
            continue

        h = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if h:
            flush_para(); close_lists()
            lvl = len(h.group(1))
            out.append(f"<h{lvl}>" + _md_inline(h.group(2).strip()) + f"</h{lvl}>")
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):  # horizontal rule
            flush_para(); close_lists()
            out.append("<hr>")
            continue

        bq = re.match(r"^&gt;\s?(.*)$", stripped)   # '>' was escaped to &gt;
        if bq:
            flush_para(); close_lists()
            out.append("<blockquote><p>" + _md_inline(bq.group(1)) + "</p></blockquote>")
            continue

        # list items — depth by leading spaces (2 spaces per level)
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", raw_line)
        if m:
            flush_para()
            depth = len(m.group(1)) // 2 + 1
            tag = "ol" if m.group(2)[:1].isdigit() else "ul"
            while len(list_stack) > depth:
                out.append("</" + list_stack.pop() + ">")
            while len(list_stack) < depth:
                out.append("<" + tag + ">")
                list_stack.append(tag)
            out.append("<li>" + _md_inline(html.escape(m.group(3))) + "</li>")
            continue

        close_lists()
        para.append(stripped)

    if in_code:                                   # unterminated fence
        out.append("<pre><code>" + "\n".join(code_buf) + "</code></pre>")
    flush_para(); close_lists()
    return "\n".join(out)


def note_markdown(stem):
    """Raw Markdown for NOTES/<stem>.note.md, or "" if there is none."""
    try:
        with open(note_path(stem), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def note_html(stem):
    """Rendered HTML for the note, or None when there is no note for this stem
    (the common case — handled gracefully)."""
    md = note_markdown(stem)
    if not md.strip():
        return None
    try:
        return md_to_html(md)
    except Exception:  # noqa: BLE001 — a bad note must never break the viewer
        return None


def save_note(stem, markdown):
    """Write NOTES/<stem>.note.md atomically, or delete it when blank. Returns
    the rendered HTML (None when cleared). NOTES is the only read-write data dir
    the fileserver touches for content, so this can never clobber a transcript."""
    path = note_path(stem)
    if not markdown.strip():
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    os.makedirs(NOTES, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    os.replace(tmp, path)
    return note_html(stem)


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
  /* Notes panel: an editable, hand-authored note per transcript, beside the
     turns. Always available now (you can add one anytime); empty is fine. */
  body.has-notes { max-width: 1180px; }
  body.has-notes #cols { display: flex; gap: 1.6rem; align-items: flex-start; }
  body.has-notes #main { flex: 1 1 0; min-width: 0; }
  #notepanel { display: none; }
  body.has-notes #notepanel { display: block; flex: 0 0 360px;
         max-width: 42%; position: sticky; top: 4.2rem; align-self: flex-start;
         max-height: calc(100vh - 5rem); overflow: auto;
         border: 1px solid #8884; border-radius: 12px; padding: .2rem 1.1rem 1rem;
         background: rgba(127,127,127,.05); }
  #notepanel .nhead { display: flex; align-items: center; justify-content: space-between;
         gap: .5rem; position: sticky; top: 0; background: Canvas;
         padding: .4rem 0; margin: .6rem 0 .35rem; }
  #notepanel h2.ntitle { font: 600 .8rem ui-monospace, monospace; color: #888;
         text-transform: uppercase; letter-spacing: .05em; margin: 0; }
  .nbtn { font: 12px ui-monospace, monospace; background: transparent;
         border: 1px solid #8886; border-radius: 6px; color: inherit;
         cursor: pointer; padding: .15rem .55rem; }
  .nbtn:hover { border-color: #4a9; color: #4a9; }
  .nempty { color: #888; font-style: italic; }
  #notepanel .nbody { font-size: 14px; }
  #notepanel .nbody h1 { font-size: 1.15rem; }
  #notepanel .nbody h2 { font-size: 1rem; text-transform: none; letter-spacing: 0;
         color: inherit; margin: 1.1rem 0 .3rem; }
  #notepanel .nbody h3 { font-size: .92rem; margin: .9rem 0 .25rem; }
  #notepanel .nbody blockquote { margin: .6rem 0; padding: .1rem 0 .1rem .8rem;
         border-left: 3px solid #8886; color: #999; }
  #notepanel .nbody code { font: 12.5px ui-monospace, monospace;
         background: rgba(127,127,127,.18); padding: .05rem .3rem; border-radius: 4px; }
  #notepanel .nbody pre { background: rgba(127,127,127,.12); padding: .6rem .8rem;
         border-radius: 8px; overflow: auto; }
  #notepanel .nbody pre code { background: none; padding: 0; }
  #notepanel .nbody a { color: #4a9; }
  #notepanel .nbody li { margin: .15rem 0; }
  #notepanel textarea { width: 100%; box-sizing: border-box; min-height: 48vh;
         font: 13px/1.5 ui-monospace, monospace; padding: .6rem .7rem;
         border: 1px solid #8886; border-radius: 8px; background: transparent;
         color: inherit; resize: vertical; }
  #notepanel .nactions { display: flex; align-items: center; gap: .5rem; margin-top: .5rem; }
  #notepanel .nstatus { font: 12px ui-monospace, monospace; color: #888; }
  @media (max-width: 860px) {
    body.has-notes { max-width: 760px; }
    body.has-notes #cols { display: block; }
    body.has-notes #notepanel { flex: none; max-width: none; position: static;
           max-height: none; margin-top: 1.5rem; }
  }
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
  .ts { font: 12px ui-monospace, monospace; color: #4a9; cursor: pointer;
        text-decoration: underline; }
  /* speaker mapping panel */
  #rosterwrap summary { cursor: pointer; font: 12px ui-monospace, monospace;
        color: #888; padding: .15rem 0; }
  #rosterwrap .hint { opacity: .7; }
  #roster { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0 .2rem; }
  .rpill { display: inline-flex; align-items: center; gap: .35rem;
           border: 1px solid #8884; border-radius: 999px;
           padding: .12rem .55rem .12rem .2rem; }
  .rpill .chip { width: 20px; height: 20px; border-radius: 50%;
                 font: 600 11px/20px system-ui; }
  .rname { font-size: 13px; cursor: pointer; }
  .rname:hover { color: #4a9; text-decoration: underline; }
  .rname.named { color: #2a8; }
  .rsrc { font: 10px ui-monospace, monospace; color: #888; }
  .rpill input { font: inherit; font-size: 13px; padding: .05rem .35rem; width: 9rem;
                 max-width: 40vw; border: 1px solid #4a9; border-radius: 6px;
                 background: transparent; color: inherit; }
  .rpill .save, .rpill .cancel { font: 12px ui-monospace, monospace; cursor: pointer;
        border: 1px solid #8886; border-radius: 6px; background: transparent;
        color: inherit; padding: .05rem .4rem; }
  .rpill .save:hover { border-color: #4a9; color: #4a9; }
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
  <details id="rosterwrap" open>
    <summary>Speakers &mdash; <span id="rcount"></span> <span class="hint">(click a name to edit the mapping)</span></summary>
    <div id="roster"></div>
  </details>
  <div id="find">
    <input id="q" type="search" placeholder="Search transcript&hellip;" autocomplete="off">
    <span id="count"></span>
    <button id="prev" title="previous match (Shift+Enter)">&#9650;</button>
    <button id="next" title="next match (Enter)">&#9660;</button>
  </div>
</div>
<div id="cols">
  <div id="main">
    <div id="state">loading&hellip;</div>
    <div id="list"></div>
  </div>
  <aside id="notepanel">
    <div class="nhead">
      <h2 class="ntitle">Notes</h2>
      <button id="noteEdit" class="nbtn" type="button">Edit</button>
    </div>
    <div class="nbody"></div>
    <div id="noteEditor" class="hide">
      <textarea id="noteText" placeholder="Write notes in Markdown&hellip;"></textarea>
      <div class="nactions">
        <button id="noteSave" class="nbtn" type="button">Save</button>
        <button id="noteCancel" class="nbtn" type="button">Cancel</button>
        <span class="nstatus" id="noteStatus"></span>
      </div>
    </div>
  </aside>
</div>
<script>
const STEM = {{STEM}};
const audio = document.getElementById('player');
const list = document.getElementById('list');
const state = document.getElementById('state');
const q = document.getElementById('q');
const countEl = document.getElementById('count');
const roster = document.getElementById('roster');
const rcount = document.getElementById('rcount');
const PALETTE = ['#3aa087', '#5b8def', '#c98a4b', '#a06ee0',
                 '#7da33c', '#d4647c', '#4aa0b5', '#b08f3e'];
let segs = [], marks = [], cur = -1, canPlay = true, lastSeg = -1, speakerOrder = [];
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

// turn headers are read-only; all naming happens in the mapping panel
function whoInner(raw, t) {
  const name = displayName(raw);
  return '<span class="chip" style="background:' + (colors[raw] || '#888') + '">' +
    esc(name[0].toUpperCase()) + '</span>' +
    '<b class="name' + (isNamed(raw) ? ' named' : '') + '">' + esc(name) + '</b>' +
    (t != null ? '<span class="ts" data-t="' + t + '">' + fmtTs(t) + '</span>' : '');
}

function render() {
  let ci = 0, html = '', lastSpk = null;
  speakerOrder = [];
  segs.forEach((s, i) => {
    const spk = s.speaker || 'UNKNOWN';
    if (spk !== lastSpk) {
      if (lastSpk !== null) html += '</p></div>';
      if (!(spk in colors)) { colors[spk] = PALETTE[ci++ % PALETTE.length]; }
      if (!speakerOrder.includes(spk)) speakerOrder.push(spk);
      html += '<div class="turn" data-spk="' + esc(spk) + '" data-t="' + s.start +
        '"><div class="who">' + whoInner(spk, s.start) + '</div><p>';
      lastSpk = spk;
    }
    html += '<span class="seg" data-i="' + i + '">' + esc(s.text.trim()) + '</span> ';
  });
  if (lastSpk !== null) html += '</p></div>';
  list.innerHTML = html;
}

// --- speaker mapping panel ------------------------------------------------
function rpillInner(raw) {
  const name = displayName(raw);
  const src = manual[raw] ? '' : (auto[raw] ? ' <span class="rsrc">(voice)</span>' : '');
  return '<span class="chip" style="background:' + (colors[raw] || '#888') + '">' +
    esc(name[0].toUpperCase()) + '</span>' +
    '<span class="rname' + (isNamed(raw) ? ' named' : '') +
    '" title="' + esc(label(raw)) + '">' + esc(name) + '</span>' + src;
}

function buildRoster() {
  roster.innerHTML = speakerOrder.map(raw =>
    '<span class="rpill" data-spk="' + esc(raw) + '">' + rpillInner(raw) + '</span>'
  ).join('');
  rcount.textContent = speakerOrder.length;
}

// repaint a speaker everywhere its name appears (turns + mapping pill)
function repaint(raw) {
  list.querySelectorAll('.turn[data-spk="' + CSS.escape(raw) + '"]').forEach(turn => {
    turn.querySelector('.who').innerHTML = whoInner(raw, Number(turn.dataset.t));
  });
  const pill = roster.querySelector('.rpill[data-spk="' + CSS.escape(raw) + '"]');
  if (pill) pill.innerHTML = rpillInner(raw);
}

// persist one name to the SQLite map; '' clears it (reverts to voice/Speaker N)
async function saveName(raw, value) {
  const name = (value || '').trim();
  const r = await fetch('/names/' + encodeURIComponent(STEM), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speaker_raw: raw, name }),
  });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  const data = await r.json();
  if (data.name) manual[raw] = data.name; else delete manual[raw];
}

function editPill(pill) {
  const raw = pill.dataset.spk;
  pill.innerHTML =
    '<span class="chip" style="background:' + (colors[raw] || '#888') + '"></span>' +
    '<input type="text" maxlength="80"> ' +
    '<button class="save">save</button> <button class="cancel">cancel</button>';
  const inp = pill.querySelector('input');
  inp.value = manual[raw] || '';
  inp.placeholder = auto[raw] || label(raw);
  inp.focus();
  inp.select();
  inp.addEventListener('keydown', ev => {
    if (ev.key === 'Enter') { ev.preventDefault(); commitPill(pill); }
    else if (ev.key === 'Escape') { ev.preventDefault(); pill.innerHTML = rpillInner(raw); }
  });
}

async function commitPill(pill) {
  const raw = pill.dataset.spk, inp = pill.querySelector('input');
  if (!inp) return;
  inp.disabled = true;
  try { await saveName(raw, inp.value); }
  catch (err) { alert('Could not save name: ' + err); }
  repaint(raw);
}

roster.addEventListener('click', e => {
  const pill = e.target.closest('.rpill');
  if (!pill) return;
  if (e.target.closest('.save')) { commitPill(pill); return; }
  if (e.target.closest('.cancel')) { pill.innerHTML = rpillInner(pill.dataset.spk); return; }
  if (e.target.closest('input')) return;          // typing — ignore
  if (!pill.querySelector('input')) editPill(pill);  // any other click opens the editor
});

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
  buildRoster();
  state.remove();
}

// Notes side panel: an editable, hand-authored note for this stem. The server
// always answers 200 — {exists:false} just means the panel starts empty. Saving
// writes NOTES/<stem>.note.md (the only read-write content dir); any failure is
// surfaced inline and never affects the transcript itself (graceful absence).
let noteMarkdown = '';
const noteBodyEl = () => document.querySelector('#notepanel .nbody');

function renderNote(htmlStr) {
  const body = noteBodyEl();
  if (htmlStr) { body.innerHTML = htmlStr; body.classList.remove('nempty'); }
  else { body.textContent = 'No notes yet.'; body.classList.add('nempty'); }
}

async function loadNote() {
  document.body.classList.add('has-notes');     // panel is always available
  try {
    const r = await fetch('/note/' + encodeURIComponent(STEM));
    if (!r.ok) { renderNote(''); return; }
    const data = await r.json();
    noteMarkdown = data.markdown || '';
    renderNote(data.exists ? data.html : '');
  } catch (e) { renderNote(''); }               // never break the viewer over a note
}

function noteEdit(on) {
  document.getElementById('noteEditor').classList.toggle('hide', !on);
  noteBodyEl().classList.toggle('hide', on);
  document.getElementById('noteEdit').classList.toggle('hide', on);
  if (on) {
    document.getElementById('noteText').value = noteMarkdown;
    document.getElementById('noteStatus').textContent = '';
    document.getElementById('noteText').focus();
  }
}

async function saveNote() {
  const md = document.getElementById('noteText').value;
  const status = document.getElementById('noteStatus');
  status.textContent = 'saving…';
  try {
    const r = await fetch('/note/' + encodeURIComponent(STEM), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ markdown: md }),
    });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || ('HTTP ' + r.status));
    noteMarkdown = md;
    renderNote(data.exists ? data.html : '');
    noteEdit(false);
  } catch (e) { status.textContent = 'save failed: ' + e.message; }
}

document.getElementById('noteEdit').addEventListener('click', () => noteEdit(true));
document.getElementById('noteCancel').addEventListener('click', () => noteEdit(false));
document.getElementById('noteSave').addEventListener('click', saveNote);

function seek(t) {
  if (!canPlay) return;
  audio.currentTime = t;
  audio.play();
}
// turn headers are read-only now; clicks just seek the audio
list.addEventListener('click', e => {
  const seg = e.target.closest('.seg'), ts = e.target.closest('.ts');
  if (ts) seek(Number(ts.dataset.t));
  else if (seg) seek(segs[Number(seg.dataset.i)].start);
});

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
loadNote();
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
        m = re.match(r"^/roster/([^/]+)$", path)
        if m:
            return self._roster_get(urllib.parse.unquote(m.group(1)))
        m = re.match(r"^/note/([^/]+)$", path)
        if m:
            return self._note_get(urllib.parse.unquote(m.group(1)))
        return super().do_GET()

    def do_HEAD(self):
        m = re.match(r"^/audio/([^/]+)$", urllib.parse.urlparse(self.path).path)
        if m:
            return self._audio(urllib.parse.unquote(m.group(1)), head=True)
        return super().do_HEAD()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        m = re.match(r"^/names/([^/]+)$", path)
        if m:
            return self._names_post(urllib.parse.unquote(m.group(1)))
        m = re.match(r"^/note/([^/]+)$", path)
        if m:
            return self._note_post(urllib.parse.unquote(m.group(1)))
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

    def _roster_get(self, raw):
        stem = safe_name(raw)
        if not stem:
            return self._json(400, {"error": "bad name"})
        roster = roster_for(stem)
        if roster is None:
            return self._json(404, {"error": "no transcript JSON for that name"})
        return self._json(200, {"stem": stem, "roster": roster})

    def _note_get(self, raw):
        """Editable note (if any) for one meeting. Always 200:
        {"exists": bool, "markdown": str, "html": str}. No note -> exists=false
        with empty strings; the viewer shows an empty (still editable) panel.
        Never an error, so the page is unaffected."""
        stem = safe_name(raw)
        if not stem:
            return self._json(400, {"error": "bad name"})
        md = note_markdown(stem)
        body = note_html(stem)
        return self._json(200, {"exists": body is not None,
                                "markdown": md, "html": body or ""})

    def _note_post(self, raw):
        """Save the editable note for one meeting (body {"markdown": "..."}); a
        blank body clears it. Requires the transcript to exist, and writes only
        to the read-write NOTES dir (OUTBOX stays read-only)."""
        stem = safe_name(raw)
        if not stem or not os.path.isfile(os.path.join(OUTBOX, stem + ".json")):
            return self._json(404, {"error": "no transcript for that name"})
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 1 << 20:        # 1 MiB ceiling for a note
            return self._json(400, {"error": "missing or oversized body"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            markdown = str(payload.get("markdown", ""))
        except Exception as exc:  # noqa: BLE001
            return self._json(400, {"error": f"bad request: {exc}"})
        try:
            body = save_note(stem, markdown)
        except Exception as exc:  # noqa: BLE001
            return self._json(500, {"error": str(exc)})
        self.log_message("note %s -> %d bytes", stem, len(markdown))
        return self._json(200, {"ok": True, "exists": body is not None,
                                "html": body or ""})

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
            if n.endswith(".note.md") or n.endswith(".reflection.md"):
                continue  # attached note (current / legacy), shown in the viewer
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
