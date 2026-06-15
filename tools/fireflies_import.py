#!/usr/bin/env python3
"""Import a Fireflies.ai meeting (transcript + audio) into this instance.

No GPU / RunPod. Pulls the transcript from the Fireflies GraphQL API, renders it
through the project's own ``orchestrator/render.py`` (so the output is
byte-identical to what the WhisperX+pyannote pipeline would produce), seeds the
speaker name map from Fireflies' speaker labels, drops the AI summary into a
editable ``.note.md`` side-panel, and downloads the audio. The finished artifacts
land straight in ``data/outbox/`` + ``data/done/``, so the orchestrator never
fires and no pod ever boots. Stdlib only.

The FREE Fireflies API tier exposes ``sentences`` + ``speakers`` + ``summary``,
but NOT ``audio_url`` (paid, ``pro_or_higher``). Audio is fetched from the signed
``cdn.fireflies.ai/<id>/audio.mp3`` URL you pass with ``--audio-url`` — grab it
from the logged-in web player (it is time-limited). ``--audio-url`` also accepts
a local file path if you already downloaded it.

API key resolution (first found wins):
  --api-key  >  $FIREFLIES_API_KEY  >  <secrets-dir>/fireflies.key

Usage (run from anywhere; paths anchor to the repo root):
  python3 tools/fireflies_import.py <fireflies-link-or-id> --audio-url '<signed url>'
  python3 tools/fireflies_import.py <link-or-id> --slug sileon-surv7x --time 1000
  python3 tools/fireflies_import.py <link-or-id> --dry-run     # render to ./_fireflies_preview, touch nothing live
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "orchestrator"))
import render  # noqa: E402  (the real pipeline renderer — source of truth)

FF_ENDPOINT = "https://api.fireflies.ai/graphql"

TRANSCRIPT_QUERY = """
query T($id: String!) {
  transcript(id: $id) {
    id title date duration
    speakers { id name }
    sentences { index speaker_name speaker_id text start_time end_time }
    summary { overview short_summary action_items keywords }
  }
}
"""

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_names (
    meeting      TEXT NOT NULL,
    speaker_raw  TEXT NOT NULL,
    name         TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (meeting, speaker_raw)
);
"""


def resolve_api_key(cli_key, secrets_dir):
    if cli_key:
        return cli_key.strip()
    env = os.environ.get("FIREFLIES_API_KEY")
    if env:
        return env.strip()
    key_file = os.path.join(secrets_dir, "fireflies.key")
    if os.path.exists(key_file):
        with open(key_file) as fh:
            return fh.read().strip()
    sys.exit(
        "No Fireflies API key. Pass --api-key, set $FIREFLIES_API_KEY, or create "
        f"{key_file}")


def transcript_id_from(arg):
    """Accept a full view link or a bare id. Examples:
      https://app.fireflies.ai/view/ar-hoi-kasoku-m4a::01KTJ8AJ...?x=y -> 01KTJ8AJ...
      https://app.fireflies.ai/view/01KV29F8...                        -> 01KV29F8...
      01KV29F8...                                                       -> 01KV29F8...
    """
    s = arg.strip().split("?")[0].split("#")[0]
    if "::" in s:                       # slug::id form
        s = s.split("::")[-1]
    elif "/" in s:                      # plain /view/<id>
        s = s.rstrip("/").split("/")[-1]
    if not re.fullmatch(r"[A-Za-z0-9]{20,32}", s):
        sys.exit(f"Could not parse a Fireflies transcript id from {arg!r} (got {s!r})")
    return s


def gql(query, variables, api_key, timeout=90):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(FF_ENDPOINT, data=body, headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            sys.exit(f"Fireflies API HTTP {e.code}: {e.reason}")


def slugify(text):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "meeting"


def derive_stem(t, args):
    ms = t.get("date") or 0
    gm = time.gmtime(ms / 1000) if ms else time.gmtime(0)
    date = args.date or time.strftime("%Y-%m-%d", gm)
    hhmm = args.time or time.strftime("%H%M", gm)
    hhmm = re.sub(r"[^0-9]", "", hhmm)
    if len(hhmm) != 4:
        sys.exit(f"--time must be HHMM (got {args.time!r})")
    slug = args.slug or slugify(t.get("title"))
    return f"{date}_{hhmm}_{slug}"


def build_result(t, model, language):
    segments = []
    for s in t.get("sentences") or []:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        sid = s.get("speaker_id")
        raw = f"SPEAKER_{int(sid):02d}" if sid is not None else "UNKNOWN"
        segments.append({
            "start": float(s["start_time"]),
            "end": float(s["end_time"]),
            "speaker": raw,
            "text": txt,
        })
    name_by_raw = {}
    for sp in (t.get("speakers") or []):
        if sp.get("id") is not None and sp.get("name"):
            name_by_raw[f"SPEAKER_{int(sp['id']):02d}"] = sp["name"]
    duration = round(float(t.get("duration") or 0) * 60, 3)  # FF duration is MINUTES
    if segments and segments[-1]["end"] > duration:
        duration = round(segments[-1]["end"], 3)
    result = {
        "segments": segments,
        "language": language,
        "duration": duration,
        "num_speakers": len({s["speaker"] for s in segments}),
        "model": model,
        "diarized": True,
    }
    return result, name_by_raw


def write_note(summary, path):
    def block(title, val):
        if not val:
            return ""
        body = "\n".join(f"- {x}" for x in val) if isinstance(val, list) else str(val).strip()
        return f"## {title}\n\n{body}\n\n"
    parts = ["# Fireflies summary\n",
             "_Imported from Fireflies.ai — AI-generated meeting summary._\n\n",
             block("Overview", summary.get("overview")),
             block("Short summary", summary.get("short_summary")),
             block("Action items", summary.get("action_items")),
             block("Keywords", summary.get("keywords"))]
    text = "".join(p for p in parts if p)
    if text.strip() == "# Fireflies summary":
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return True


def seed_names(db_path, stem, name_by_raw):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(DB_SCHEMA)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for raw, name in name_by_raw.items():
            conn.execute(
                "INSERT INTO speaker_names (meeting, speaker_raw, name, updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(meeting, speaker_raw) "
                "DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
                (stem, raw, name, now))
        conn.commit()
    finally:
        conn.close()


def fetch_audio(src, dest):
    """src may be a signed CDN URL or a local file path."""
    if os.path.exists(src):
        shutil.copyfile(src, dest)
        return os.path.getsize(dest)
    req = urllib.request.Request(src, headers={"User-Agent": "diarize-batch/fireflies-import"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    return os.path.getsize(dest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("meeting", help="Fireflies view link or bare transcript id")
    p.add_argument("--audio-url", help="signed cdn.fireflies.ai audio URL (or a local file path)")
    p.add_argument("--slug", help="override the slug (default: slugified Fireflies title)")
    p.add_argument("--date", help="override meeting date YYYY-MM-DD (default: from API, UTC)")
    p.add_argument("--time", help="override meeting time HHMM (default: from API, UTC). "
                                   "Tip: use the owner's LOCAL clock time.")
    p.add_argument("--api-key", help="Fireflies API key (else $FIREFLIES_API_KEY or secrets/fireflies.key)")
    p.add_argument("--model", default="fireflies", help="provenance label in the frontmatter")
    p.add_argument("--language", default="en")
    p.add_argument("--outbox", help="transcript dir (default <repo>/data/outbox)")
    p.add_argument("--done", help="audio archive dir (default <repo>/data/done)")
    p.add_argument("--notes", help="editable-notes dir (default <repo>/data/notes)")
    p.add_argument("--db", help="speaker name DB (default <repo>/data/db/speakers.db)")
    p.add_argument("--secrets-dir", default=os.path.join(REPO, "secrets"))
    p.add_argument("--no-names", action="store_true", help="don't seed the speaker name map")
    p.add_argument("--dry-run", action="store_true",
                   help="render into ./_fireflies_preview and touch nothing live")
    args = p.parse_args(argv)

    if args.dry_run:
        base = os.path.join(os.getcwd(), "_fireflies_preview")
        outbox = done = notes = base
        db_path = os.path.join(base, "speakers.db")
    else:
        outbox = args.outbox or os.path.join(REPO, "data", "outbox")
        done = args.done or os.path.join(REPO, "data", "done")
        notes = args.notes or os.path.join(REPO, "data", "notes")
        db_path = args.db or os.path.join(REPO, "data", "db", "speakers.db")

    api_key = resolve_api_key(args.api_key, args.secrets_dir)
    tid = transcript_id_from(args.meeting)
    print(f"[fireflies] fetching transcript {tid} ...")
    resp = gql(TRANSCRIPT_QUERY, {"id": tid}, api_key)
    for err in resp.get("errors", []):
        print(f"  ! API: {err.get('message')} (path={err.get('path')})", file=sys.stderr)
    t = (resp.get("data") or {}).get("transcript")
    if not t:
        sys.exit("No transcript returned (bad id, wrong account key, or not accessible).")
    sentences = t.get("sentences") or []
    if not sentences:
        sys.exit("Transcript has 0 sentences — not transcribed on Fireflies yet (or still processing).")

    stem = derive_stem(t, args)
    result, name_by_raw = build_result(t, args.model, args.language)
    source_name = stem + ".mp3"

    os.makedirs(outbox, exist_ok=True)
    out_stem = os.path.join(outbox, stem)
    paths = render.write_outputs(result, out_stem, source_name)
    # The Fireflies summary seeds the editable note (NOTES/<stem>.note.md); the
    # user can then edit/replace it in the viewer. NOTES is a separate dir.
    os.makedirs(notes, exist_ok=True)
    note_file = os.path.join(notes, stem + ".note.md")
    has_note = write_note(t.get("summary") or {}, note_file)

    if not args.no_names and name_by_raw:
        seed_names(db_path, stem, name_by_raw)

    audio_note = "skipped (no --audio-url)"
    if args.audio_url:
        os.makedirs(done, exist_ok=True)
        dest = os.path.join(done, stem + ".mp3")
        try:
            size = fetch_audio(args.audio_url, dest)
            audio_note = f"{size/1e6:.1f} MB -> {dest}"
        except Exception as e:
            audio_note = f"FAILED ({e}); transcript imported, audio missing"

    print()
    print(f"  stem        : {stem}")
    print(f"  title       : {t.get('title')!r}")
    print(f"  segments    : {len(result['segments'])} | speakers: {result['num_speakers']} "
          f"| dur: {render._short_ts(result['duration'])}")
    print(f"  names       : {name_by_raw or '(none)'}{' [SKIPPED]' if args.no_names else ''}")
    print(f"  note        : {'seeded ' + os.path.basename(note_file) if has_note else 'no summary available'}")
    print(f"  audio       : {audio_note}")
    print(f"  files       : {', '.join(os.path.basename(x) for x in paths)}"
          + (", "+os.path.basename(note_file) if has_note else ""))
    if args.dry_run:
        print(f"\n  DRY RUN — wrote to {outbox} ; nothing live touched.")
    else:
        host = os.environ.get("DIARIZE_HOST", "http://192.168.1.129:8080")
        print(f"\n  live: {host}/view/{stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
