#!/usr/bin/env python3
"""Enroll a speaker's voice from a finished meeting into refs/<Name>.flac.

"Remembering" a person = dropping a clean clip of their voice into refs/. This
mines that clip straight from a meeting you've already transcribed: it reads the
diarized segments from OUTBOX/<stem>.json, grabs that speaker's longest stretches
of speech from the archived audio in DONE/<stem>.<ext>, and writes a 16 kHz mono
FLAC to REF_DIR/<Name>.flac. The orchestrator then matches that voice in every
future upload (writing matched names into each <stem>.speakers.json).

Runs where ffmpeg lives (the orchestrator image). Defaults match that container's
mounts; override with env or flags for local use.

Examples:
  # one person, explicit:
  enroll.py 2026-06-12_1213_Siloen SPEAKER_01 "Leon"
  enroll.py 2026-06-12_1213_Siloen "Speaker 2" "Leon"     # cosmetic label ok

  # everyone you named in the viewer/DB for that meeting (skips existing refs):
  enroll.py 2026-06-12_1213_Siloen --from-db
"""
import argparse
import json
import os
import re
import subprocess
import sys

OUTBOX = os.environ.get("OUTBOX_DIR", "/data/outbox")
DONE = os.environ.get("DONE_DIR", "/data/done")
REF_DIR = os.environ.get("REF_DIR", "/refs")
DB_PATH = os.environ.get("DB_PATH", "/data/db/speakers.db")
AUDIO_EXTS = (".m4a", ".mp4", ".wav", ".flac", ".mp3", ".aac",
              ".ogg", ".webm", ".opus", ".mkv")

MAX_EMBED_SECONDS = 30.0   # how much speech to capture per person
MIN_SEG_SECONDS = 0.8      # ignore micro-segments


def die(msg):
    print(f"enroll: {msg}", file=sys.stderr)
    sys.exit(1)


def raw_label(s):
    """Accept 'SPEAKER_02' as-is, or convert cosmetic 'Speaker 3' -> 'SPEAKER_02'."""
    s = s.strip()
    if re.fullmatch(r"SPEAKER_\d+", s):
        return s
    m = re.fullmatch(r"[Ss]peaker[ _]?(\d+)", s)
    if m:
        return f"SPEAKER_{int(m.group(1)) - 1:02d}"
    return s  # let it fail later if it matches nothing


def safe_ref_name(name):
    name = os.path.basename(name.replace("\\", "/")).strip().lstrip(".")
    if not name:
        die("empty speaker name")
    return name


def find_audio(stem):
    for ext in AUDIO_EXTS:
        p = os.path.join(DONE, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def load_segments(stem):
    path = os.path.join(OUTBOX, stem + ".json")
    if not os.path.isfile(path):
        die(f"no transcript JSON: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("segments", [])


def pick_clips(segments, speaker_raw):
    """That speaker's segments, longest first up to MAX_EMBED_SECONDS, then
    re-sorted chronologically for a natural-sounding clip."""
    mine = [s for s in segments if (s.get("speaker") or "UNKNOWN") == speaker_raw]
    if not mine:
        return []
    longest = sorted(mine, key=lambda s: s.get("end", 0) - s.get("start", 0), reverse=True)
    chosen, total = [], 0.0
    for s in longest:
        dur = s.get("end", 0) - s.get("start", 0)
        if dur < MIN_SEG_SECONDS:
            continue
        chosen.append(s)
        total += dur
        if total >= MAX_EMBED_SECONDS:
            break
    if not chosen:                      # speaker only has micro-segments
        chosen = longest[:8]
    return sorted(chosen, key=lambda s: s.get("start", 0))


def cut(audio, clips, out_path):
    """ffmpeg: atrim each chosen window, concat, downmix to 16 kHz mono FLAC."""
    parts, labels = [], []
    for i, s in enumerate(clips):
        parts.append(
            f"[0:a]atrim=start={float(s['start']):.3f}:end={float(s['end']):.3f},"
            f"asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[a{i}]")
    fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(clips)}:v=0:a=1[out]"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", audio, "-filter_complex", fc,
           "-map", "[out]", "-ac", "1", "-ar", "16000", "-c:a", "flac", out_path]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        die(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[-400:]}")


def db_names(stem):
    import sqlite3
    if not os.path.isfile(DB_PATH):
        die(f"no name DB at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH, timeout=5)
    try:
        rows = conn.execute(
            "SELECT speaker_raw, name FROM speaker_names WHERE meeting=?",
            (stem,)).fetchall()
    finally:
        conn.close()
    return rows


def enroll_one(stem, speaker_raw, name, audio, segments, force):
    out = os.path.join(REF_DIR, safe_ref_name(name) + ".flac")
    if os.path.exists(out) and not force:
        print(f"skip {name!r}: {out} already exists (use --force to overwrite)")
        return False
    clips = pick_clips(segments, speaker_raw)
    if not clips:
        print(f"skip {name!r}: no segments for {speaker_raw} in {stem}")
        return False
    secs = sum(c["end"] - c["start"] for c in clips)
    cut(audio, clips, out)
    print(f"enrolled {name!r} <- {speaker_raw} ({len(clips)} clips, {secs:.1f}s) -> {out}")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="enroll a speaker's voice into refs/")
    p.add_argument("stem", help="meeting stem, e.g. 2026-06-12_1213_Siloen")
    p.add_argument("speaker", nargs="?", help="SPEAKER_xx or 'Speaker N'")
    p.add_argument("name", nargs="?", help="person name -> refs/<name>.flac")
    p.add_argument("--from-db", action="store_true",
                   help="enroll every speaker named in the DB for this meeting")
    p.add_argument("--force", action="store_true", help="overwrite existing refs")
    args = p.parse_args(argv)

    audio = find_audio(args.stem)
    if not audio:
        die(f"no archived audio in {DONE} for {args.stem} "
            f"(can't enroll without the source audio)")
    segments = load_segments(args.stem)
    if not segments:
        die(f"no segments in {args.stem}.json")

    if args.from_db:
        rows = db_names(args.stem)
        if not rows:
            die(f"no names in DB for {args.stem} — name speakers first "
                f"(viewer or tools/names.py)")
        n = sum(enroll_one(args.stem, spk, nm, audio, segments, args.force)
                for spk, nm in rows)
        print(f"done: enrolled {n}/{len(rows)} named speaker(s)")
    else:
        if not (args.speaker and args.name):
            die("need <speaker> and <name>, or use --from-db")
        enroll_one(args.stem, raw_label(args.speaker), args.name,
                   audio, segments, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
