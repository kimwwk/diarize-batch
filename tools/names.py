#!/usr/bin/env python3
"""CLI for the speaker name map — the ssh-side twin of the viewer's edit box.

Same SQLite store the fileserver writes (`data/db/speakers.db`), keyed by the
raw diarizer label (SPEAKER_07), so an edit here shows up in the viewer and vice
versa. Stdlib only.

Usage (run from the project root, or pass --db / set DB_PATH):
  python3 tools/names.py list                       # every mapping
  python3 tools/names.py list <meeting-stem>        # one meeting
  python3 tools/names.py set  <stem> <SPEAKER_xx> <name>
  python3 tools/names.py del  <stem> <SPEAKER_xx>

<meeting-stem> is the transcript name without extension, e.g.
  2026-06-12_1213_Siloen
<SPEAKER_xx> is the RAW label from the .json/.speakers.json ("SPEAKER_02"),
not the cosmetic "Speaker 3".
"""
import argparse
import os
import sqlite3
import sys
import time

DEFAULT_DB = os.environ.get("DB_PATH", os.path.join("data", "db", "speakers.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_names (
    meeting      TEXT NOT NULL,
    speaker_raw  TEXT NOT NULL,
    name         TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (meeting, speaker_raw)
);
"""


def connect(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)
    return conn


def cmd_list(conn, meeting=None):
    if meeting:
        rows = conn.execute(
            "SELECT meeting, speaker_raw, name, updated_at FROM speaker_names "
            "WHERE meeting=? ORDER BY speaker_raw", (meeting,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT meeting, speaker_raw, name, updated_at FROM speaker_names "
            "ORDER BY meeting, speaker_raw").fetchall()
    if not rows:
        print("(no names yet)")
        return
    w = max(len(r[0]) for r in rows)
    for meeting, spk, name, ts in rows:
        print(f"{meeting:<{w}}  {spk:<12} {name}   ({ts})")


def cmd_set(conn, meeting, speaker_raw, name):
    name = name.strip()
    if not name:
        return cmd_del(conn, meeting, speaker_raw)
    conn.execute(
        "INSERT INTO speaker_names (meeting, speaker_raw, name, updated_at) "
        "VALUES (?,?,?,?) ON CONFLICT(meeting, speaker_raw) "
        "DO UPDATE SET name=excluded.name, updated_at=excluded.updated_at",
        (meeting, speaker_raw, name,
         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    conn.commit()
    print(f"set {meeting} / {speaker_raw} -> {name!r}")


def cmd_del(conn, meeting, speaker_raw):
    cur = conn.execute(
        "DELETE FROM speaker_names WHERE meeting=? AND speaker_raw=?",
        (meeting, speaker_raw))
    conn.commit()
    print(f"deleted {cur.rowcount} row(s) for {meeting} / {speaker_raw}")


def main(argv=None):
    p = argparse.ArgumentParser(description="speaker name map CLI")
    p.add_argument("--db", default=DEFAULT_DB, help=f"SQLite path (default {DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)
    s_list = sub.add_parser("list", help="show mappings")
    s_list.add_argument("meeting", nargs="?")
    s_set = sub.add_parser("set", help="add/update a name")
    s_set.add_argument("meeting"); s_set.add_argument("speaker_raw"); s_set.add_argument("name")
    s_del = sub.add_parser("del", help="remove a name")
    s_del.add_argument("meeting"); s_del.add_argument("speaker_raw")
    args = p.parse_args(argv)

    conn = connect(args.db)
    try:
        if args.cmd == "list":
            cmd_list(conn, args.meeting)
        elif args.cmd == "set":
            cmd_set(conn, args.meeting, args.speaker_raw, args.name)
        elif args.cmd == "del":
            cmd_del(conn, args.meeting, args.speaker_raw)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
