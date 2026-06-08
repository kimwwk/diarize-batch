"""Render the worker's diarized segments into local transcript files.

The **JSON is the source of truth**: it holds the full metadata + raw segments
(SPEAKER_00 labels intact) + pre-merged turns, so every other format can be
re-rendered from it without re-running the GPU. The **Markdown** is the
human-and-LLM-facing view: YAML frontmatter (browsable + machine-parseable) plus
speaker-labelled turns. SRT/TXT are convenience exports.

Diarizer labels look like "SPEAKER_00"; we relabel those to a friendly
"Speaker 1" in the human-readable outputs while keeping the raw labels in JSON.
"""
import json
import os
import re


def _parse_stem(source_name):
    """Pull (date, time, title) out of a corpus-style filename like
    '2026-05-27_0959_the-owners.m4a'. Falls back to (None, None, prettified-stem)
    when the name doesn't match that convention."""
    stem = os.path.splitext(os.path.basename(source_name))[0]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})_(.+)$", stem)
    if m:
        date, hh, mm, slug = m.groups()
        title = slug.replace("-", " ").replace("_", " ").strip().title()
        return date, f"{hh}:{mm}", title
    return None, None, stem.replace("-", " ").replace("_", " ").strip().title()


def _doc_meta(result, source_name):
    """Frontmatter/metadata shared by the JSON source-of-truth and the Markdown.
    Drops keys whose value is None so the frontmatter stays clean."""
    date, time, title = _parse_stem(source_name)
    duration = result.get("duration")
    meta = {
        "title": title,
        "date": date,
        "time": time,
        "audio": os.path.basename(source_name),
        "language": result.get("language"),
        "duration": _short_ts(duration) if duration else None,
        "duration_seconds": duration,
        "speakers": result.get("num_speakers"),
        "model": result.get("model"),
        "diarized": result.get("diarized"),
        "generated_by": "diarize-batch",
    }
    return {k: v for k, v in meta.items() if v is not None}


def _yaml_scalar(v):
    """Render a value as a YAML scalar. Strings are emitted as double-quoted
    (JSON-style) so colons/leading-zeros/unicode never break the frontmatter."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return json.dumps(str(v), ensure_ascii=False)


def _frontmatter(meta):
    lines = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def _srt_ts(seconds):
    """HH:MM:SS,mmm for SRT."""
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _short_ts(seconds):
    """MM:SS, or HH:MM:SS past an hour."""
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _label(speaker):
    if not speaker or speaker == "UNKNOWN":
        return "Unknown"
    if speaker.startswith("SPEAKER_"):
        try:
            return f"Speaker {int(speaker.split('_')[1]) + 1}"
        except ValueError:
            return speaker
    return speaker


def _merge_turns(segments):
    """Collapse consecutive segments from the same speaker into one turn."""
    turns = []
    for seg in segments:
        speaker = seg.get("speaker") or "UNKNOWN"
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = seg.get("end", turns[-1]["end"])
        else:
            turns.append({
                "speaker": speaker,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text,
            })
    return turns


def _write_json(result, source_name, path):
    """The source-of-truth file: full metadata + raw segments + merged turns."""
    meta = _doc_meta(result, source_name)
    segments = result.get("segments", [])
    payload = {
        **meta,
        "source": os.path.basename(source_name),
        "segments": segments,
        "turns": _merge_turns(segments),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _write_srt(segments, path):
    out = []
    index = 1
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append(str(index))
        out.append(f"{_srt_ts(seg.get('start'))} --> {_srt_ts(seg.get('end'))}")
        out.append(f"{_label(seg.get('speaker'))}: {text}")
        out.append("")
        index += 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


def _write_txt(segments, path):
    lines = [
        f"[{_short_ts(t['start'])}] {_label(t['speaker'])}: {t['text']}"
        for t in _merge_turns(segments)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_md(result, source_name, path):
    """Human-and-LLM view: YAML frontmatter + speaker-labelled turns."""
    meta = _doc_meta(result, source_name)
    turns = _merge_turns(result.get("segments", []))
    lines = [_frontmatter(meta), "", f"# {meta.get('title', 'Transcript')}", ""]
    for turn in turns:
        lines.append(f"**{_label(turn['speaker'])}** · `[{_short_ts(turn['start'])}]`")
        lines.append("")
        lines.append(turn["text"])
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_outputs(result, out_stem, source_name):
    """Write all output formats next to out_stem. Returns the list of file paths."""
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    segments = result.get("segments", [])
    _write_json(result, source_name, out_stem + ".json")
    _write_srt(segments, out_stem + ".srt")
    _write_txt(segments, out_stem + ".txt")
    _write_md(result, source_name, out_stem + ".md")
    return [out_stem + ext for ext in (".md", ".srt", ".txt", ".json")]
