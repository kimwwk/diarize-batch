"""Optional speaker identification (CPU, resemblyzer).

Writes an ADDITIVE `<stem>.speakers.json` that maps each diarized speaker to a
known person by matching their voice to reference clips in REF_DIR
(`<Name>.<ext>` -> person "<Name>"). It never touches the .md/.srt/.txt/.json
outputs. If resemblyzer isn't installed or no refs exist, it quietly no-ops.

Same proven approach as the old adhoc tool: embed each speaker's concatenated
speech, cosine-compare to each reference embedding, assign 1:1 above a cutoff.
"""
import json
import os
import subprocess

import config

try:
    import numpy as np
    from resemblyzer import VoiceEncoder, preprocess_wav
    _DEPS_OK = True
except Exception:  # torch/resemblyzer absent -> feature off, transcription unaffected
    _DEPS_OK = False

SR = 16000
MAX_EMBED_SECONDS = 30.0   # cap speech per speaker fed to the encoder
MIN_SEG_SECONDS = 0.8      # ignore micro-segments when gathering a speaker's audio
_REF_EXTS = (".wav", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".opus", ".webm")

_encoder = None
_refs = None               # {name: normalized embedding}


def _log(m):
    print(f"[speaker-id] {m}", flush=True)


def _decode_f32(path):
    """Decode any audio to mono 16k float32 PCM via ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def _embed(wav):
    return _norm(_encoder.embed_utterance(preprocess_wav(wav, source_sr=SR)))


def _load_refs():
    """Enroll REF_DIR/<name>.<ext> as one embedding per person (cached once)."""
    refs = {}
    if not os.path.isdir(config.REF_DIR):
        return refs
    for fn in sorted(os.listdir(config.REF_DIR)):
        if fn.startswith(".") or os.path.splitext(fn)[1].lower() not in _REF_EXTS:
            continue
        name = os.path.splitext(fn)[0]
        try:
            refs[name] = _embed(_decode_f32(os.path.join(config.REF_DIR, fn)))
            _log(f"enrolled '{name}' from {fn}")
        except Exception as e:  # noqa: BLE001
            _log(f"could not enroll {fn}: {e}")
    return refs


def _speaker_audio(wav, segs):
    """Concatenate a speaker's segments, longest first, up to MAX_EMBED_SECONDS."""
    parts, total = [], 0.0
    for s in sorted(segs, key=lambda x: (x.get("end", 0) - x.get("start", 0)), reverse=True):
        dur = s.get("end", 0) - s.get("start", 0)
        if dur < MIN_SEG_SECONDS:
            continue
        parts.append(wav[int(s["start"] * SR):int(s["end"] * SR)])
        total += dur
        if total >= MAX_EMBED_SECONDS:
            break
    if not parts:
        parts = [wav[int(s.get("start", 0) * SR):int(s.get("end", 0) * SR)] for s in segs]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def identify(audio_path, segments):
    """Map render labels -> {name, matched, similarity, ...}, or None if unavailable."""
    global _encoder, _refs
    if not (_DEPS_OK and config.SPEAKER_ID):
        return None
    if _encoder is None:
        _encoder = VoiceEncoder("cpu")
        _refs = _load_refs()
    if not _refs:
        return None

    by_spk = {}
    for s in segments:
        spk = s.get("speaker") or "UNKNOWN"
        if spk != "UNKNOWN":
            by_spk.setdefault(spk, []).append(s)
    if not by_spk:
        return None

    wav = _decode_f32(audio_path)
    emb = {}
    for spk, segs in by_spk.items():
        a = _speaker_audio(wav, segs)
        if len(a) >= SR // 2:
            emb[spk] = _embed(a)

    # cosine of every speaker against every reference
    scored = {spk: {name: float(ref @ e) for name, ref in _refs.items()}
              for spk, e in emb.items()}
    # greedy 1:1: take the highest (speaker, ref) pairs above the cutoff
    pairs = sorted(((sim, spk, name)
                    for spk, d in scored.items() for name, sim in d.items()), reverse=True)
    assigned, used = {}, set()
    for sim, spk, name in pairs:
        if sim < config.SPEAKER_MIN_SIM or spk in assigned or name in used:
            continue
        assigned[spk] = (name, sim)
        used.add(name)

    from render import _label
    mapping = {}
    for spk in by_spk:
        label = _label(spk)
        if spk in assigned:
            name, sim = assigned[spk]
            mapping[label] = {"name": name, "raw": spk, "matched": True,
                              "similarity": round(sim, 3)}
        else:
            d = scored.get(spk) or {}
            best = max(d, key=d.get) if d else None
            mapping[label] = {
                "name": label, "raw": spk, "matched": False,
                "closest_ref": best,
                "closest_similarity": round(d[best], 3) if best is not None else None,
            }
    return mapping


def write_speaker_map(audio_path, segments, out_stem, source_name):
    """Write <out_stem>.speakers.json (additive). Returns its path or None."""
    mapping = identify(audio_path, segments)
    if not mapping:
        return None
    payload = {
        "audio": os.path.basename(source_name),
        "generated_by": "diarize-batch speaker-id (resemblyzer)",
        "min_similarity": config.SPEAKER_MIN_SIM,
        "speakers": mapping,
    }
    path = out_stem + ".speakers.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path
