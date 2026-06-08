#!/usr/bin/env python3
"""
LOCAL validation harness for the diarize-batch pipeline.

Runs the SAME transcription+diarization pipeline as runpod-worker/handler.py on
the local GPU (RTX 4070 Laptop, 8 GB), as a stand-in for the not-yet-deployed
RunPod serverless worker. It produces the SAME output-contract dict the handler
returns, then renders .md/.json/.srt/.txt via orchestrator/render.py.

It is deliberately the ONLY new code file in the repo. It IMPORTS render.py and
mirrors handler.py; it does not modify either.

KEY DIFFERENCES vs. handler.py (all forced by the 8 GB local GPU + HF gating):

  1. VRAM DISCIPLINE: handler.py caches every model in module globals for warm
     reuse. Here we load each stage, run it, then DELETE it and empty the CUDA
     cache before loading the next stage, so peak VRAM stays ~= one model. 8 GB
     cannot hold ASR + align + diarization simultaneously.

  2. compute_type = float16 (handler default is also float16; float32 is
     contractually allowed but OOMs on 8 GB for large-v3).

  3. DIARIZATION MODEL FALLBACK: handler.py uses whisperx's default
     pyannote/speaker-diarization-community-1. On this HF account that repo is
     gated (403 — user conditions not accepted), so we FALL BACK to a
     3.1-equivalent pipeline built from the components this account *can* access:
     pyannote/segmentation-3.0 + pyannote/wespeaker-voxceleb-resnet34-LM with
     AgglomerativeClustering and the frozen speaker-diarization-3.1
     hyper-parameters. (The frozen adhoc job proved 3.1 is accepted here.)

     pyannote-audio 4.0.4's SpeakerDiarization.__init__ eagerly downloads a PLDA
     calibration file from community-1 even when clustering is Agglomerative
     (which never uses PLDA). We neutralize that single eager download with a
     local shim so no gated community-1 asset is ever fetched.

Output contract (identical to handler.py return value):
    {segments:[{start,end,speaker,text}], language, duration,
     num_speakers, model, diarized}

Usage:
    .venv/bin/python tools/local_validate.py AUDIO [--language en]
        [--min-speakers N] [--max-speakers N] [--no-diarize]
"""
import argparse
import datetime as _dt
import gc
import os
import sys
import time
import traceback

# --- Repo wiring -------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
# Import the repo's render.py (orchestrator/) WITHOUT modifying it.
sys.path.insert(0, os.path.join(_REPO, "orchestrator"))
import render  # noqa: E402  (orchestrator/render.py — source of truth for outputs)

_VALIDATION_DIR = os.path.join(_REPO, "_validation")
_LOG_PATH = os.path.join(_VALIDATION_DIR, "validate.log")

# Mirror handler.py constants.
SAMPLE_RATE = 16000

# Frozen pyannote/speaker-diarization-3.1 pipeline definition (from its
# config.yaml). Used to rebuild a 3.1-equivalent pipeline from ungated parts.
_SD31_SEGMENTATION = "pyannote/segmentation-3.0"
_SD31_EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"
_SD31_PARAMS = {
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 12,
        "threshold": 0.7045654963945799,
    },
    "segmentation": {"min_duration_off": 0.0},
}

_COMMUNITY1 = "pyannote/speaker-diarization-community-1"


def log(msg):
    """Append a timestamped line to validate.log and echo to stdout."""
    line = f"[{_dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    os.makedirs(_VALIDATION_DIR, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _read_env_file(path):
    """Minimal .env reader (no python-dotenv dependency)."""
    if not os.path.exists(path):
        return
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def _free_cuda(*objs):
    """Delete model objects and reclaim VRAM. Critical on 8 GB."""
    import torch

    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _vram(tag):
    import torch

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        log(f"[vram {tag}] allocated={used:.2f}GiB reserved={reserved:.2f}GiB")


# --- Diarization: default community-1, fall back to 3.1-equivalent -----------
def _build_diarizer(hf_token, device):
    """Return (callable_pipeline, model_label).

    The callable takes (audio_ndarray, min_speakers, max_speakers) and returns a
    whisperx-style diarize DataFrame (columns: segment,label,speaker,start,end),
    exactly what whisperx.assign_word_speakers expects.

    First tries whisperx's default DiarizationPipeline (community-1). If that
    raises (gated/403/auth), falls back to a 3.1-equivalent pipeline built from
    ungated components.
    """
    import torch
    import pandas as pd
    from whisperx.diarize import DiarizationPipeline

    # --- Attempt 1: whisperx default (pyannote/speaker-diarization-community-1)
    try:
        log("[diarize] trying whisperx default pipeline "
            f"({_COMMUNITY1}) ...")
        pipe = DiarizationPipeline(token=hf_token, device=device)

        def _run_default(audio, min_speakers, max_speakers):
            kw = {}
            if min_speakers is not None:
                kw["min_speakers"] = int(min_speakers)
            if max_speakers is not None:
                kw["max_speakers"] = int(max_speakers)
            return pipe(audio, **kw)

        log("[diarize] community-1 loaded OK")
        return _run_default, "pyannote/speaker-diarization-community-1"
    except Exception as exc:  # gated / 403 / auth → fall back
        log(f"[diarize] community-1 unavailable ({type(exc).__name__}: "
            f"{str(exc)[:140]}) — falling back to speaker-diarization-3.1")

    # --- Attempt 2: 3.1-equivalent from ungated components -------------------
    # pyannote 4.0.4 eagerly downloads community-1's PLDA in
    # SpeakerDiarization.__init__ even for AgglomerativeClustering (which never
    # uses it). Neutralize just that eager fetch.
    import pyannote.audio.pipelines.speaker_diarization as _sd_mod

    _orig_get_plda = _sd_mod.get_plda
    _sd_mod.get_plda = lambda *a, **k: None  # Agglomerative ignores PLDA
    try:
        from pyannote.audio.pipelines import SpeakerDiarization

        sd = SpeakerDiarization(
            segmentation=_SD31_SEGMENTATION,
            embedding=_SD31_EMBEDDING,
            embedding_exclude_overlap=True,
            clustering="AgglomerativeClustering",
            segmentation_batch_size=32,
            embedding_batch_size=32,
            token=hf_token,
        )
        sd.instantiate(_SD31_PARAMS)
        sd.to(torch.device(device))
    finally:
        _sd_mod.get_plda = _orig_get_plda  # restore

    def _run_31(audio, min_speakers, max_speakers):
        # pyannote wants an in-memory {waveform, sample_rate} dict; this also
        # avoids torchcodec (broken here: ffmpeg 8 unsupported by torchcodec).
        waveform = torch.from_numpy(audio[None, :])
        kw = {}
        if min_speakers is not None:
            kw["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            kw["max_speakers"] = int(max_speakers)
        out = sd({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kw)
        # pyannote 4.x returns DiarizeOutput; .speaker_diarization is an
        # Annotation. Convert to the whisperx DataFrame format (same code path
        # whisperx.diarize.DiarizationPipeline uses internally).
        annotation = out.speaker_diarization
        df = pd.DataFrame(
            annotation.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        df["start"] = df["segment"].apply(lambda s: s.start)
        df["end"] = df["segment"].apply(lambda s: s.end)
        return df

    log("[diarize] 3.1-equivalent pipeline built "
        f"({_SD31_SEGMENTATION} + {_SD31_EMBEDDING}, AgglomerativeClustering)")
    return _run_31, "pyannote/speaker-diarization-3.1 (fallback)"


def run(audio_path, language=None, min_speakers=None, max_speakers=None,
        diarize=True, model_name="large-v3", batch_size=8,
        compute_type="float16", initial_prompt=None, hf_token=None):
    """Mirror handler.handler() but with stage-wise VRAM freeing. Returns the
    output-contract dict (plus, on the side, the diarization model label)."""
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} model={model_name} compute_type={compute_type} "
        f"batch_size={batch_size} language={language or 'auto'} diarize={diarize}")

    # --- Load audio (16 kHz mono float; same as handler) --------------------
    t0 = time.time()
    audio = whisperx.load_audio(audio_path)
    duration = round(len(audio) / SAMPLE_RATE, 3)
    log(f"[audio] loaded {os.path.basename(audio_path)} "
        f"duration={duration:.1f}s ({duration/60:.1f}min)")

    # --- Stage 1: Transcribe ------------------------------------------------
    log("[asr] loading large-v3 ...")
    asr_options = {"initial_prompt": initial_prompt} if initial_prompt else None
    asr_model = whisperx.load_model(
        model_name, device, compute_type=compute_type, asr_options=asr_options,
    )
    _vram("after asr load")
    t_asr = time.time()
    result = asr_model.transcribe(audio, batch_size=batch_size, language=language)
    detected_language = result.get("language") or language or "en"
    log(f"[asr] transcribed in {time.time()-t_asr:.1f}s "
        f"({len(result.get('segments', []))} raw segments, lang={detected_language})")

    # Free the ASR model BEFORE loading the align model (8 GB discipline).
    _free_cuda(asr_model)
    _vram("after asr free")

    # --- Stage 2: Align (word-level timestamps) -----------------------------
    align_model = metadata = None
    try:
        log(f"[align] loading align model for '{detected_language}' ...")
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language, device=device,
        )
        _vram("after align load")
        t_align = time.time()
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )
        log(f"[align] aligned in {time.time()-t_align:.1f}s")
    except Exception as align_err:
        log(f"[warn] alignment skipped: {align_err}")
    finally:
        # Free the align model BEFORE diarization (8 GB discipline).
        _free_cuda(align_model, metadata)
        _vram("after align free")

    # --- Stage 3: Diarize + assign speakers ---------------------------------
    diarized = False
    diarize_model_label = "none"
    if diarize:
        if not hf_token:
            raise ValueError(
                "diarize=true but no HF token (set HF_TOKEN in .env)."
            )
        diarizer, diarize_model_label = _build_diarizer(hf_token, device)
        _vram("after diarizer load")
        t_diar = time.time()
        diarize_df = diarizer(audio, min_speakers, max_speakers)
        log(f"[diarize] diarized in {time.time()-t_diar:.1f}s "
            f"({len(diarize_df)} speaker turns) via {diarize_model_label}")
        result = whisperx.assign_word_speakers(diarize_df, result)
        diarized = True
        _free_cuda(diarizer)
        _vram("after diarize free")

    # --- Assemble output contract (identical to handler.py) -----------------
    segments = []
    speakers = set()
    for seg in result.get("segments", []):
        speaker = seg.get("speaker") or "UNKNOWN"
        if speaker != "UNKNOWN":
            speakers.add(speaker)
        segments.append({
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "speaker": speaker,
            "text": (seg.get("text") or "").strip(),
        })

    out = {
        "segments": segments,
        "language": detected_language,
        "duration": duration,
        "num_speakers": len(speakers),
        "model": model_name,
        "diarized": diarized,
    }
    log(f"[done] {len(segments)} segments, {len(speakers)} distinct speakers, "
        f"total {time.time()-t0:.1f}s")
    # diarize_model_label is returned alongside for the validation report.
    return out, diarize_model_label


def main():
    ap = argparse.ArgumentParser(description="Local validation of the diarize-batch pipeline.")
    ap.add_argument("audio", help="Path to the audio file to process.")
    ap.add_argument("--language", default="en", help="ISO language code (default en). Pass empty for autodetect.")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--no-diarize", action="store_true", help="Skip diarization.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--compute-type", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    _read_env_file(os.path.join(_REPO, ".env"))
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_AUTH_TOKEN") or ""

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        sys.exit(f"audio not found: {audio_path}")

    source_name = os.path.basename(audio_path)
    stem = os.path.splitext(source_name)[0]
    out_stem = os.path.join(_VALIDATION_DIR, stem)

    log("=" * 70)
    log(f"VALIDATE START source={source_name}")
    try:
        result, diar_label = run(
            audio_path,
            language=(args.language or None),
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            diarize=not args.no_diarize,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            hf_token=hf_token,
        )
    except Exception:
        log("VALIDATE FAILED:\n" + traceback.format_exc())
        raise

    files = render.write_outputs(result, out_stem, source_name)
    log(f"[render] wrote: {files}")
    log(f"VALIDATE OK source={source_name} diarization_model={diar_label} "
        f"segments={len(result['segments'])} speakers={result['num_speakers']} "
        f"duration={result['duration']:.1f}s language={result['language']}")
    log("=" * 70)

    # Compact machine-readable summary on the last stdout line.
    print(f"SUMMARY model={diar_label} segments={len(result['segments'])} "
          f"speakers={result['num_speakers']} duration={result['duration']:.1f} "
          f"language={result['language']} files={files}")


if __name__ == "__main__":
    main()
