"""Shared transcription + diarization pipeline.

WhisperX large-v3 + a 3.1-equivalent pyannote diarizer (community-1 is avoided —
pyannote-audio 4.0.4 hangs 20+ min on its eager PLDA download). Used by BOTH the
pod FastAPI server (server.py) and the legacy serverless handler (handler.py), so
the GPU logic lives in exactly one place. Models are cached in module globals.
"""
import os
import time

import torch
import whisperx

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Frozen pyannote/speaker-diarization-3.1 definition, rebuilt from ungated parts.
_SD31_SEGMENTATION = "pyannote/segmentation-3.0"
_SD31_EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"
_SD31_PARAMS = {
    "clustering": {"method": "centroid", "min_cluster_size": 12,
                   "threshold": 0.7045654963945799},
    "segmentation": {"min_duration_off": 0.0},
}

_asr_models: dict = {}
_align_models: dict = {}
_diarize_pipeline = None


def gpu_ready() -> bool:
    return DEVICE == "cuda" and torch.cuda.is_available()


def _get_asr_model(model_name, compute_type, initial_prompt=None):
    key = (model_name, compute_type, initial_prompt)
    if key not in _asr_models:
        print(f"[load] ASR model={model_name} compute_type={compute_type} device={DEVICE}", flush=True)
        asr_options = {"initial_prompt": initial_prompt} if initial_prompt else None
        _asr_models[key] = whisperx.load_model(model_name, DEVICE, compute_type=compute_type, asr_options=asr_options)
    return _asr_models[key]


def _get_align_model(language_code):
    if language_code not in _align_models:
        print(f"[load] align model language={language_code} device={DEVICE}", flush=True)
        _align_models[language_code] = whisperx.load_align_model(language_code=language_code, device=DEVICE)
    return _align_models[language_code]


def _get_diarize_pipeline(hf_token):
    """3.1-equivalent SpeakerDiarization from ungated parts, with pyannote 4.0.4's
    eager community-1 PLDA download neutralized (it hangs 20+ min otherwise)."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        import pyannote.audio.pipelines.speaker_diarization as _sd_mod
        from pyannote.audio.pipelines import SpeakerDiarization
        print(f"[load] diarization pipeline (3.1-equivalent) device={DEVICE}", flush=True)
        _orig = _sd_mod.get_plda
        _sd_mod.get_plda = lambda *a, **k: None
        try:
            sd = SpeakerDiarization(
                segmentation=_SD31_SEGMENTATION, embedding=_SD31_EMBEDDING,
                embedding_exclude_overlap=True, clustering="AgglomerativeClustering",
                segmentation_batch_size=32, embedding_batch_size=32, token=hf_token,
            )
            sd.instantiate(_SD31_PARAMS)
            sd.to(torch.device(DEVICE))
        finally:
            _sd_mod.get_plda = _orig
        _diarize_pipeline = sd
    return _diarize_pipeline


def warmup(hf_token=None, model_name="large-v3", compute_type="float16", language="en"):
    """Pre-load every model so the first request is fast (best-effort)."""
    _get_asr_model(model_name, compute_type)
    _get_align_model(language)
    _get_diarize_pipeline(hf_token or ENV_HF_TOKEN)


def run_pipeline(audio_path, *, diarize=True, language=None, min_speakers=None,
                 max_speakers=None, model_name="large-v3", batch_size=8,
                 compute_type="float16", initial_prompt=None, hf_token=None,
                 progress=None) -> dict:
    """Run the full pipeline on a local audio file. Returns the output-contract
    dict; raises on error (the caller wraps). `progress` is an optional cb(str)."""
    def _prog(m):
        if progress:
            try: progress(m)
            except Exception: pass

    t0 = time.time(); timings = {}
    if not gpu_ready():
        raise RuntimeError(f"GPU unavailable (device={DEVICE}, cuda={torch.cuda.is_available()})")
    gpu_name = torch.cuda.get_device_name(0)
    hf_token = hf_token or ENV_HF_TOKEN
    compute_type = compute_type if compute_type in ("float16", "float32") else "float16"

    audio = whisperx.load_audio(audio_path)
    duration = round(len(audio) / SAMPLE_RATE, 3)

    _prog(f"loading ASR {model_name} ({compute_type})"); t = time.time()
    asr = _get_asr_model(model_name, compute_type, initial_prompt)
    timings["load_asr_s"] = round(time.time() - t, 1)
    _prog(f"transcribing {duration:.0f}s"); t = time.time()
    result = asr.transcribe(audio, batch_size=batch_size, language=language)
    timings["transcribe_s"] = round(time.time() - t, 1)
    detected_language = result.get("language") or language or "en"

    _prog("aligning"); t = time.time()
    try:
        am, md = _get_align_model(detected_language)
        result = whisperx.align(result["segments"], am, md, audio, DEVICE, return_char_alignments=False)
        timings["align_s"] = round(time.time() - t, 1)
    except Exception as e:
        timings["align_s"] = round(time.time() - t, 1); _prog(f"align skipped: {e}")

    diarized = False
    if diarize:
        if not hf_token:
            raise ValueError("diarize=True but no HF token (set HF_TOKEN).")
        _prog("loading diarizer"); t = time.time()
        sd = _get_diarize_pipeline(hf_token); timings["load_diarize_s"] = round(time.time() - t, 1)
        _prog("diarizing"); t = time.time()
        import pandas as pd
        waveform = torch.from_numpy(audio[None, :])
        dkw = {}
        if min_speakers is not None: dkw["min_speakers"] = int(min_speakers)
        if max_speakers is not None: dkw["max_speakers"] = int(max_speakers)
        out_d = sd({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **dkw)
        ann = out_d.speaker_diarization
        df = pd.DataFrame(ann.itertracks(yield_label=True), columns=["segment", "label", "speaker"])
        df["start"] = df["segment"].apply(lambda s: s.start)
        df["end"] = df["segment"].apply(lambda s: s.end)
        result = whisperx.assign_word_speakers(df, result)
        timings["diarize_s"] = round(time.time() - t, 1); diarized = True

    segments, speakers = [], set()
    for seg in result.get("segments", []):
        sp = seg.get("speaker") or "UNKNOWN"
        if sp != "UNKNOWN": speakers.add(sp)
        segments.append({
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "speaker": sp, "text": (seg.get("text") or "").strip(),
        })
    timings["total_s"] = round(time.time() - t0, 1)
    _prog(f"done {len(segments)} segs {len(speakers)} spk {timings['total_s']}s")
    return {
        "segments": segments, "language": detected_language, "duration": duration,
        "num_speakers": len(speakers), "model": model_name, "diarized": diarized,
        "device": "cuda", "gpu": gpu_name, "timings": timings,
    }
