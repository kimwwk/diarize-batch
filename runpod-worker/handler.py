"""
RunPod Serverless handler — batch speaker-diarized transcription with WhisperX.

Job contract (FIXED — the local orchestrator is built to match this exactly):

INPUT  (event["input"]):
    audio_path     str   PRIMARY. Path to a file already on the attached network
                          volume, e.g. "/runpod-volume/<key>.flac". Used when present.
    audio_base64   str   Fallback for small files. Decoded to a temp file.
    diarize        bool  Default True.
    language       str   None = autodetect, else ISO code e.g. "en".
    min_speakers   int   Optional hint passed to diarization.
    max_speakers   int   Optional hint passed to diarization.
    model          str   Default "large-v3".
    batch_size     int   Default 8.
    compute_type   str   Default "float16"; "float32" also allowed.
    initial_prompt str   Optional. Seeds Whisper with domain vocab/names to bias
                          spelling (e.g. product names, acronyms).
    hf_token       str   Optional override for the HF_TOKEN env var.

OUTPUT (handler return):
    {
      "segments": [ {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00", "text": "..."} ],
      "language": "en",
      "duration": 1234.5,
      "num_speakers": 3,
      "model": "large-v3",
      "diarized": true
    }
    Every segment carries start, end, text, speaker ("UNKNOWN" when no speaker was
    assigned). start/end are rounded to 3 decimals. num_speakers counts distinct
    non-UNKNOWN speakers. On any failure the handler returns {"error": "<message>"}.

Models are cached in module-level globals so warm invocations reuse them and never
reload per request.
"""
import base64
import os
import tempfile
import time
import traceback

import runpod
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


# --- Constants ---------------------------------------------------------------
SAMPLE_RATE = 16000  # WhisperX loads/resamples all audio to 16 kHz mono.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HF token comes from the worker environment; input.hf_token can override it.
ENV_HF_TOKEN = os.environ.get("HF_TOKEN", "")


# --- Module-level caches (reused across warm invocations) --------------------
# Whisper ASR models keyed by (model_name, compute_type). The same worker may be
# asked for different model/compute combinations across its lifetime.
_asr_models: dict = {}
# Alignment models keyed by language code (each language has its own align model).
_align_models: dict = {}
# Diarization pipeline — a single instance is enough; it is language-agnostic.
_diarize_pipeline = None


def _get_asr_model(model_name: str, compute_type: str, initial_prompt=None):
    """Load (or fetch from cache) the WhisperX ASR model.

    initial_prompt is part of the cache key because it is baked into the model's
    asr_options at load time."""
    key = (model_name, compute_type, initial_prompt)
    if key not in _asr_models:
        print(f"[load] ASR model={model_name} compute_type={compute_type} device={DEVICE}")
        asr_options = {"initial_prompt": initial_prompt} if initial_prompt else None
        _asr_models[key] = whisperx.load_model(
            model_name,
            DEVICE,
            compute_type=compute_type,
            asr_options=asr_options,
        )
    return _asr_models[key]


def _get_align_model(language_code: str):
    """Load (or fetch from cache) the alignment model for a language."""
    if language_code not in _align_models:
        print(f"[load] align model language={language_code} device={DEVICE}")
        align_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=DEVICE,
        )
        _align_models[language_code] = (align_model, metadata)
    return _align_models[language_code]


def _get_diarize_pipeline(hf_token: str):
    """Load (or fetch from cache) the diarization pipeline.

    Uses the default gated model pyannote/speaker-diarization-community-1.
    """
    global _diarize_pipeline
    if _diarize_pipeline is None:
        print(f"[load] diarization pipeline device={DEVICE}")
        _diarize_pipeline = DiarizationPipeline(token=hf_token, device=DEVICE)
    return _diarize_pipeline


def _resolve_audio_path(inp: dict) -> tuple:
    """Return (audio_path, temp_path_to_cleanup_or_None).

    audio_path on the network volume is preferred. When it is absent we decode
    audio_base64 into a NamedTemporaryFile that the caller must clean up.
    """
    audio_path = inp.get("audio_path")
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio_path does not exist: {audio_path}")
        return audio_path, None

    audio_b64 = inp.get("audio_base64")
    if not audio_b64:
        raise ValueError("Provide either 'audio_path' or 'audio_base64'.")

    # Decode the base64 payload to a temp file on local disk.
    tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    try:
        tmp.write(base64.b64decode(audio_b64))
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name, tmp.name


def handler(job):
    inp = job.get("input", {}) or {}
    temp_path = None
    t0 = time.time()
    timings = {}

    def _prog(msg):
        try:
            runpod.serverless.progress_update(job, msg)
        except Exception:
            pass
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    try:
        # --- Fail fast if the container did not get a GPU ------------------
        # A silent CPU fallback runs large-v3 for well over an hour and bills
        # the whole time; refuse so the caller gets an instant, clear error.
        if DEVICE != "cuda" or not torch.cuda.is_available():
            return {
                "error": "GPU unavailable on this worker — refusing to run on CPU.",
                "device": DEVICE,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_device_count": torch.cuda.device_count(),
            }
        gpu_name = torch.cuda.get_device_name(0)
        _prog(f"start device=cuda gpu={gpu_name}")

        # --- Parse / default the contract inputs ---------------------------
        diarize = bool(inp.get("diarize", True))
        # language: None => autodetect. Keep None explicitly when not provided.
        language = inp.get("language") or None
        min_speakers = inp.get("min_speakers")
        max_speakers = inp.get("max_speakers")
        model_name = inp.get("model") or "large-v3"
        batch_size = int(inp.get("batch_size") or 8)
        compute_type = inp.get("compute_type") or "float16"
        if compute_type not in ("float16", "float32"):
            compute_type = "float16"
        initial_prompt = inp.get("initial_prompt") or None
        hf_token = inp.get("hf_token") or ENV_HF_TOKEN

        # --- Resolve audio source ------------------------------------------
        audio_path, temp_path = _resolve_audio_path(inp)

        # --- Load audio (16 kHz mono float waveform) -----------------------
        audio = whisperx.load_audio(audio_path)
        duration = round(len(audio) / SAMPLE_RATE, 3)

        # --- Transcribe -----------------------------------------------------
        _prog(f"loading ASR model={model_name} ({compute_type})")
        t = time.time()
        asr_model = _get_asr_model(model_name, compute_type, initial_prompt)
        timings["load_asr_s"] = round(time.time() - t, 1)
        _prog(f"transcribing (audio {duration:.0f}s, batch={batch_size})")
        t = time.time()
        # language=None lets WhisperX autodetect; an ISO code forces a language.
        result = asr_model.transcribe(audio, batch_size=batch_size, language=language)
        timings["transcribe_s"] = round(time.time() - t, 1)
        detected_language = result.get("language") or language or "en"
        _prog(f"transcribed in {timings['transcribe_s']}s, language={detected_language}")

        # --- Align (word-level timestamps) ---------------------------------
        # Alignment is required for accurate per-word speaker assignment.
        _prog("aligning word timestamps")
        t = time.time()
        try:
            align_model, metadata = _get_align_model(detected_language)
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
            timings["align_s"] = round(time.time() - t, 1)
            _prog(f"aligned in {timings['align_s']}s")
        except Exception as align_err:
            # No alignment model for this language: continue with segment-level
            # timestamps so transcription still succeeds.
            timings["align_s"] = round(time.time() - t, 1)
            _prog(f"alignment skipped: {align_err}")

        # --- Diarize + assign speakers -------------------------------------
        diarized = False
        if diarize:
            if not hf_token:
                raise ValueError(
                    "diarize=true but no HF token available "
                    "(set HF_TOKEN env var or pass input.hf_token)."
                )
            _prog("loading diarization pipeline")
            t = time.time()
            pipeline = _get_diarize_pipeline(hf_token)
            timings["load_diarize_s"] = round(time.time() - t, 1)
            # Only forward speaker-count hints when actually provided.
            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = int(min_speakers)
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = int(max_speakers)
            _prog("diarizing")
            t = time.time()
            diarize_segments = pipeline(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            timings["diarize_s"] = round(time.time() - t, 1)
            _prog(f"diarized in {timings['diarize_s']}s")
            diarized = True

        # --- Assemble output contract --------------------------------------
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

        timings["total_s"] = round(time.time() - t0, 1)
        _prog(f"done: {len(segments)} segments, {len(speakers)} speakers, {timings['total_s']}s")
        return {
            "segments": segments,
            "language": detected_language,
            "duration": duration,
            "num_speakers": len(speakers),
            "model": model_name,
            "diarized": diarized,
            "device": "cuda",
            "gpu": gpu_name,
            "timings": timings,
        }

    except Exception as exc:  # noqa: BLE001 — surface any failure to the caller.
        return {"error": f"{exc}\n{traceback.format_exc()}"}

    finally:
        # Remove the temp file created from audio_base64 (never a volume path).
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
