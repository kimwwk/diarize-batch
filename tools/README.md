# tools/local_validate.py

Local stand-in for the (not-yet-deployed) RunPod serverless worker. Runs the
**same** WhisperX transcribe + align + pyannote-diarize pipeline as
`runpod-worker/handler.py` on the local GPU, produces the **same** output-contract
dict, and renders `.md/.json/.srt/.txt` via `orchestrator/render.py`.

Use it to validate transcription/diarization quality and the render pipeline
without paying for / waiting on the RunPod endpoint.

## Run

```bash
# from the repo root, with the project venv (.venv, python 3.11)
.venv/bin/python tools/local_validate.py /path/to/audio.m4a --language en
# optional: --min-speakers N --max-speakers N --no-diarize --compute-type float16
```

Outputs land in `_validation/<stem>.{md,json,srt,txt}` and a run log is written to
`_validation/validate.log`. The HF token is read from the repo `.env`
(`HF_TOKEN` / `HF_AUTH_TOKEN`).

## Differences from handler.py (forced by an 8 GB laptop GPU + HF gating)

- **Stage-wise VRAM freeing.** handler.py caches every model in module globals for
  warm reuse; here each stage (ASR → align → diarize) is loaded, run, then deleted
  with `gc.collect()` + `torch.cuda.empty_cache()` before the next, so peak VRAM
  stays ~= one model. (Measured peak ≈ 6.3 GiB for large-v3 on an 8 GiB card.)
- **compute_type float16** (handler default is also float16; float32 OOMs here).
- **Diarization model fallback.** handler.py uses whisperx's default
  `pyannote/speaker-diarization-community-1`. That repo is gated on this HF account
  (403), so the tool falls back to a `speaker-diarization-3.1`-equivalent pipeline
  built from accessible components: `pyannote/segmentation-3.0` +
  `pyannote/wespeaker-voxceleb-resnet34-LM` with AgglomerativeClustering and the
  frozen 3.1 hyper-parameters. pyannote-audio 4.0.4 eagerly downloads community-1's
  PLDA in `SpeakerDiarization.__init__` even when it's unused (Agglomerative); the
  tool neutralizes just that one eager fetch. The diarization model actually used is
  logged and printed in the run summary.
