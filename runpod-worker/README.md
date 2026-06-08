# meetily diarize WhisperX worker (RunPod Serverless)

A **new, standalone** RunPod Serverless worker for batch, speaker-diarized
transcription with [WhisperX](https://github.com/m-bain/whisperX). It loads audio,
transcribes (Whisper `large-v3` by default), aligns word timestamps, runs
[`pyannote`](https://github.com/pyannote/pyannote-audio) speaker diarization, and
returns per-segment speaker labels.

> This image and the endpoint it powers are **brand new**. Deploying it does
> **not** modify any existing RunPod template, endpoint, or image.

---

## 1. HuggingFace prep (required for diarization)

Diarization uses gated `pyannote` models, so you need a HuggingFace token and
must accept two user agreements:

1. Create a **READ** token: <https://huggingface.co/settings/tokens>.
2. While logged in to the same HF account, accept the user agreement on **both**:
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

The token is supplied to the worker via the `HF_TOKEN` environment variable
(below). A per-request `input.hf_token` override is supported but not required in
the payload.

---

## 2. Build & push the image

Replace `<registry>` with your Docker registry / namespace (e.g. Docker Hub user).

```bash
# Plain build (diarization model downloads on first request):
docker build -t <registry>/meetily-diarize-whisperx-worker:latest .

# Optional: bake the gated diarization model into the image at build time.
# Requires the HF agreements above to be accepted.
docker build --build-arg HF_TOKEN=hf_xxx \
  -t <registry>/meetily-diarize-whisperx-worker:latest .

docker push <registry>/meetily-diarize-whisperx-worker:latest
```

The `HF_TOKEN` build-arg is **optional**: the build succeeds without it and the
model is fetched at runtime instead.

Alternatively, deploy directly from GitHub via RunPod's **Deploy from GitHub**
flow (point it at this directory; RunPod builds the Dockerfile for you).

---

## 3. Create the Serverless endpoint

In the RunPod console, create a **new** Serverless endpoint using the pushed image:

- **Network volume**: attach one. It mounts at `/runpod-volume` and is used for
  both the audio drop (`audio_path` like `/runpod-volume/<key>.flac`) and the
  model cache at `/runpod-volume/models` (the image already points `HF_HOME` /
  `PYANNOTE_CACHE` there).
- **Environment variable**: `HF_TOKEN = hf_xxx` (your READ token).
- **Max Workers**: `1`.
- **Idle timeout**: `~300s` (keeps a warm worker so cached models are reused).
- **FlashBoot**: **ON**.
- **GPU**: **24 GB+** — RTX 4090 or L40S recommended.

Because models are cached in module-level globals, a warm worker never reloads
them between requests.

---

## 4. Run the bundled test

Use [`test_input.json`](./test_input.json) (which targets
`/runpod-volume/test.flac` — drop a sample `test.flac` onto the volume first):

- **RunPod console**: open the endpoint's **Requests / Test** tab, paste the
  contents of `test_input.json`, and run.
- **API**:

  ```bash
  curl -s -X POST \
    https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
    -H "Authorization: Bearer <RUNPOD_API_KEY>" \
    -H "Content-Type: application/json" \
    -d @test_input.json
  ```

---

## Job contract

### Input (`event["input"]`)

| Field          | Type   | Default      | Notes |
|----------------|--------|--------------|-------|
| `audio_path`   | str    | —            | **Primary.** File already on the network volume, e.g. `/runpod-volume/<key>.flac`. Used when present. |
| `audio_base64` | str    | —            | Fallback for small files; decoded to a temp file. |
| `diarize`      | bool   | `true`       | Speaker diarization on/off. |
| `language`     | str    | `null`       | `null` = autodetect, else ISO code (e.g. `"en"`). |
| `min_speakers` | int    | `null`       | Optional hint forwarded to diarization. |
| `max_speakers` | int    | `null`       | Optional hint forwarded to diarization. |
| `model`        | str    | `"large-v3"` | Whisper model name. |
| `batch_size`   | int    | `8`          | Transcription batch size. |
| `compute_type` | str    | `"float16"`  | `"float16"` or `"float32"`. |
| `hf_token`     | str    | `HF_TOKEN`   | Optional per-request override of the env token. |

### Output

```json
{
  "segments": [
    {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00", "text": "..."}
  ],
  "language": "en",
  "duration": 1234.5,
  "num_speakers": 3,
  "model": "large-v3",
  "diarized": true
}
```

Every segment carries `start`, `end`, `text`, and `speaker` (`"UNKNOWN"` when no
speaker was assigned). `start`/`end` are rounded to 3 decimals. `num_speakers`
counts distinct non-`UNKNOWN` speakers. On any error the handler returns
`{"error": "<message>"}`.

---

## Pinned versions

| Package         | Version  |
|-----------------|----------|
| whisperx        | 3.8.5    |
| ctranslate2     | 4.7.1    |
| pyannote-audio  | 4.0.4    |
| runpod          | >= 1.7   |
| Base image      | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` |

`torch` / `torchaudio` come from the base image and are intentionally not pinned
in `requirements.txt`.
