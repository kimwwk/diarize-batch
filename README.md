# diarize-batch

Drop a meeting recording in a folder → get a clean, **speaker-labelled** transcript
back, locally. Heavy GPU work (WhisperX `large-v3` + pyannote diarization) runs on a
**RunPod serverless endpoint**; the transcript is written only to your Proxmox box.

```
  you drop audio.mp4
        │
        ▼
  ┌──────────────────────────────┐        ┌────────────────────────────┐
  │  orchestrator  (this box)     │        │  RunPod serverless (NEW)    │
  │  watch inbox, one at a time   │ ─────► │  WhisperX large-v3 +        │
  │  ffmpeg → 16k mono FLAC        │ submit │  pyannote diarization       │
  │  upload to RunPod volume       │        │  returns [{start,end,        │
  │  poll → render → cleanup       │ ◄───── │   speaker,text}]            │
  │  write .md/.srt/.txt/.json     │ result └────────────────────────────┘
  └──────────────────────────────┘
        │
        ▼
  transcripts stay here (outbox/) — audio is deleted from RunPod after each job
```

- **One at a time, drop-and-forget.** Files queue in `inbox/` and process FIFO; a cold
  RunPod worker just means the queue waits.
- **Nothing existing is touched.** Brand-new Docker image + brand-new serverless
  endpoint. Your current RunPod pod/template is left alone.
- **Transcript stays local.** Audio transits RunPod's GPU (unavoidable for cloud GPU)
  but is deleted from the volume after each job; the transcript never leaves this box.

## Layout

```
diarize-batch/
├── runpod-worker/     # the NEW RunPod serverless image (build + deploy once)
├── orchestrator/      # the local watcher container (runs on Proxmox)
├── docker-compose.yml # runs the orchestrator
└── .env.example       # copy to .env and fill in
```

## One-time setup

### 1. Hugging Face (for pyannote diarization)
Create a **read** token at huggingface.co/settings/tokens, and accept the user
agreement on both gated models:
- `pyannote/speaker-diarization-community-1`
- `pyannote/segmentation-3.0`

### 2. Deploy the RunPod worker
Follow `runpod-worker/README.md`. In short: create a **network volume**, build/push
the image (or deploy-from-GitHub), then create a **serverless endpoint** with the
volume attached, `HF_TOKEN` set, **Max Workers = 1**, FlashBoot on, a 24 GB+ GPU.
Note the **endpoint ID**.

### 3. RunPod S3 API key
RunPod console → **Settings → S3 API Keys** → create one. Note the endpoint URL and
region for your volume's datacenter (e.g. `https://s3api-us-il-1.runpod.io/`,
`us-il-1`) and the **network volume ID** (this doubles as the S3 bucket name).

### 4. Configure the orchestrator
```bash
cd diarize-batch
cp .env.example .env
# edit .env: RunPod API key, endpoint ID, S3 keys/endpoint/region, volume ID
```

## Run

```bash
docker compose up -d --build      # starts the watcher
docker compose logs -f            # watch progress
```

Then just copy recordings into the inbox (the dir comes from `DATA_DIR` in `.env`,
default `./data`):

```bash
cp ~/some-meeting.mp4 ./data/inbox/
```

Results land in `./data/outbox/` as `some-meeting.md` / `.srt` / `.txt` / `.json`.
The processed input moves to `./data/done/`; anything that errors moves to
`./data/failed/` with a `.error.txt` beside it.

> Meetily users: the recording is at
> `~/Music/meetily-recordings/<Meeting>/audio.mp4` (Windows; `~/Movies/...` on macOS,
> `~/Documents/...` on Linux). Copy that file into `inbox/`. The saved audio is a
> mono mix, so diarization is done entirely by pyannote on the worker.

## Tuning (in `.env`)

| Var | Default | Notes |
|-----|---------|-------|
| `LANGUAGE` | auto | Set `en` etc. to skip language detection (a bit faster/safer). |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | unset | Hint pyannote if you know the count — improves labels. |
| `MODEL` | `large-v3` | Whisper model on the worker. |
| `COMPUTE_TYPE` | `float16` | Use `float32` on a big-VRAM GPU for max quality. |
| `KEEP_REMOTE_AUDIO` | `false` | `true` keeps the uploaded audio on the volume. |
| `DELETE_INPUT_AFTER` | `false` | `true` deletes the input instead of moving to `done/`. |

## Notes
- Cost is roughly **$0.10–0.30 per meeting** on a 24 GB serverless GPU; the endpoint
  scales to zero between jobs.
- Restart-safe: if the container restarts mid-job the input is still in `inbox/`
  (it's only archived on success) and will be re-run.
