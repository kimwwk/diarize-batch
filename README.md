<div align="center">

# 🎙️ diarize-batch

### Turn any meeting recording into a clean, speaker-labelled transcript — privately, on your own hardware, for pennies.

[![GitHub stars](https://img.shields.io/github/stars/kimwwk/diarize-batch?style=flat&logo=github)](https://github.com/kimwwk/diarize-batch/stargazers)
![Self-hosted](https://img.shields.io/badge/self--hosted-yes-success)
![Idle cost](https://img.shields.io/badge/idle_cost-%240-success)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-RunPod-673AB7)

**[Quickstart](#quickstart) • [How it works](#how-it-works) • [Speaker ID](#speaker-identification) • [Roadmap](#roadmap)**

<p align="center">
Drop an audio file into a folder (or onto a web page) and get back
<code>.md / .srt / .txt / .json</code> of who-said-what — plus an optional tag for
<i>which speaker is you</i>. The heavy GPU work (WhisperX <code>large-v3</code> + pyannote
diarization) runs on a RunPod GPU pod that spins up on demand and self-destructs when
idle, so it costs <b>$0 at rest</b> and your transcript never leaves your machine.
</p>

</div>

---

> **Stack:** WhisperX `large-v3` · pyannote `speaker-diarization-community-1` · on-demand RunPod GPU · resemblyzer voice-ID · Docker Compose

<details>
<summary><b>Table of Contents</b></summary>

- [Why it exists](#why-it-exists)
- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [Speaker identification](#speaker-identification)
- [Configuration](#configuration)
- [Layout](#layout)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)

</details>

## Why it exists

Cloud transcription is convenient but keeps your meetings on someone else's servers. A
self-hosted GPU is private but wasteful to leave idling. **diarize-batch splits the
difference** — local-first and private, but backed by on-demand cloud GPU you pay for
only while it's actually transcribing.

| | |
|---|---|
| 🔒 **Your data stays home** | Audio only briefly transits the GPU pod (unavoidable for cloud GPU) over a private, key-gated SSH tunnel — never a public endpoint. It's destroyed with the pod; the transcript is written *only* to your box. |
| 💸 **Pay only while it works** | No always-on server, no idle GPU, no storage volume. A pod is created per meeting and torn down minutes later — ≈ **$0.10–0.30** each, **$0** in between. |
| 🎯 **Drop-and-forget** | Put a file in, come back to a finished transcript. No dashboards, no babysitting — a simple FIFO queue. |
| 🗣️ **Knows who's who** | Optional voice matching tags which speaker is *you* (or any enrolled voice) across every meeting — in a side file that never alters the transcript. |

## How it works

```
  add audio.mp4   (web upload :8080  ·  or  cp into inbox/)
        │
        ▼
  ┌──────────────────────────────┐   SSH tunnel    ┌─────────────────────────────┐
  │  orchestrator  (your box)     │   (key-gated)   │  RunPod GPU pod (on-demand)  │
  │  watch inbox, one at a time   │ ──────────────► │  WhisperX large-v3 +         │
  │  ffmpeg → 16k mono FLAC        │   POST FLAC     │  pyannote diarization        │
  │  boot pod → tunnel → infer     │ ◄────────────── │  → [{start,end,speaker,text}]│
  │  render → tag speakers → clean │    segments     └─────────────────────────────┘
  │  write .md/.srt/.txt/.json     │          (pod deleted after POD_IDLE_MINUTES)
  └──────────────────────────────┘
        │
        ▼
  transcripts stay here (outbox/, served at :8080) — the pod, and the audio on it,
  are destroyed after each idle period
```

A small always-on **orchestrator** container watches a folder. When a file lands it
downmixes to 16 kHz mono, boots a GPU pod, reaches its private API over an SSH tunnel,
gets back diarized segments, renders the transcripts locally, optionally tags known
voices, then deletes the pod once the queue is idle.

## Quickstart

**Prerequisites:** a Hugging Face read token (accept the gated `pyannote/
speaker-diarization-community-1` and `pyannote/segmentation-3.0`) and a RunPod API key.
The GPU side uses a **public pre-built image** (`kimwwk/
meetily-diarize-whisperx-worker:pod-baked`, models baked in, no volume) — nothing to
build unless you want your own worker.

```bash
git clone https://github.com/kimwwk/diarize-batch && cd diarize-batch

# 1. an SSH keypair for the pod tunnel (stays local, gitignored)
mkdir -p secrets && ssh-keygen -t ed25519 -f secrets/pod_ed25519 -N "" -C "diarize-pod"

# 2. config
cp .env.example .env          # fill in RUNPOD_API_KEY and HF_TOKEN

# 3. (optional) tag yourself: drop a ~30s clip of your voice
#    at refs/<YourTag>.wav  (e.g. refs/USER_KW.wav) — gitignored, local only

# 4. go
docker compose up -d --build  # orchestrator + web page + cost watchdog
```

Then add a recording either way:

- 🌐 **Web (easiest):** open `http://<host>:8080/`, drag the file onto the drop zone. The
  same page lists every transcript.
- ⌨️ **CLI:** `cp ~/some-meeting.mp4 ./data/inbox/`.

Results appear in `./data/outbox/` as `<name>.md / .srt / .txt / .json` (+ an optional
`.speakers.json`), browsable at `http://<host>:8080/`. Processed input moves to `done/`;
failures move to `failed/` with a `.error.txt`. Files follow a
`YYYY-MM-DD_HHMM_<slug>.<ext>` naming convention (meeting date/time + a short slug).

> **Meetily users:** the recording is at `~/Music/meetily-recordings/<Meeting>/audio.mp4`
> (Windows; `~/Movies/...` macOS, `~/Documents/...` Linux). Upload that file.

## Speaker identification

If `refs/<Name>.<ext>` voice clips exist and `SPEAKER_ID=true` (default), each job also
writes `outbox/<stem>.speakers.json` mapping the diarized labels to people:

```json
{ "speakers": {
    "Speaker 4": { "name": "USER_KW",   "matched": true,  "similarity": 0.873 },
    "Speaker 1": { "name": "Speaker 1", "matched": false, "closest_similarity": 0.74 }
} }
```

- 🧠 CPU-only (resemblyzer), reusing the FLAC the orchestrator already made — no extra GPU.
- 🎯 A speaker is tagged only when its best match is ≥ `SPEAKER_MIN_SIM` (default 0.70),
  assigned **1:1** so one person can't be tagged on two speakers. Each file reports the
  similarity so you can tune the cutoff.
- ✋ It **never edits** the `.md/.srt/.txt/.json` — names live only in `.speakers.json`.

See [`refs/README.md`](refs/README.md) to enroll voices.

## Configuration

All via `.env` (copy from `.env.example`):

| Var | Default | Notes |
|-----|---------|-------|
| `LANGUAGE` | auto | Set `en` etc. to skip language detection (a bit faster/safer). |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | unset | Hint pyannote if you know the count — improves labels. |
| `INITIAL_PROMPT` | unset | Seed Whisper with names/acronyms so it spells them right. |
| `MODEL` | `large-v3` | Whisper model on the pod. |
| `COMPUTE_TYPE` | `float16` | `float32` on a big-VRAM GPU for max quality. |
| `POD_IDLE_MINUTES` | `5` | Delete the pod after this long idle (lower = cheaper, more cold boots). |
| `SPEAKER_ID` / `SPEAKER_MIN_SIM` | `true` / `0.70` | Speaker-id on/off and match cutoff. |
| `DELETE_INPUT_AFTER` | `false` | `true` deletes the input instead of moving it to `done/`. |

## Layout

```
diarize-batch/
├── orchestrator/      # local watcher: pod lifecycle + render + speaker-id
├── fileserver/        # web upload + transcript page (server.py)
├── runpod-worker/     # the GPU pod image (pre-built & public; build only to customize)
├── refs/              # speaker-id reference voices   (gitignored; you add these)
├── secrets/           # SSH keypair for the pod tunnel (gitignored; you add this)
└── docker-compose.yml # orchestrator + fileserver + watchdog
```

> `.env`, `secrets/`, and `refs/` are **gitignored** — they hold your keys and private
> audio and never belong in the repo.

## Roadmap

- [ ] **Fully-local inference.** An optional offline backend that runs WhisperX + pyannote
  on your own GPU (or a box on your LAN) instead of RunPod — zero cloud, zero per-meeting
  cost. The orchestrator already talks to the worker over a small HTTP contract, so the
  GPU backend is designed to be swappable.
- [ ] **Meeting intelligence.** Turn transcripts into structured outputs — summaries,
  action items, decisions, and follow-ups — generated locally alongside the raw transcript.

## Notes

- **Cost** ≈ $0.10–0.30 per meeting on a cheap 16–24 GB GPU; **$0 when idle**. The first
  job after idle pays a ~2–5 min pod cold-boot; transcription itself is ~40–50× real-time.
- **Restart-safe:** a container restart mid-job leaves the input in `inbox/` (only
  archived on success) to re-run; a `watchdog` container kills an orphaned pod if the
  orchestrator dies.

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) — word-level transcription on the worker.
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization.
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) — the CPU voice embeddings behind speaker-id.
- [RunPod](https://runpod.io) — the on-demand GPU.
- Built for the [Meetily](https://github.com/Zackriya-Solutions/meetily) recording workflow.

## Star History

[![Star History Chart](https://api.star-history.com/chart?repos=kimwwk/diarize-batch&type=Date)](https://star-history.com/#kimwwk/diarize-batch&Date)
