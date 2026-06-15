# tools/fireflies_import.py

Pull a **Fireflies.ai** meeting (transcript + audio) into this instance **without
the GPU pipeline**. It fetches the transcript from the Fireflies GraphQL API,
renders it through the project's own `orchestrator/render.py` (byte-identical to a
real pipeline run), seeds the speaker name map, drops the AI summary into a
`.reflection.md` side-panel, and downloads the audio. The finished files land in
`data/outbox/` + `data/done/`, so the orchestrator never fires — **no pod, no GPU,
$0**. Stdlib only.

## What maps across

| Fireflies (GraphQL) | → | this instance |
|---|---|---|
| `sentences[]` (`speaker_id`, `text`, `start_time`/`end_time` in **seconds**) | → | `data/outbox/<stem>.{json,md,srt,txt}` (segments `SPEAKER_0N`) |
| `speakers[]` (`id` → `name`) | → | manual name map in `data/db/speakers.db` |
| `summary` (overview / action_items / keywords) | → | `data/outbox/<stem>.reflection.md` (viewer side panel) |
| audio (signed CDN URL — see below) | → | `data/done/<stem>.mp3` (click-to-play) |

## Run (on the instance)

```bash
ssh diarize-ct 'cd /opt/diarize-batch && \
  python3 tools/fireflies_import.py "<fireflies-link-or-id>" --audio-url "<signed cdn url>"'
```

- `<fireflies-link-or-id>` — a `https://app.fireflies.ai/view/...::<id>` link, a
  `/view/<id>` link, or a bare transcript id. The id is parsed out automatically.
- The API key is read from **`secrets/fireflies.key`** on the instance (never stored
  in the repo or on the WSL box). Override with `--api-key` or `$FIREFLIES_API_KEY`.
- `--dry-run` renders into `./_fireflies_preview/` and touches nothing live — use it
  to eyeball quality first.

Useful flags: `--slug <name>` (the Fireflies title is often just the account name,
so set the person/audience, e.g. `--slug sileon-surv7x`), `--time HHMM` (the API
date is **UTC**; pass the owner's local clock time — the transcript text often
reveals the timezone), `--date YYYY-MM-DD`, `--no-names`, `--model <label>`.

The resulting meeting is live at `http://192.168.1.129:8080/view/<stem>`.

## The audio caveat (important)

The free API tier exposes the transcript but **not** `audio_url`/`video_url` (those
are `pro_or_higher`, HTTP 403). Audio is still retrievable from the **signed
CloudFront URL** the web player uses:

```
https://cdn.fireflies.ai/<TRANSCRIPT_ID>/audio.mp3?Expires=...&Signature=...&Key-Pair-Id=...
```

The transcript id == the audio asset id. This URL is **time-limited** and is **not**
returned by the API — grab it from the logged-in web session (DevTools → Network →
the `audio.mp3` request → Copy URL) and pass it to `--audio-url`. `--audio-url` also
accepts a **local file path** if you already downloaded it.

Without `--audio-url` the transcript still imports perfectly; you just lose
click-to-play until you re-run with audio (or `scp` an mp3 into `data/done/<stem>.mp3`).

## The API key on the instance

```bash
# stored once, on the instance only (gitignored dir, 0600):
ssh diarize-ct 'umask 077 && printf "%s" "<key>" > /opt/diarize-batch/secrets/fireflies.key'
```

Currently holds the **kimwong.wwk@gmail.com** account key. To use a different
Fireflies account, overwrite that file (or pass `--api-key`). The link you import
must belong to the same account as the key.
