# refs/ — speaker-id reference voices

Drop one reference clip per person here and the orchestrator will tag that
person in an extra `<stem>.speakers.json` next to each transcript (the
`.md/.srt/.txt/.json` files are never modified).

## How to enroll someone

Name the file after the label you want — `refs/<Name>.<ext>`:

```
refs/USER_KW.wav        ->  speakers matching this voice are tagged "USER_KW"
refs/Some-Guest.m4a     ->  ... tagged "Some-Guest"
```

- A clean ~20–40 s clip of just that person speaking works well.
- Any audio format ffmpeg can read; it's downmixed to 16 kHz mono internally.
- Matching is cosine similarity of CPU voice embeddings (resemblyzer); a speaker
  is tagged only when its best match is ≥ `SPEAKER_MIN_SIM` (default 0.70).
  Each `.speakers.json` includes the similarity so you can tune the cutoff.

## Privacy

**The audio files here are gitignored** (`refs/*`, except this README) — they are
private and must never be committed or pushed. They live only on the LXC.
Disable the whole feature with `SPEAKER_ID=false`.
