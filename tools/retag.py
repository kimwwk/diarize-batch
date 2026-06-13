#!/usr/bin/env python3
"""Re-run ONLY the speaker-id voice-match on a finished meeting (no GPU).

Regenerates <stem>.speakers.json from the meeting's existing transcript segments
+ archived audio, matched against the CURRENT refs/. Use it after enrolling a new
voice to see who now gets auto-recognized — without paying to re-diarize.

It reuses the orchestrator's own speaker_id module, so it must run in that image
(resemblyzer + ffmpeg + /app modules are there):

  docker run --rm \
    -v /opt/diarize-batch/data:/data \
    -v /opt/diarize-batch/refs:/refs \
    -v /opt/diarize-batch/tools:/tools:ro \
    meetily-diarize-orchestrator:latest python /tools/retag.py <stem>
"""
import json
import os
import sys

sys.path.insert(0, "/app")  # config, speaker_id from the orchestrator image
import speaker_id  # noqa: E402

OUTBOX = os.environ.get("OUTBOX_DIR", "/data/outbox")
DONE = os.environ.get("DONE_DIR", "/data/done")
AUDIO_EXTS = (".m4a", ".mp4", ".wav", ".flac", ".mp3", ".aac",
              ".ogg", ".webm", ".opus", ".mkv")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: retag.py <stem>")
    stem = sys.argv[1]
    jpath = os.path.join(OUTBOX, stem + ".json")
    if not os.path.isfile(jpath):
        sys.exit(f"no transcript JSON: {jpath}")
    data = json.load(open(jpath, encoding="utf-8"))
    segments = data.get("segments", [])
    audio = next((os.path.join(DONE, stem + e) for e in AUDIO_EXTS
                  if os.path.isfile(os.path.join(DONE, stem + e))), None)
    if not audio:
        sys.exit(f"no archived audio in {DONE} for {stem}")

    out_stem = os.path.join(OUTBOX, stem)
    path = speaker_id.write_speaker_map(audio, segments, out_stem,
                                        data.get("source", stem))
    if not path:
        sys.exit("speaker-id produced nothing "
                 "(no refs / resemblyzer missing / SPEAKER_ID off)")

    spk = json.load(open(path, encoding="utf-8")).get("speakers", {})
    matched = [i for i in spk.values() if i.get("matched")]
    print(f"matched {len(matched)}/{len(spk)} speakers "
          f"(min_sim default 0.70):")
    for lbl, info in sorted(spk.items(),
                            key=lambda kv: not kv[1].get("matched")):
        tag = "MATCH" if info.get("matched") else "  -  "
        sim = info.get("similarity", info.get("closest_similarity"))
        who = info.get("name") if info.get("matched") else info.get("closest_ref")
        print(f"  {tag} {lbl:<11} raw={info.get('raw'):<12} {who}  sim={sim}")
    print("wrote", path)


if __name__ == "__main__":
    main()
