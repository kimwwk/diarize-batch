"""Build-time model bake.

Downloads whisper large-v3 + the English alignment model + the 3.1-equivalent
diarization components into the image's HF/torch caches, so the pod needs NO
network volume and can therefore run on a cheap GPU in ANY datacenter. Runs on
CPU at build time (download only). Requires the HF_TOKEN build-arg for the gated
pyannote models.
"""
import os

import whisperx

import pipeline  # sets HF cache env + the shared GPU code (DEVICE=cpu at build)

HF = os.environ.get("HF_TOKEN", "")

print("[bake] whisper large-v3 ...", flush=True)
whisperx.load_model("large-v3", "cpu", compute_type="int8")

print("[bake] english alignment model ...", flush=True)
whisperx.load_align_model(language_code="en", device="cpu")

print("[bake] diarization 3.1 components (segmentation-3.0 + wespeaker) ...", flush=True)
pipeline._get_diarize_pipeline(HF)

print("[bake] DONE — models cached in the image", flush=True)
