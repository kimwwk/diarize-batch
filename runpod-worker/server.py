"""FastAPI transcription server for the RunPod POD.

Binds to 127.0.0.1 ONLY — it is never exposed on RunPod's public proxy. The
orchestrator reaches it through an SSH local-forward (ssh -L 8000:localhost:8000),
so the transcription API is private; only key-gated sshd is public.

Endpoints:
  GET  /health      -> {ready, device, gpu, asr_loaded, diar_loaded}
  POST /inference   -> multipart 'file' (+ form opts) -> output-contract dict
"""
import os
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

import pipeline

app = FastAPI(title="diarize-batch pod", version="1.0")


@app.on_event("startup")
def _startup():
    # Warm all models so the first request is fast. Best-effort: if it fails
    # (e.g. models not yet on the volume) the first /inference will load them.
    try:
        pipeline.warmup()
        print("[startup] models warmed", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[startup] warmup skipped: {e}", flush=True)


@app.get("/health")
def health():
    gpu = None
    if pipeline.gpu_ready():
        import torch
        gpu = torch.cuda.get_device_name(0)
    return {
        "ready": pipeline.gpu_ready(),
        "device": pipeline.DEVICE,
        "gpu": gpu,
        "asr_loaded": bool(pipeline._asr_models),
        "diar_loaded": pipeline._diarize_pipeline is not None,
    }


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    language: str = Form("en"),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    compute_type: str = Form("float16"),
    batch_size: int = Form(8),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await file.read())
        tmp.flush(); tmp.close()
        result = pipeline.run_pipeline(
            tmp.name, diarize=diarize, language=(language or None),
            min_speakers=min_speakers, max_speakers=max_speakers,
            initial_prompt=(initial_prompt or None), compute_type=compute_type,
            batch_size=batch_size, progress=lambda m: print(f"[infer] {m}", flush=True),
        )
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()[:2000]})
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
