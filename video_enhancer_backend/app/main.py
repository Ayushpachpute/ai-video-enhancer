import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from . import utils, workers

app = FastAPI(title="AI Video Enhancer Backend")

# Simple in-memory job store
JOBS: Dict[str, dict] = {}

# CORS for local dev; adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"

# Serve processed results as static files so frontend can access resultUrl returned by jobs
app.mount("/results", StaticFiles(directory=utils.RESULTS_DIR, html=False, check_dir=False), name="results")


@app.on_event("startup")
async def on_startup():
    # Ensure required directories and ffmpeg availability (auto-download on Windows)
    utils.ensure_dirs()
    try:
        utils.ensure_ffmpeg()
    except Exception:
        # Non-fatal: backend can still run; upload/processing will fail if ffmpeg is needed and missing
        pass


@app.post(f"{API_PREFIX}/upload")
async def upload_video(file: UploadFile = File(...), model: Optional[str] = Form(None)):
    # Validate input
    if file.content_type not in ("video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/x-matroska"):
        raise HTTPException(status_code=400, detail="Unsupported format. Use MP4, MOV, MKV, AVI or WEBM.")

    # Persist upload
    job_id = uuid.uuid4().hex
    utils.ensure_dirs()
    upload_path = utils.upload_path(job_id, file.filename)

    try:
        # Save to disk
        with open(upload_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()

    # Initialize job state
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "uploadPath": upload_path,
        "resultPath": None,
        "resultUrl": None,
        "canceled": False,
        "error": None,
        "model": model or "realesrgan-x4plus",
    }

    # Launch background processing
    asyncio.create_task(workers.process_job(job_id, JOBS))

    return {"jobId": job_id}


@app.get(f"{API_PREFIX}/status/{{job_id}}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Build client-friendly payload
    payload = {
        "id": job["id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "resultUrl": job.get("resultUrl"),
        "model": job.get("model"),
        "processedFrames": job.get("processedFrames"),
        "totalFrames": job.get("totalFrames"),
        "avgMsPerFrame": job.get("avgMsPerFrame"),
    }
    return JSONResponse(payload)


# Optional: query-based status for frontend that calls /api/status?jobId=
@app.get(f"{API_PREFIX}/status")
async def get_status_query(jobId: Optional[str] = None):
    if not jobId:
        raise HTTPException(status_code=400, detail="Missing jobId")
    return await get_status(jobId)


# Frontend calls /api/enhance to start processing; our upload already started it.
# We accept and ensure the job exists, optionally update model, and return ok.
@app.post(f"{API_PREFIX}/enhance")
async def enhance_job(payload: Dict[str, Optional[str]]):
    job_id = payload.get("jobId") if payload else None
    model = payload.get("model") if payload else None
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing jobId")
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if model:
        job["model"] = model
    # If somehow not started, start now
    if job.get("status") in ("queued", "pending") and not job.get("_started"):
        job["_started"] = True
        asyncio.create_task(workers.process_job(job_id, JOBS))
    return {"ok": True}


@app.get(f"{API_PREFIX}/status/stream/{{job_id}}")
async def stream_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator() -> AsyncGenerator[bytes, None]:
        last_snapshot: Optional[str] = None
        while True:
            await asyncio.sleep(0.5)
            job = JOBS.get(job_id)
            if not job:
                break
            # Serialize
            data = {
                "id": job["id"],
                "status": job["status"],
                "progress": job.get("progress", 0),
                "message": job.get("message", ""),
                "resultUrl": job.get("resultUrl"),
            }
            snap = json.dumps(data, separators=(",", ":"))
            if snap != last_snapshot:
                last_snapshot = snap
                yield f"data: {snap}\n\n".encode("utf-8")
            # End stream on terminal states
            if data["status"] in ("completed", "failed", "canceled"):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete(f"{API_PREFIX}/job/{{job_id}}")
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["canceled"] = True
    job["message"] = "Cancel requested"
    return {"ok": True}


# Serve the SPA from /web at project root (one level above BACKEND_ROOT)
WEB_DIR = utils.BACKEND_ROOT.parent / "web"
if WEB_DIR.exists():
    # Mount after API routes so API keeps priority
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:
    @app.get("/")
    async def root():
        return {"ok": True, "service": "ai-video-enhancer-backend"}


# Dev entrypoint: uvicorn app.main:app --reload --port 3000
