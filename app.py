from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

import spherex_core as core
import report

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="SPHEREx Time-Lapse")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a single clear English message instead of raw Pydantic JSON."""
    messages = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", []) if x != "body"]
        field = ".".join(loc)
        msg = err.get("msg", "Invalid value")
        err_type = err.get("type", "")
        if field == "target" and (
            err_type == "string_too_short"
            or "at least 1 character" in msg
            or err_type == "missing"
        ):
            messages.append("Please enter RA, Dec coordinates.")
        elif field == "target":
            messages.append(msg)
        elif field:
            messages.append(f"{field}: {msg}")
        else:
            messages.append(msg)
    seen = set()
    unique = []
    for m in messages:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    detail = "; ".join(unique) if unique else "Invalid request"
    return JSONResponse(status_code=422, content={"detail": detail})

MAX_PARALLEL_JOBS = 2
_slots = threading.Semaphore(MAX_PARALLEL_JOBS)
_lock = threading.Lock()
JOBS: dict[str, dict] = {}
JOB_BY_KEY: dict[str, str] = {}
KEEP_JOBS = 30


PRESETS = [
    {"name": "WISE 0855", "ra": 133.79513, "dec": -7.24515, "fov": 0.1},
    {"name": "Barnard's Star", "ra": 269.44626, "dec": 4.76823, "fov": 0.1},
    {"name": "Epsilon Indi B", "ra": 331.09638, "dec": -56.80109, "fov": 0.05},
    {"name": "RW Aur", "ra": 76.95654, "dec": 30.40144, "fov": 0.05},
    {"name": "R Dra", "ra": 248.16792, "dec": 66.75500, "fov": 0.05},
]


class JobRequest(BaseModel):
    target: str = Field(..., min_length=1, max_length=120,
                        description="Coordinates 'RA, Dec' in degrees or sexagesimal")
    fov: float = Field(0.1, ge=0.01, le=0.3, description="Field of view, degrees")
    detector: int = Field(..., ge=1, le=6)


FINISHED = ("done", "error", "cancelled")


def _purge() -> None:
    if len(JOBS) <= KEEP_JOBS:
        return
    finished = sorted((j for j in JOBS.values() if j["status"] in FINISHED),
                      key=lambda j: j["created"])
    for j in finished[: len(JOBS) - KEEP_JOBS]:
        JOBS.pop(j["id"], None)
        if JOB_BY_KEY.get(j["key"]) == j["id"]:
            JOB_BY_KEY.pop(j["key"], None)


def _release_slot(job: dict) -> None:
    with _lock:
        if job["has_slot"]:
            job["has_slot"] = False
            _slots.release()


def _finish(job: dict, status: str, **extra) -> None:
    with _lock:
        if job["status"] == "cancelled":
            return
        job["status"] = status
        job.update(extra)
        
def _friendly_error(exc: Exception) -> str:
    """Convert technical exceptions into clear user-facing messages."""
    name = type(exc).__name__
    text = str(exc).strip()

    if name in ("DALServiceError", "HTTPError", "RemoteServiceError") or \
       "irsa.ipac.caltech.edu" in text or "TAP" in text:
        if any(code in text for code in ("502", "503", "504", "Proxy Error", "Gateway")):
            return ("The IRSA archive server is temporarily overloaded or unavailable "
                    "(error 502/503). Please try again in a few minutes.")
        if "timeout" in text.lower() or "timed out" in text.lower():
            return ("Timed out waiting for a response from the IRSA archive. "
                    "Please try again later.")
        return ("Could not reach the IRSA archive. "
                "The server may be temporarily unavailable. Please try again later.")

    if name in ("ValueError", "RuntimeError") and text:
        return text

    if name in ("ConnectionError", "TimeoutError", "OSError", "URLError"):
        return ("Network problem while downloading data. "
                "Check your internet connection or try again later.")

    short = text[:180] + ("…" if len(text) > 180 else "")
    return f"An error occurred during processing: {short}"

def _run(job: dict) -> None:
    def progress(stage: str, done: int, total: int) -> None:
        if job["cancel"]:
            raise core.Cancelled()
        job["stage"], job["done"], job["total"] = stage, done, total


    while not _slots.acquire(timeout=0.2):
        if job["cancel"]:
            return
    with _lock:
        if job["cancel"]:
            _slots.release()
            return
        job["has_slot"] = True
        job["status"] = "running"
    try:
        stack = core.run_pipeline(job["req"]["target"], job["req"]["fov"],
                                  job["req"]["detector"], progress)
        _finish(job, "done", stack=stack)
    except core.Cancelled:
        _finish(job, "cancelled")
    except Exception as e: 
        import logging
        logging.exception("Job %s failed", job["id"])
        _finish(job, "error", error=_friendly_error(e))
    finally:
        _release_slot(job)


@app.post("/api/jobs")
def create_job(req: JobRequest):

    try:
        norm = core.normalize_target(req.target)
    except ValueError as e:
        raise HTTPException(422, str(e))
    key = core.cache_key(norm, req.fov, req.detector)
    with _lock:
        jid = JOB_BY_KEY.get(key)
        if jid and JOBS.get(jid, {}).get("status") in ("queued", "running", "done"):
            return {"job_id": jid}
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "key": key, "req": {"target": norm, "fov": req.fov, "detector": req.detector},
               "status": "queued",
               "stage": "queued", "done": 0, "total": 0, "created": time.time(),
               "stack": None, "error": None, "cancel": False, "has_slot": False}
        JOBS[jid] = job
        JOB_BY_KEY[key] = jid
        _purge()
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"job_id": jid}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    with _lock:
        if job["status"] in FINISHED:
            return {"ok": True, "status": job["status"]}
        job["cancel"] = True
        job["status"] = "cancelled"
        job["stage"] = "cancelled"
    _release_slot(job)
    return {"ok": True, "status": "cancelled"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    out = {"id": job_id, "status": job["status"], "stage": job["stage"],
           "done": job["done"], "total": job["total"], "error": job["error"]}
    if job["status"] == "done":
        out["result"] = job["stack"].describe()
    return out


@app.get("/api/jobs/{job_id}/frame/{index}.png")
def job_frame(job_id: str, index: int):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Frames are not ready yet")
    try:
        png = job["stack"].png(index, "raw")
    except IndexError:
        raise HTTPException(404, "No such frame")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/presets")
def presets():
    return PRESETS


@app.get("/api/detectors")
def detectors():
    return [{"detector": d, "band_um": list(b)} for d, b in core.DETECTOR_BANDS.items()]


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


_FROGS_DIR = os.path.join(HERE, "kwakwaziabry")
if os.path.isdir(_FROGS_DIR):
    app.mount("/kwakwaziabry", StaticFiles(directory=_FROGS_DIR), name="kwakwaziabry")


class ReportRegion(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)
    w: float = Field(..., gt=0, le=1)
    h: float = Field(..., gt=0, le=1)


class ReportRequest(BaseModel):
    frame_number: int = Field(..., ge=1, description="Frame number, starting at 1")
    notes: str = Field("", max_length=report.MAX_NOTES_CHARS)
    region: Optional[ReportRegion] = None
    area_image: Optional[str] = Field(None, max_length=report.MAX_AREA_IMAGE_B64_CHARS,
                                      description="Frame PNG with a red outline, base64 / data URL")


@app.post("/api/jobs/{job_id}/report")
def job_report(job_id: str, req: ReportRequest):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Frames are not ready yet")
    try:
        pdf = report.make_report_pdf(job["stack"].describe(), req.frame_number, req.notes,
                                     req.region.model_dump() if req.region else None,
                                     req.area_image)
    except ValueError as e:
        raise HTTPException(422, str(e))
    filename = f"SPHEREx_report_{job_id}_frame_{req.frame_number}.pdf"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
