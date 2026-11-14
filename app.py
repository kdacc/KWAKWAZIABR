"""
Веб-бекенд SPHEREx Time-Lapse (FastAPI).

Запуск:  uvicorn app:app --reload
Відкрити: http://127.0.0.1:8000/

API:
  POST /api/jobs                      {target, fov, detector} -> {job_id}
  GET  /api/jobs/{id}                 стан, прогрес, список кадрів (коли готово)
  GET  /api/jobs/{id}/frame/{i}.png   ?mode=raw
  GET  /api/detectors                 довідка про 6 детекторів
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import spherex_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="SPHEREx Time-Lapse")

MAX_PARALLEL_JOBS = 2
_slots = threading.Semaphore(MAX_PARALLEL_JOBS)
_lock = threading.Lock()
JOBS: dict[str, dict] = {}
JOB_BY_KEY: dict[str, str] = {}
KEEP_JOBS = 30


class JobRequest(BaseModel):
    target: str = Field(..., min_length=1, max_length=120,
                        description="Назва об'єкта або 'RA, Dec' у градусах / сексагесимально")
    fov: float = Field(0.1, ge=0.01, le=0.3, description="Поле зору, градуси")
    detector: int = Field(..., ge=1, le=6)


def _purge() -> None:
    if len(JOBS) <= KEEP_JOBS:
        return
    done = sorted((j for j in JOBS.values() if j["status"] in ("done", "error")),
                  key=lambda j: j["created"])
    for j in done[: len(JOBS) - KEEP_JOBS]:
        JOBS.pop(j["id"], None)
        JOB_BY_KEY.pop(j["key"], None)


def _run(job: dict) -> None:
    def progress(stage: str, done: int, total: int) -> None:
        job["stage"], job["done"], job["total"] = stage, done, total

    with _slots:
        job["status"] = "running"
        try:
            job["stack"] = core.run_pipeline(job["req"]["target"], job["req"]["fov"],
                                             job["req"]["detector"], progress)
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"


@app.post("/api/jobs")
def create_job(req: JobRequest):
    key = core.cache_key(req.target, req.fov, req.detector)
    with _lock:
        jid = JOB_BY_KEY.get(key)
        if jid and JOBS.get(jid, {}).get("status") in ("queued", "running", "done"):
            return {"job_id": jid}
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "key": key, "req": req.model_dump(), "status": "queued",
               "stage": "queued", "done": 0, "total": 0, "created": time.time(),
               "stack": None, "error": None}
        JOBS[jid] = job
        JOB_BY_KEY[key] = jid
        _purge()
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Невідоме завдання")
    out = {"id": job_id, "status": job["status"], "stage": job["stage"],
           "done": job["done"], "total": job["total"], "error": job["error"]}
    if job["status"] == "done":
        out["result"] = job["stack"].describe()
    return out


@app.get("/api/jobs/{job_id}/frame/{index}.png")
def job_frame(job_id: str, index: int):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Кадри ще не готові")
    try:
        png = job["stack"].png(index, "raw")
    except IndexError:
        raise HTTPException(404, "Немає такого кадру")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/detectors")
def detectors():
    return [{"detector": d, "band_um": list(b)} for d, b in core.DETECTOR_BANDS.items()]


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))
