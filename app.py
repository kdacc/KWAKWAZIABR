"""
Веб-бекенд SPHEREx Time-Lapse (FastAPI).

Запуск:  uvicorn app:app --reload
Відкрити: http://127.0.0.1:8000/

API:
  POST /api/jobs                      {target, fov, detector} -> {job_id}
  GET  /api/jobs/{id}                 стан, прогрес, список кадрів (коли готово)
  GET  /api/jobs/{id}/frame/{i}.png   ?mode=raw|diff
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
def job_frame(job_id: str, index: int, mode: str = Query("raw", pattern="^(raw|diff)$")):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Кадри ще не готові")
    try:
        png = job["stack"].png(index, mode)
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


# --- звіт у PDF ---

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import red, Color
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def _make_pdf_report(job_id: str, frame_num: int, selected_region: dict, user_notes: str) -> bytes:
    """Generate a PDF report for the given job and frame."""
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Job not done")

    stack = job["stack"]
    frames = stack.frames if hasattr(stack, "frames") else []
    if not frames or frame_num < 0 or frame_num >= len(frames):
        raise HTTPException(404, "Invalid frame number")

    f = frames[frame_num]

    # Create the PDF buffer
    buffer = io.BytesIO()

    # Page setup
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch)

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    label_style = ParagraphStyle("Label", fontName="Helvetica-Bold", fontSize=10, textColor=Color(0.2, 0.2, 0.2))
    value_style = ParagraphStyle("Value", fontName="Helvetica", fontSize=10, textColor=Color(0.4, 0.4, 0.4))

    elements = []

    # Title
    elements.append(Paragraph("SPHEREx Time-Lapse Report", title_style))
    elements.append(Spacer(1, 12))

    # Job info table
    job_req = job["req"]
    table_data = [
        ["Parameter", "Value"],
        ["Об'єкт", job_req.get("target", "N/A")],
        ["RA", f"{getattr(f, 'ra', 'N/A')} {getattr(f, 'ra_unit', '')}".strip()],
        ["Dec", f"{getattr(f, 'dec', 'N/A')} {getattr(f, 'dec_unit', '')}".strip()],
        ["Епоха", str(getattr(f, 'epoch', 'N/A'))],
        ["Детектор", f"D{job_req.get('detector', 'N/A')}: {core.DETECTOR_BANDS.get(job_req.get('detector', 0), ['N/A', 'N/A'])[0]}–{core.DETECTOR_BANDS.get(job_req.get('detector', 0), ['N/A', 'N/A'])[1]} мкм"],
        ["FOV", f"{job_req.get('fov', 'N/A')} °"],
        ["Загальна кількість кадрів", str(len(frames))],
        ["Номер фрейму у звіті", str(frame_num + 1)],
        ["Дата фрейму", f.get('utc', f.get('time', 'N/A'))],
        ["Примітка", user_notes if user_notes else "—"],
    ]

    t = Table(table_data, colWidths=[1.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), Color(0.8, 0.8, 0.9)),
    ("TEXTCOLOR", (0, 0), (-1, 0), Color(0.1, 0.1, 0.1)),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("GRID", (0, 0), (-1, -1), 0.5, Color(0.5, 0.5, 0.5)),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [Color(1, 1, 1), Color(0.95, 0.95, 0.97)]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    try:
        png_data = stack.png(frame_num, mode="raw") if hasattr(stack, "png") else None
        if png_data:
            img = RLImage(io.BytesIO(png_data), width=4*inch, height=3*inch)
            elements.append(Paragraph("Фрейм з виділеною участю", label_style))
            elements.append(img)
            elements.append(Spacer(1, 8))

            # Draw rectangle on the image if region selected
            if selected_region and "x" in selected_region and "y" in selected_region and "w" in selected_region and "h" in selected_region:
                from reportlab.graphics import renderPDF
                from reportlab.graphics.shapes import Drawing, Rect
                from reportlab.lib.pagesizes import letter

                drawing = Drawing(4*inch, 3*inch)
                rect = Rect(selected_region["x"], selected_region["y"],
                        selected_region["x"] + selected_region["w"],
                        selected_region["y"] + selected_region["h"])
                rect.setFillColor = None
                rect.setStrokeColor = red
                rect.setStrokeWidth(2)
                drawing.add(rect)
                elements.append(Paragraph("Область виділена червоного контуру", label_style))
                elements.append(drawing)
    except Exception as e:
        elements.append(Paragraph(f"Не вдалося згенерувати зображення: {str(e)}", ParagraphStyle("Error", fontName="Helvetica-Bold", fontSize=9, textColor=red)))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Звіт згенерований: {time.strftime('%Y-%m-%d %H:%M:%S')}", ParagraphStyle("Footer", fontName="Helvetica-Oblique", fontSize=8, textColor=Color(0.6, 0.6, 0.6))))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()


@app.get("/api/reports/{job_id}.pdf")
def get_report(job_id: str, frame: int = Query(...), notes: str = Query("")):
    """Generate and return PDF report for a job."""
    import io
    from fastapi import FastAPI

    try:
        pdf_bytes = _make_pdf_report(job_id, frame, {}, notes)
        filename = f"SPHEREx_Report_{job_id}_frame_{frame + 1}.pdf"
        return Response(content=pdf_bytes, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Помилка генерації звіту: {str(e)}")
