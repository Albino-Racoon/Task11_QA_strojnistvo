"""API routes: health, inspect, history, stats, files."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.config import get_settings
from api.database import get_db
from api.services.inspection_service import get_registry, run_inspection
from api.services.stats_service import (
    defect_frequency,
    get_today_stats,
    list_inspection_lines,
    list_inspections,
)
from camera_service.capture import capture_frame, list_cameras

router = APIRouter()


def _read_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Neveljavna slika.")
    return image


@router.get("/health")
def health():
    settings = get_settings()
    reg = get_registry()
    return {
        "status": "ok",
        "app": settings.app_name,
        "models": reg.health(),
    }


@router.post("/models/reload")
def reload_models():
    get_registry().reload()
    return {"status": "reloaded", "models": get_registry().health()}


@router.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    part_id: Optional[str] = Form(default=None),
    line: str = Form(default="demo"),
    db: Session = Depends(get_db),
):
    image = _read_upload(file)
    return run_inspection(
        db, image, part_id=part_id, line=line, source_name=file.filename or "upload"
    )


@router.post("/inspect/camera")
def inspect_camera(
    part_id: Optional[str] = None,
    line: str = "demo",
    device_index: Optional[int] = None,
    db: Session = Depends(get_db),
):
    settings = get_settings()
    idx = settings.camera_device_index if device_index is None else device_index
    try:
        frame = capture_frame(idx)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return run_inspection(db, frame, part_id=part_id, line=line, source_name=f"camera:{idx}")


@router.get("/cameras")
def cameras():
    return {"cameras": list_cameras()}


@router.get("/inspections")
def inspections(
    status: Optional[str] = Query(default=None),
    line: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    return {"items": list_inspections(db, status=status, line=line, limit=limit)}


@router.get("/inspections/lines")
def inspection_lines(db: Session = Depends(get_db)):
    return {"items": list_inspection_lines(db)}


@router.get("/stats/today")
def stats_today(db: Session = Depends(get_db)):
    return get_today_stats(db)


@router.get("/stats/defects")
def stats_defects(db: Session = Depends(get_db)):
    return {"items": defect_frequency(db)}


@router.get("/files/{file_path:path}")
def get_file(file_path: str):
    settings = get_settings()
    full = (settings.storage_dir / file_path).resolve()
    if not str(full).startswith(str(settings.storage_dir.resolve())):
        raise HTTPException(status_code=400, detail="Neveljavna pot.")
    if not full.exists():
        raise HTTPException(status_code=404, detail="Datoteka ne obstaja.")
    return FileResponse(full)
