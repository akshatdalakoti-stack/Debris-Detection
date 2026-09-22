import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ml.heatmap import build as build_heatmap
from ml.heatmap import to_geojson as heatmap_to_geojson
from ml.registry import Entry as MLEntry

from ..db import get_db
from ..deps import require_viewer
from ..models import Detection, Job, RegistryEntry, SurveyFile

router = APIRouter(prefix="/api/registry", tags=["registry"],
                   dependencies=[Depends(require_viewer)])

def _to_ml_entry(e: RegistryEntry) -> MLEntry:
    return MLEntry(
        hazard_id=e.hazard_id,
        cls=e.class_name,
        lat=e.lat,
        lon=e.lon,
        first_seen=e.first_seen,
        last_seen=e.last_seen,
        times_seen=e.times_seen,
        consecutive_misses=e.consecutive_misses,
        status=e.status,
        best_confidence=e.best_confidence,
        surveys=json.loads(e.surveys),
        note=e.note
    )

def _serialise(e: RegistryEntry) -> dict:
    return {
        "hazard_id": e.hazard_id,
        "class": e.class_name,
        "lat": e.lat,
        "lon": e.lon,
        "status": e.status,
        "times_seen": e.times_seen,
        "last_seen": e.last_seen,
        "surveys": json.loads(e.surveys),
        "note": e.note,
    }


@router.get("")
def list_registry(
    status: str | None = Query(default=None,
                               description="present, unconfirmed, gone or recovered"),
    limit: int = Query(default=500, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """The registry, newest first.

    This used to select every row and serialise all of them. The registry grows
    by one entry per new hazard per survey and is never pruned, so it is the one
    table here with no natural ceiling - the response would have grown until it
    timed out. `total` is returned alongside so a caller can page.
    """
    query = db.query(RegistryEntry)
    if status is not None:
        query = query.filter(RegistryEntry.status == status)
    total = query.count()
    entries = (query.order_by(RegistryEntry.id.desc())
               .offset(offset).limit(limit).all())
    return {
        "entries": [_serialise(e) for e in entries],
        "total": total,
        "limit": limit,
        "offset": offset,
    }

@router.get("/heatmap")
def get_heatmap(db: Session = Depends(get_db)):
    # We build the heatmap only for active/unconfirmed items
    active_entries = db.query(RegistryEntry).filter(
        RegistryEntry.status.in_(["present", "unconfirmed"])
    ).all()

    ml_entries = [_to_ml_entry(e) for e in active_entries]

    # We could also provide a risk_scores dict if we joined with the detections table to get max risk.
    # For now, let's just pass the entries to the heatmap builder.
    cells = build_heatmap(ml_entries)
    geojson = heatmap_to_geojson(cells)

    return geojson


@router.get("/{hazard_id}/detections")
def hazard_detections(
    hazard_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Every detection that was folded into this hazard, newest first.

    A registry entry says a hazard is at a position and has been seen N times.
    Until detections carried the link, there was no way to get from that back
    to the surveys, the jobs or the boxes behind it - so "show me why you think
    there is a net here" had no answer.
    """
    entry = db.query(RegistryEntry).filter(
        RegistryEntry.hazard_id == hazard_id).first()
    if entry is None:
        raise HTTPException(status_code=404, detail="Hazard not found")

    query = (db.query(Detection, Job.id, SurveyFile.filename, SurveyFile.survey_id)
             .join(Job, Detection.job_id == Job.id)
             .join(SurveyFile, Job.file_id == SurveyFile.id)
             .filter(Detection.registry_entry_id == entry.id))
    total = query.count()
    rows = query.order_by(Detection.id.desc()).offset(offset).limit(limit).all()

    return {
        "hazard_id": entry.hazard_id,
        "class": entry.class_name,
        "times_seen": entry.times_seen,
        "total": total,
        "limit": limit,
        "offset": offset,
        "detections": [
            {
                "id": d.id,
                "job_id": job_id,
                "survey_id": survey_id,
                "filename": filename,
                "class": d.class_name,
                "confidence": d.confidence,
                "bbox": [d.x, d.y, d.width, d.height],
                "lat": d.lat,
                "lon": d.lon,
                "risk_score": d.risk_score,
                "risk_band": d.risk_band,
                "detected_at": d.created_at,
            }
            for d, job_id, filename, survey_id in rows
        ],
    }
