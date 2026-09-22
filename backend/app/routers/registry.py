import json
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_viewer
from ..models import RegistryEntry
from ml.registry import Entry as MLEntry
from ml.heatmap import build as build_heatmap, to_geojson as heatmap_to_geojson

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
