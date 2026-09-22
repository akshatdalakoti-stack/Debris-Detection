from __future__ import annotations

import json
import logging

from sqlalchemy import delete

from ml.inference import run_inference
from ml.risk import score_detection

from ..config import settings
from ..db import SessionLocal
from ..models import Detection, Job, utc_now
from .enrichment import cached_enrich
from .registry import RegistryService

log = logging.getLogger(__name__)


def process_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            return

        job.status = "processing"
        job.progress = 5
        job.started_at = utc_now()
        job.error = None
        db.commit()

        cfg = {"output_dir": str(settings.overlays_dir)}

        def report(stage: str, pct: int) -> None:
            """Drive the progress bar from the pipeline's own stages.

            run_inference has taken a progress_cb since it was written and
            nothing ever passed one, so the bar went 10 -> 30 -> 80 -> 100
            regardless of what was happening - the long part, tiling and
            running both heads over every tile, was a single jump. Capped below
            100 because the detections still have to be enriched, scored and
            written after the model is done.
            """
            job.progress = min(pct, 90)
            db.commit()
            log.debug("job %s: %s (%d%%)", job_id, stage, pct)

        result = run_inference(job.file.storage_path, config=cfg, progress_cb=report)

        job.progress = 95
        db.commit()
        db.execute(delete(Detection).where(Detection.job_id == job.id))
        rows: list[Detection] = []
        for item in result["detections"]:
            x, y, width, height = item["bbox"]
            lat, lon = item.get("lat"), item.get("lon")

            ctx = cached_enrich(lat, lon)
            risk = score_detection(item["class"], item["confidence"], ctx)

            rows.append(
                Detection(
                    job_id=job.id,
                    class_name=item["class"],
                    confidence=item["confidence"],
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    lat=lat,
                    lon=lon,
                    size_m=item.get("size_m"),
                    frame_index=item.get("frame_index"),
                    depth_m=ctx.depth_m if ctx else None,
                    biodiversity_species=ctx.biodiversity.get("species") if ctx and ctx.biodiversity else None,
                    nearest_port_km=ctx.nearest_port.get("distance_km") if ctx and ctx.nearest_port else None,
                    risk_score=risk.score,
                    risk_band=risk.band,
                    risk_reasons=json.dumps(risk.reasons)
                )
            )

        db.add_all(rows)

        registry_service = RegistryService(db)
        entries = registry_service.reconcile(result["detections"],
                                             survey=job.file.survey.name)
        # Reconciliation is already working out which hazard each detection is;
        # recording it is what lets a hazard be traced back to the imagery and
        # the boxes that found it.
        for row, entry in zip(rows, entries, strict=True):
            if entry is not None:
                row.registry_entry_id = entry.id

        job.overlay_path = result["overlay_path"]
        job.processing_ms = result["processing_ms"]
        coverage = result.get("coverage")
        job.area_covered_m2 = coverage["area_m2"] if coverage else None
        job.progress = 100
        job.status = "done"
        job.finished_at = utc_now()
        db.commit()
    except Exception as exc:
        db.rollback()
        # The traceback names absolute paths, the account the service runs as and
        # the installed library layout. It belongs in the log, not in an API
        # response any client can read. The client gets the exception type and
        # the job id, which is enough to report a fault and enough for an
        # operator to find the entry.
        log.exception("job %s failed", job_id)
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: processing failed (job {job_id})"
            job.finished_at = utc_now()
            db.commit()
    finally:
        db.close()

