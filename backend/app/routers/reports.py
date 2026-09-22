from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_viewer
from ..models import Survey, Job, Detection, SurveyFile
from ml.report import build_report, to_csv

router = APIRouter(prefix="/api/surveys", tags=["reports"],
                   dependencies=[Depends(require_viewer)])


@router.get("/{survey_id}/report")
def get_survey_report(
    survey_id: int,
    format: str = Query("json", pattern="^(json|csv)$"),
    db: Session = Depends(get_db)
):
    survey = db.get(Survey, survey_id)
    if not survey:
        raise HTTPException(status_code=404, detail="Survey not found")

    detections = db.query(Detection).join(Job).join(SurveyFile).filter(
        SurveyFile.survey_id == survey_id
    ).all()

    processing_ms = sum((j.processing_ms or 0) for f in survey.files for j in f.jobs)

    dets = []
    for d in detections:
        dets.append({
            "class": d.class_name,
            "confidence": d.confidence,
            "bbox": [d.x, d.y, d.width, d.height],
            "lat": d.lat,
            "lon": d.lon,
            "size_m": d.size_m,
            "frame_index": d.frame_index
        })

    result = {
        "image_id": survey.name,
        "width": 1920,   # Placeholder if not stored
        "height": 1080,  # Placeholder if not stored
        "processing_ms": processing_ms,
        "detections": dets
    }

    report = build_report(
        result,
        survey_name=survey.name,
        navigation_source="DB Records",
        model_version="backend-v1"
    )

    if format == "json":
        return report

    # Built in memory. This used to write a NamedTemporaryFile with
    # delete=False, read it back and unlink it - so any exception between the
    # write and the remove left the file on disk, and the report carries every
    # position in the survey.
    return Response(
        content=to_csv(report),
        media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="survey_{survey_id}_report.csv"'},
    )
