import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_analyst
from ..models import SurveyFile

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/active-learning", tags=["active_learning"],
                   dependencies=[Depends(require_analyst)])


class RankRequest(BaseModel):
    images: list[str] | None = None
    # Unbounded, this ran a model over every image in the database and built the
    # whole ranking in memory before returning any of it.
    top_k: int = Field(default=50, ge=1, le=500)


@router.post("/rank")
def rank_images(req: RankRequest, db: Session = Depends(get_db)):
    """Rank unlabelled imagery by how much annotating it would teach the model.

    Needs ultralytics and the checkpoints, so it is imported lazily - the rest
    of the API has to start on a host that has neither.
    """
    try:
        from ml.active import annotation_budget_note, rank_for_annotation
        from ml.pipeline import load_models
    except ImportError as exc:
        log.exception("active learning dependencies are missing")
        raise HTTPException(
            status_code=503,
            detail="Ranking needs the ML dependencies, which are not installed here",
        ) from exc

    images = req.images
    if not images:
        images = [path for (path,) in db.query(SurveyFile.storage_path).all()]
    if not images:
        return {"note": "No images available in the database to rank.", "candidates": []}

    try:
        models = load_models()
        candidates = rank_for_annotation(models, images, top_k=req.top_k)
    except FileNotFoundError as exc:
        # Missing checkpoints are worth naming: it is a deployment problem
        # somebody can fix, not a bug.
        log.exception("active learning could not load the model heads")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Everything else goes to the log. The traceback carries absolute paths,
        # the service account and the installed library layout - the same
        # reasoning as services/jobs.py, which this endpoint did not follow.
        log.exception("active learning ranking failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: ranking failed",
        ) from exc

    return {
        "note": annotation_budget_note(len(images), len(candidates)),
        "candidates": [c.to_dict() for c in candidates],
    }
