"""Reconciliation against the database.

The ML package has its own registry (ml/registry.py, JSON-backed) and its own
tests. This is the SQLAlchemy one the worker actually uses, which had no
coverage - and which differs from the ML version in exactly the places a
database makes a difference: unique constraints and how much of the table a
lookup reads.
"""

from __future__ import annotations

import pytest
from app.db import Base, SessionLocal, engine
from app.models import RegistryEntry
from app.services.registry import MATCH_RADIUS_M, RegistryService


@pytest.fixture
def db():
    # These talk to the session directly rather than through the API, so the
    # app's lifespan - which is what normally creates the tables - never runs.
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    session.query(RegistryEntry).delete()
    session.commit()
    try:
        yield session
    finally:
        session.query(RegistryEntry).delete()
        session.commit()
        session.close()


def detection(lat: float, lon: float, cls: str = "net", conf: float = 0.8) -> dict:
    return {"class": cls, "confidence": conf, "bbox": [0, 0, 10, 10],
            "lat": lat, "lon": lon}


def test_hazard_ids_survive_a_deletion(db):
    """The id was f"HZ-{count + 1:05d}". Delete an entry and the next insert
    reuses a live id, which the unique constraint then rejects."""
    service = RegistryService(db)
    service.reconcile([detection(12.0, 74.0), detection(12.5, 74.5)], survey="s1")

    first = db.query(RegistryEntry).order_by(RegistryEntry.id).first()
    db.delete(first)
    db.commit()

    # Would have collided with the surviving entry's id.
    service.reconcile([detection(13.0, 75.0)], survey="s2")
    ids = [e.hazard_id for e in db.query(RegistryEntry).all()]
    assert len(ids) == len(set(ids)), ids


def test_a_detection_inside_the_radius_matches_the_same_entry(db):
    service = RegistryService(db)
    service.reconcile([detection(12.0, 74.0)], survey="s1")
    assert db.query(RegistryEntry).count() == 1

    # ~10 m north, inside the 25 m tolerance.
    service.reconcile([detection(12.0 + 10 / 111_320, 74.0)], survey="s2")
    assert db.query(RegistryEntry).count() == 1
    assert db.query(RegistryEntry).one().times_seen == 2


def test_a_detection_outside_the_radius_is_a_separate_hazard(db):
    service = RegistryService(db)
    service.reconcile([detection(12.0, 74.0)], survey="s1")

    # ~100 m north, well outside it.
    service.reconcile([detection(12.0 + 100 / 111_320, 74.0)], survey="s2")
    assert db.query(RegistryEntry).count() == 2


@pytest.mark.parametrize("lat", [0.0, 12.0, 60.0, 78.0])
def test_the_bounding_box_is_wide_enough_at_every_latitude(db, lat):
    """The prefilter converts metres to degrees, and a degree of longitude
    shrinks with cos(lat). Too narrow a box silently stops matching and every
    revisit becomes a new hazard - which reads as debris that was never
    recovered."""
    import math

    service = RegistryService(db)
    service.reconcile([detection(lat, 74.0)], survey="s1")

    # Just inside the radius, displaced entirely in longitude - the direction
    # the conversion can get wrong.
    metres = MATCH_RADIUS_M * 0.8
    d_lon = metres / (111_320.0 * math.cos(math.radians(lat)))
    service.reconcile([detection(lat, 74.0 + d_lon)], survey="s2")

    assert db.query(RegistryEntry).count() == 1
    assert db.query(RegistryEntry).one().times_seen == 2


def test_a_wreck_is_never_presumed_removed(db):
    """Same rule as the ML registry: a shipwreck does not leave, so repeated
    misses are the detector's, not a recovery."""
    service = RegistryService(db)
    service.reconcile([detection(12.0, 74.0, cls="wreck")], survey="s1")
    for n in range(4):
        service.reconcile([], survey=f"miss-{n}")

    entry = db.query(RegistryEntry).one()
    assert entry.status == "unconfirmed"
    assert entry.status != "gone"


def test_reconcile_reports_which_hazard_each_detection_became(db):
    """The link the worker records. Aligned with the input list, None where a
    detection had no position to track."""
    service = RegistryService(db)
    entries = service.reconcile(
        [detection(12.0, 74.0),
         {"class": "net", "confidence": 0.5, "bbox": [0, 0, 5, 5],
          "lat": None, "lon": None},
         detection(13.0, 75.0)],
        survey="s1",
    )

    assert len(entries) == 3
    assert entries[1] is None                     # ungeotagged
    assert entries[0] is not None and entries[2] is not None
    assert entries[0].hazard_id != entries[2].hazard_id


def test_a_revisit_points_at_the_same_hazard(db):
    service = RegistryService(db)
    first = service.reconcile([detection(12.0, 74.0)], survey="s1")[0]
    second = service.reconcile([detection(12.0 + 10 / 111_320, 74.0)], survey="s2")[0]
    assert first.id == second.id
