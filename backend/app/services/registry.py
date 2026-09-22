import json
import math
from datetime import date
from uuid import uuid4

from sqlalchemy.orm import Session

from ..models import RegistryEntry

MATCH_RADIUS_M = 25.0        # positional tolerance when matching to the registry
MISSES_TO_GONE = 2           # consecutive surveys missing before presumed removed

IMMOVABLE = {"ship", "wreck", "aircraft"}

PRESENT = "present"          # seen in the most recent survey covering it
UNCONFIRMED = "unconfirmed"  # missed once - could be a miss, could be gone
GONE = "gone"                # missed repeatedly, presumed recovered or shifted
RECOVERED = "recovered"      # a crew reported recovering it - authoritative


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6_371_008.8
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


class RegistryService:
    """Backend service for managing the persistent debris registry in the database."""

    def __init__(self, db: Session):
        self.db = db

    def _add_with_hazard_id(self, **fields) -> RegistryEntry:
        """Insert an entry and give it a hazard id derived from its own key.

        The id used to be f"HZ-{count + 1:05d}". hazard_id is UNIQUE, so that
        collided two ways: delete any entry and the next insert reuses a live
        id, and two workers reconciling different surveys at the same moment
        both read the same count and one of them fails the constraint - taking
        a whole job down with it. The primary key comes from a sequence, so
        deriving from it after the flush is unique without coordinating.
        """
        entry = RegistryEntry(hazard_id=f"pending-{uuid4()}", **fields)
        self.db.add(entry)
        self.db.flush()                  # assigns the primary key
        entry.hazard_id = f"HZ-{entry.id:05d}"
        self.db.flush()
        return entry

    def _nearest(self, lat: float, lon: float, cls: str) -> RegistryEntry | None:
        """Closest entry of the same class inside MATCH_RADIUS_M, or None.

        The haversine has to run in Python, but it no longer runs over the whole
        class. A bounding box in SQL cuts it down first - this is called once
        per detection, so it was a full scan of the table per box, and the
        registry is the one table here that only ever grows.
        """
        # Latitude is ~111.32 km/degree everywhere; longitude shrinks with
        # cos(lat). Generous by design - the box only has to contain the circle,
        # and the haversine below does the real test.
        d_lat = MATCH_RADIUS_M / 111_320.0
        cos_lat = max(math.cos(math.radians(lat)), 1e-6)
        d_lon = MATCH_RADIUS_M / (111_320.0 * cos_lat)

        entries = self.db.query(RegistryEntry).filter(
            RegistryEntry.class_name == cls,
            RegistryEntry.status != RECOVERED,
            RegistryEntry.lat.between(lat - d_lat, lat + d_lat),
            RegistryEntry.lon.between(lon - d_lon, lon + d_lon),
        ).all()

        best, best_d = None, MATCH_RADIUS_M
        for e in entries:
            d = haversine_m((lat, lon), (e.lat, e.lon))
            if d < best_d:
                best, best_d = e, d
        return best

    def reconcile(self, detections: list[dict], survey: str,
                  when: str | None = None,
                  covered: list[str] | None = None) -> None:
        """Fold one survey's detections into the registry."""
        when = when or date.today().isoformat()
        matched: set[str] = set()

        for d in detections:
            lat, lon = d.get("lat"), d.get("lon")
            if lat is None or lon is None:
                continue
            cls = d.get("class", "unidentified")
            conf = float(d.get("confidence", 0.0))

            hit = self._nearest(lat, lon, cls)
            if hit:
                hit.last_seen = when
                hit.times_seen += 1
                hit.consecutive_misses = 0
                hit.status = PRESENT
                hit.best_confidence = max(hit.best_confidence, conf)

                surveys = json.loads(hit.surveys)
                if survey not in surveys:
                    surveys.append(survey)
                    hit.surveys = json.dumps(surveys)

                matched.add(hit.hazard_id)
            else:
                new_entry = self._add_with_hazard_id(
                    class_name=cls,
                    lat=lat,
                    lon=lon,
                    first_seen=when,
                    last_seen=when,
                    best_confidence=conf,
                    surveys=json.dumps([survey]),
                )
                matched.add(new_entry.hazard_id)

        # anything covered by this survey but not matched counts as a miss
        all_live = self.db.query(RegistryEntry).filter(
            ~RegistryEntry.status.in_([GONE, RECOVERED])
        ).all()

        for e in all_live:
            if e.hazard_id in matched:
                continue
            if covered is not None and e.hazard_id not in covered:
                continue  # survey never went near it

            e.consecutive_misses += 1
            if e.class_name in IMMOVABLE:
                if e.status != UNCONFIRMED:
                    e.status = UNCONFIRMED
                    e.note = (f"missed {e.consecutive_misses}x, but a {e.class_name} "
                              f"cannot be recovered - treat as a detection miss "
                              f"and keep the position as a permanent snag hazard")
            elif e.consecutive_misses >= MISSES_TO_GONE:
                if e.status != GONE:
                    e.status = GONE
                    e.note = (f"not detected in {e.consecutive_misses} consecutive "
                              f"surveys - presumed recovered or moved")
            else:
                if e.status != UNCONFIRMED:
                    e.status = UNCONFIRMED
                    e.note = ("missed once; detector recall is 0.46, so a single "
                              "miss is not evidence of removal")

        self.db.commit()

    def mark_recovered(self, hazard_id: str, when: str | None = None) -> bool:
        e = self.db.query(RegistryEntry).filter(RegistryEntry.hazard_id == hazard_id).first()
        if e:
            e.status = RECOVERED
            e.note = f"recovered {when or date.today().isoformat()}"
            self.db.commit()
            return True
        return False
