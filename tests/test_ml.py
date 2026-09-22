"""Tests for the ML package.

The backend suite covers the API. Nothing covered ml/ itself, which is where the
geometry and the bookkeeping live - the parts that are arithmetically right and
still wrong at the edges. That is the class of bug these look for: a port list
that only covers one coast, a registry that marks a wreck as removed, a risk
score that leaves the range when nothing is known.

Run from the repository root:

    pytest tests
"""

from __future__ import annotations

import pytest

from ml.contract import CLASSES, ContractError, Detection, validate_result
from ml.enrich import PORTS, enrich_detection, haversine_km, nearest_port
from ml.heatmap import build as build_heatmap
from ml.heatmap import to_geojson
from ml.recovery import (
    INSPECT_FIRST,
    NOT_RECOVERABLE,
    ON_SITE_HOURS,
    day_plan,
    plan_recovery,
)
from ml.registry import MISSES_TO_GONE, Entry, Registry
from ml.risk import HARM, score_detection
from ml.survey import ALTITUDE_FRACTION_OF_RANGE, synthetic_track


def payload(**overrides) -> dict:
    base = {
        "image_id": "x",
        "width": 100,
        "height": 100,
        "processing_ms": 5,
        "detections": [{"class": "wreck", "confidence": 0.5, "bbox": [1, 1, 10, 10]}],
    }
    base.update(overrides)
    return base


class Context:
    """Stand-in for ml.enrich.Context, so these tests never touch the network."""

    def __init__(self, depth_m=None, biodiversity=None, nearest_port=None):
        self.depth_m = depth_m
        self.biodiversity = biodiversity
        self.nearest_port = nearest_port


# --- contract --------------------------------------------------------------

def test_valid_payload_is_accepted():
    validate_result(payload())


@pytest.mark.parametrize("detection", [
    {"class": "debris", "confidence": 0.5, "bbox": [1, 1, 10, 10]},   # not a class
    {"class": "wreck", "confidence": 1.4, "bbox": [1, 1, 10, 10]},    # out of range
    {"class": "wreck", "confidence": 0.5, "bbox": [90, 90, 50, 50]},  # past the edge
    {"class": "wreck", "confidence": 0.5, "bbox": [1, 1, 0, 10]},     # zero width
])
def test_off_contract_payloads_are_rejected(detection):
    with pytest.raises(ContractError):
        validate_result(payload(detections=[detection]))


def test_detection_survives_a_round_trip():
    d = Detection(cls="net", confidence=0.4, bbox=[1, 2, 3, 4])
    assert Detection.from_dict(d.to_dict()) == d


# --- enrichment ------------------------------------------------------------

def test_simulated_coordinates_are_refused():
    """A real depth for a place the sonar never saw is worse than no depth."""
    assert enrich_detection(12.92, 74.60, navigation_is_real=False) is None


def test_missing_coordinates_are_handled():
    assert enrich_detection(None, None, navigation_is_real=True) is None


def test_every_port_sits_within_the_indian_eez():
    outside = [n for n, (lat, lon) in PORTS.items()
               if not (6.0 <= lat <= 24.0 and 68.0 <= lon <= 94.0)]
    assert not outside


def test_no_two_ports_share_a_position():
    seen: dict[tuple[float, float], str] = {}
    for name, coords in PORTS.items():
        assert coords not in seen, f"{name} duplicates {seen.get(coords)}"
        seen[coords] = name


def test_each_port_is_its_own_nearest():
    """Catches a coordinate typo that would silently misroute a whole coast."""
    wrong = [n for n, (lat, lon) in PORTS.items() if nearest_port(lat, lon)["port"] != n]
    assert not wrong


def test_coasts_are_covered_not_just_karnataka():
    east = nearest_port(13.05, 80.35)["port"]        # off Chennai
    west = nearest_port(22.60, 69.60)["port"]        # Gulf of Kachchh
    assert east == "Chennai"
    assert nearest_port(13.05, 80.35)["distance_km"] < 50
    assert west != "Karwar"


def test_haversine_matches_a_known_distance():
    assert 590 < haversine_km((12.92, 74.80), (13.10, 80.30)) < 610


# --- risk ------------------------------------------------------------------

def test_every_contract_class_has_a_harm_weight():
    assert not [c for c in CLASSES if c not in HARM]


def test_score_stays_in_range_without_any_context():
    risk = score_detection("net", 0.5, None)
    assert 0.0 <= risk.score <= 1.0
    assert risk.band in {"HIGH", "MEDIUM", "LOW"}


def test_a_shallow_net_outranks_a_deep_wreck():
    net = score_detection("net", 0.95, Context(depth_m=8))
    wreck = score_detection("wreck", 0.30, Context(depth_m=90))
    assert net.score > wreck.score


def test_an_unknown_class_does_not_crash():
    assert 0.0 <= score_detection("something_new", 0.5, None).score <= 1.0


# --- recovery --------------------------------------------------------------

def test_every_contract_class_has_an_on_site_estimate():
    assert not [c for c in CLASSES if c not in ON_SITE_HOURS]


@pytest.mark.parametrize("depth,expected", [
    (5, "diver"), (25, "diver"), (45, "ROV"), (120, "ROV"),
])
def test_depth_chooses_the_method(depth, expected):
    assert expected in plan_recovery("H1", "net", depth_m=depth, transit_hours=1.0).method


def test_a_wreck_is_not_a_recovery_job():
    assert plan_recovery("H1", "wreck", depth_m=20, transit_hours=1.0).recoverable is False


def test_an_unidentified_object_is_inspected_before_it_is_lifted():
    plan = plan_recovery("H1", "unidentified", depth_m=20, transit_hours=1.0)
    assert plan.recoverable is False
    assert "identify" in plan.method
    assert any("ordnance" in n for n in plan.notes)


def test_a_day_plan_fits_inside_the_day():
    plans = [(plan_recovery(f"H{i}", "net", depth_m=10, transit_hours=0.5), 0.8)
             for i in range(6)]
    result = day_plan(plans, hours_available=8.0)
    assert result["hours_planned"] <= 8.0
    assert result["deferred"] == len(plans) - len(result["items"])


def test_nothing_to_recover_is_not_an_error():
    assert day_plan([], hours_available=8.0)["items"] == []


def test_classes_that_cannot_be_recovered_never_reach_a_crew():
    for cls in NOT_RECOVERABLE | INSPECT_FIRST:
        plan = plan_recovery("H1", cls, depth_m=20, transit_hours=1.0)
        assert day_plan([(plan, 0.9)], hours_available=8.0)["items"] == []


# --- registry --------------------------------------------------------------

def detection(lat: float, lon: float, cls: str = "net") -> dict:
    return {"class": cls, "confidence": 0.5, "bbox": [0, 0, 5, 5], "lat": lat, "lon": lon}


def test_the_same_object_seen_twice_is_one_entry():
    reg = Registry()
    reg.reconcile([detection(12.9231, 74.6012)], survey="S1")
    reg.reconcile([detection(12.9231, 74.6012)], survey="S2")
    assert len(reg.entries) == 1
    assert reg.entries[0].times_seen == 2


def test_objects_a_kilometre_apart_stay_separate():
    reg = Registry()
    reg.reconcile([detection(12.9231, 74.6012)], survey="S1")
    reg.reconcile([detection(12.9331, 74.6112)], survey="S2")
    assert len(reg.entries) == 2


def test_a_wreck_is_never_presumed_removed():
    """Wrecks do not drift, so a missed sighting is a miss, not a removal."""
    reg = Registry()
    reg.reconcile([detection(12.9231, 74.6012, cls="wreck")], survey="S1")
    for i in range(MISSES_TO_GONE + 2):
        reg.reconcile([], survey=f"S{i + 2}")
    assert reg.entries[0].status != "gone"


def test_gear_that_stops_appearing_is_marked_gone():
    reg = Registry()
    reg.reconcile([detection(12.9231, 74.6012)], survey="S1")
    for i in range(MISSES_TO_GONE + 1):
        reg.reconcile([], survey=f"S{i + 2}")
    assert reg.entries[0].status == "gone"


def test_detections_without_a_position_are_not_recorded():
    reg = Registry()
    reg.reconcile([detection(None, None)], survey="S1")
    assert reg.entries == []


# --- survey geometry -------------------------------------------------------

def test_altitude_follows_the_range():
    """Pinned at 12 m, a 13 m range gave an 8 degree grazing angle and sizes
    two and a half times too small."""
    nav = synthetic_track(10, slant_range_m=13.0)
    assert nav[0].altitude_m == pytest.approx(13.0 * ALTITUDE_FRACTION_OF_RANGE)


def test_the_long_standing_default_has_not_moved():
    assert synthetic_track(10, slant_range_m=75.0)[0].altitude_m == pytest.approx(12.0)


def test_an_altitude_above_the_range_images_nothing():
    with pytest.raises(ValueError):
        synthetic_track(10, slant_range_m=13.0, altitude_m=20.0)


# --- heatmap ---------------------------------------------------------------

def test_an_empty_heatmap_is_still_valid_geojson():
    geo = to_geojson(build_heatmap([]))
    assert geo["type"] == "FeatureCollection"
    assert geo["features"] == []


def test_heatmap_polygons_are_closed_and_in_range():
    entries = [Entry(hazard_id=f"H{i}", cls="net", lat=12.9231 + i * 0.0001,
                     lon=74.6012, first_seen="2026-01-10", last_seen="2026-09-01")
               for i in range(5)]
    for feature in to_geojson(build_heatmap(entries))["features"]:
        ring = feature["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "polygon ring is not closed"
        for lon, lat in ring:
            assert -180 <= lon <= 180 and -90 <= lat <= 90


# --- weights loading -------------------------------------------------------
#
# Every case here ends the same way if it goes wrong: the pipeline runs a mock
# detector, invents plausible boxes, and the backend stores them as real
# detections. Nothing downstream can tell the difference, so the checks have to
# happen at the point the weights are resolved.

def test_a_missing_weights_env_var_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    """SIH_ML_WEIGHTS is an explicit instruction. Pointing it at nothing used to
    pass the existence check - which ran before the override was applied - and
    land in mock mode."""
    from ml.inference import load_config

    monkeypatch.setenv("SIH_ML_WEIGHTS", str(tmp_path / "not-here.pt"))
    with pytest.raises(FileNotFoundError, match="SIH_ML_WEIGHTS"):
        load_config(tmp_path / "no-such-config.yaml")


def test_a_stale_path_in_the_config_file_falls_back(tmp_path, monkeypatch):
    """Unlike the environment override, a stale line in a checked-in config is
    worth stepping over: the shipped heads are the right answer."""
    import yaml as _yaml

    from ml.inference import load_config

    monkeypatch.delenv("SIH_ML_WEIGHTS", raising=False)
    cfg_path = tmp_path / "inference.yaml"
    cfg_path.write_text(_yaml.safe_dump({"weights": str(tmp_path / "gone.pt"),
                                         "imgsz": 640}))
    cfg = load_config(cfg_path)
    assert "weights" not in cfg
    assert cfg["imgsz"] == 640


def test_a_detector_without_weights_refuses_to_run(tmp_path):
    from ml.detector import Detector

    with pytest.raises(FileNotFoundError):
        Detector(weights=str(tmp_path / "absent.pt"))
    with pytest.raises(FileNotFoundError):
        Detector(weights=None)


def test_mock_mode_has_to_be_asked_for_and_still_warns():
    from ml.detector import Detector

    with pytest.warns(RuntimeWarning, match="INVENTED"):
        detector = Detector(weights=None, allow_mock=True)
    assert detector.kind == "mock"
    assert detector.version == "mock-0"


def test_the_checkpoint_fingerprint_is_computed_once(tmp_path, monkeypatch):
    """`version` stamps every inference result and the checkpoints are ~22 MB,
    so hashing on each access meant re-reading 44 MB per request."""
    from ml.detector import Detector

    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"not a real checkpoint, but it hashes the same way")
    # Skip the ultralytics load; this is about the fingerprint, not the model.
    monkeypatch.setattr(Detector, "_load",
                        lambda self, path: setattr(self, "_kind", "torch"))

    detector = Detector(weights=str(checkpoint))
    first = detector.version
    assert first.startswith("head-")

    # If it were still hashing on demand, losing the file would break this.
    checkpoint.unlink()
    assert detector.version == first


def test_a_report_renders_to_csv_without_a_file():
    """The API serves this straight down the response."""
    from ml.report import CSV_FIELDS, build_report, to_csv

    report = build_report(payload(), survey_name="S", navigation_source="test",
                          model_version="test-0")
    text = to_csv(report)
    header, *rows = text.strip().splitlines()
    assert header.split(",") == CSV_FIELDS
    assert len(rows) == len(report["anomalies"]) == 1


# --- across-track geometry -------------------------------------------------
#
# A side-scan row is sampled in slant range - time of flight - so pixels are
# not evenly spaced over the seabed. The scale used to be computed once per
# line from the outer edge and then multiplied by the pixel offset, which is
# the altitude = 0 case and overstates everything inside it.

def _ground(px, half=512.0, slant=75.0, alt=12.0):
    from ml.interfaces import ground_range_m
    return ground_range_m(px, half, slant, alt)


def test_the_outer_edge_is_unchanged_by_the_correction():
    """The old flat scale was derived from the edge, so the edge is the one
    place the two agree - far-field positions do not move."""
    import math
    half, slant, alt = 512.0, 75.0, 12.0
    flat = (2 * math.sqrt(slant**2 - alt**2)) / (2 * half)
    assert _ground(half) == pytest.approx(half * flat)


def test_zero_altitude_degenerates_to_the_flat_scale():
    """Flying on the seabed is the only case where pixels are linear in ground
    range, and it is the assumption the old code made everywhere."""
    from ml.interfaces import ground_range_m
    assert ground_range_m(200, 512.0, 75.0, 0.0) == pytest.approx(200 * (75.0 / 512.0))


def test_the_correction_shrinks_distances_towards_nadir():
    """Where it matters: the flat scale put an object 100 px off centre at
    14.5 m when it is really 8.4 m out - against a registry that matches
    hazards within 25 m."""
    import math
    flat = (2 * math.sqrt(75.0**2 - 12.0**2)) / 1024
    for px in (100, 200, 350):
        assert _ground(px) < px * flat
    assert px * flat - _ground(px) < 1.0          # and converges at the edge


def test_nothing_is_imaged_inside_the_water_column():
    """Closer than the towfish altitude, the pulse has not reached the seabed."""
    assert _ground(40) == 0.0                      # 12/75 * 512 = 82 px
    assert _ground(-40) == 0.0
    assert _ground(200) != 0.0


def test_the_sign_follows_the_channel():
    """Negative to port, positive to starboard - the bearing depends on it."""
    assert _ground(-200) == pytest.approx(-_ground(200))
    assert _ground(-200) < 0 < _ground(200)


def test_a_box_covers_more_seabed_near_nadir_than_at_the_edge():
    """Same box in pixels, different amount of seabed - which is why size_m has
    to come from both edges rather than one flat multiplier.

    Near nadir the grazing angle is steep, so a small step in slant range is a
    large step across the seabed. That is the compression you see in the middle
    of a slant-range waterfall, and it means a flat metres-per-pixel understates
    the size of anything sitting there."""
    near = _ground(140) - _ground(100)
    far = _ground(500) - _ground(460)
    assert near > far


def test_georeferencing_places_a_nadir_detection_closer_than_the_flat_scale():
    """End to end through the postprocessor, not just the helper."""
    import numpy as np

    from ml.interfaces import ReferencePostProcessor, SonarImage
    from ml.registry import haversine_m
    from ml.survey import attach_track

    sonar = SonarImage(image=np.zeros((200, 1024), np.uint8), image_id="line")
    attach_track(sonar)
    nav = sonar.nav[100]

    # a 40 px box 100 px to starboard of nadir
    det = {"class": "net", "confidence": 0.9, "bbox": [612.0, 100.0, 40.0, 20.0]}
    out = ReferencePostProcessor().georeference([det], sonar)[0]

    offset_m = haversine_m((nav.lat, nav.lon), (out["lat"], out["lon"]))
    flat_m = 120 * sonar.ground_range_per_px_m     # box centre, old scale
    assert offset_m < flat_m
    assert out["size_m"] > 0


def test_every_harm_weight_is_either_a_contract_class_or_a_model_label():
    """HARM carries model-native names - crab_pot, ship, aircraft - alongside
    the contract ones. Nothing reaches scoring under those names: the detector
    maps them (crab_pot -> net, ship/aircraft -> wreck) before the contract is
    ever built. They are kept because each agrees with the weight of what it
    maps to, so a native label arriving by another route still scores the same.
    This pins that agreement, which is the only thing making them harmless."""
    from ml.detector import NAME_TO_CONTRACT

    for name, weight in HARM.items():
        if name in CLASSES:
            continue
        assert name in NAME_TO_CONTRACT, f"{name!r} is in HARM but nothing maps it"
        mapped = NAME_TO_CONTRACT[name]
        assert HARM[mapped] == weight, (
            f"{name!r} weighs {weight} but maps to {mapped!r} which weighs "
            f"{HARM[mapped]} - the mapping would silently change the score"
        )


# --- coverage --------------------------------------------------------------

def test_coverage_is_none_without_navigation():
    """A bare .png has no geography, so it has no area. Reporting a number
    there would be inventing one."""
    import numpy as np

    from ml.inference import _coverage
    from ml.interfaces import SonarImage

    assert _coverage(SonarImage(image=np.zeros((100, 100), np.uint8),
                                image_id="png")) is None


def test_coverage_is_swath_times_track_length():
    import numpy as np

    from ml.inference import _coverage
    from ml.interfaces import SonarImage
    from ml.survey import attach_track

    sonar = SonarImage(image=np.zeros((2000, 1024), np.uint8), image_id="line")
    attach_track(sonar)
    cover = _coverage(sonar)

    assert cover["swath_m"] == pytest.approx(sonar.meta["swath_m"], abs=0.2)
    # The components are reported to 0.1 m and the area is computed at full
    # precision, so they agree to about that, not exactly.
    assert cover["area_m2"] == pytest.approx(
        cover["swath_m"] * cover["track_length_m"], rel=1e-3)


def test_a_coverage_block_is_contract_valid_and_optional():
    """1.1.0 is additive: a 1.0.0 payload with no coverage still validates."""
    validate_result(payload())                       # no coverage key at all
    validate_result(payload(coverage=None))
    validate_result(payload(coverage={"swath_m": 148.1, "track_length_m": 411.6,
                                      "area_m2": 60938.0}))
    with pytest.raises(ContractError):
        validate_result(payload(coverage={"area_m2": -1}))
    with pytest.raises(ContractError):
        validate_result(payload(coverage="0.06 km2"))


# --- sensor check ----------------------------------------------------------

class _FakeDetector:
    version = "fake-0"


def test_the_sensor_check_passes_on_sidescan_imagery():
    """A waterfall fills the frame edge to edge, which is what both shipped
    heads were trained on."""
    import numpy as np

    from ml.inference import _check_sensor
    from ml.interfaces import SonarImage

    # bright everywhere: no dark corners, so not a fan
    sonar = SonarImage(image=np.full((400, 400), 180, np.uint8), image_id="line")
    report = _check_sensor(sonar, {"wreck": (_FakeDetector(), 0.25)})

    assert report["detected"] == "sidescan"
    assert report["heads_match"] is True
    assert report["trained_on"] == ["sidescan"]


def test_a_forward_looking_frame_is_flagged_not_silently_processed(caplog):
    """The gap this closes: an FLS fan used to get two side-scan heads with
    nothing said, and a model off its own sensor is close to useless rather
    than merely worse."""
    import logging

    import numpy as np

    from ml.inference import _check_sensor
    from ml.interfaces import SonarImage

    # a fan: bright centre, dark corners
    image = np.zeros((400, 400), np.uint8)
    yy, xx = np.mgrid[0:400, 0:400]
    image[((xx - 200) ** 2 + (yy - 200) ** 2) < 150**2] = 200
    sonar = SonarImage(image=image, image_id="fan")

    with caplog.at_level(logging.WARNING):
        report = _check_sensor(sonar, {"wreck": (_FakeDetector(), 0.25)})

    assert report["detected"] == "fls"
    assert report["heads_match"] is False
    assert "unreliable" in caplog.text


def test_a_pinned_checkpoint_has_no_sensor_to_compare_against():
    """--weights bypasses HEADS, so there is no label to check. None, rather
    than a guess."""
    import numpy as np

    from ml.inference import _check_sensor
    from ml.interfaces import SonarImage

    sonar = SonarImage(image=np.full((400, 400), 180, np.uint8), image_id="x")
    assert _check_sensor(sonar, {"custom": (_FakeDetector(), 0.25)}) is None


def test_a_sensor_block_is_contract_valid_and_optional():
    validate_result(payload())
    validate_result(payload(sensor=None))
    validate_result(payload(sensor={"detected": "sidescan", "confident": True,
                                    "heads_match": True}))
    with pytest.raises(ContractError):
        validate_result(payload(sensor={"detected": 7}))
    with pytest.raises(ContractError):
        validate_result(payload(sensor={"heads_match": "yes"}))
