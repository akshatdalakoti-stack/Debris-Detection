"""Anomaly reporting: turn detections into the structured output PS 57 asks for.

The problem statement requires a report giving, per detected hazard, its exact
location (latitude/longitude), bounding dimensions, and classification, in
JSON or CSV. This module is that engine.

JSON carries the full record including survey provenance. CSV is the flat
table an analyst opens in a spreadsheet or loads into GIS.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_VERSION = "1.0"


def build_report(
    result: dict[str, Any],
    *,
    survey_name: str = "unnamed-survey",
    navigation_source: str = "unknown",
    model_version: str = "unset",
) -> dict[str, Any]:
    """Assemble the anomaly report from a run_inference() result.

    `navigation_source` is recorded verbatim in the report. When positions come
    from a simulated track it must say so - a geotagged report that does not
    disclose simulated navigation is a report that can mislead someone into
    sending a vessel to a coordinate.
    """
    detections = result.get("detections", [])
    geotagged = [d for d in detections if d.get("lat") is not None]

    anomalies = []
    for i, d in enumerate(detections, 1):
        x, y, w, h = d["bbox"]
        anomalies.append(
            {
                "anomaly_id": f"{result['image_id']}-{i:03d}",
                "classification": d["class"],
                "confidence_pct": round(float(d["confidence"]) * 100, 1),
                "latitude": d.get("lat"),
                "longitude": d.get("lon"),
                "bbox_pixels": {"x": round(x), "y": round(y),
                                "width": round(w), "height": round(h)},
                "dimensions_m": {"length": d.get("size_m")},
                "frame_index": d.get("frame_index"),
                # present only when the pipeline ran with enrichment
                **({"context": d["context"]} if d.get("context") else {}),
                **({"risk": d["risk"]} if d.get("risk") else {}),
            }
        )

    by_class: dict[str, int] = {}
    for d in detections:
        by_class[d["class"]] = by_class.get(d["class"], 0) + 1

    return {
        "report_version": REPORT_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "survey": {
            "name": survey_name,
            "source_image": result["image_id"],
            "image_size_px": [result["width"], result["height"]],
            "navigation_source": navigation_source,
        },
        "model": {
            "version": model_version,
            "processing_ms": result.get("processing_ms"),
        },
        "summary": {
            "total_anomalies": len(detections),
            "geotagged": len(geotagged),
            "by_classification": by_class,
            "mean_confidence_pct": round(
                sum(float(d["confidence"]) for d in detections) / len(detections) * 100, 1
            ) if detections else 0.0,
        },
        "anomalies": anomalies,
    }


def write_json(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


CSV_FIELDS = [
    "anomaly_id", "classification", "confidence_pct",
    "latitude", "longitude", "length_m",
    "bbox_x", "bbox_y", "bbox_width", "bbox_height",
    "frame_index", "survey", "navigation_source",
]


def to_csv(report: dict[str, Any]) -> str:
    """Flat one-row-per-anomaly table for spreadsheets and GIS import.

    Returns the text rather than writing it. The API serves this straight down
    the response, and routing that through a temporary file on disk only added
    a way to leave one behind.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for a in report["anomalies"]:
        b = a["bbox_pixels"]
        writer.writerow({
            "anomaly_id": a["anomaly_id"],
            "classification": a["classification"],
            "confidence_pct": a["confidence_pct"],
            "latitude": a["latitude"],
            "longitude": a["longitude"],
            "length_m": a["dimensions_m"]["length"],
            "bbox_x": b["x"], "bbox_y": b["y"],
            "bbox_width": b["width"], "bbox_height": b["height"],
            "frame_index": a["frame_index"],
            "survey": report["survey"]["name"],
            "navigation_source": report["survey"]["navigation_source"],
        })
    return buffer.getvalue()


def write_csv(report: dict[str, Any], path: str | Path) -> Path:
    """to_csv(), written to `path`. Kept for the demo CLI."""
    path = Path(path)
    path.write_text(to_csv(report), encoding="utf-8", newline="")
    return path
