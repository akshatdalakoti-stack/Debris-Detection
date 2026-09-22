from __future__ import annotations

import importlib.util
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(admin_client: TestClient) -> TestClient:
    """These exercise the API's behaviour, not its access control, so they run
    as an admin. Access control has its own tests in test_auth.py."""
    return admin_client


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_survey_history_and_error_shape(client: TestClient) -> None:
    client.post("/api/surveys", json={"name": "History survey"})

    surveys_response = client.get("/api/surveys")
    assert surveys_response.status_code == 200
    assert surveys_response.json()[0]["name"] == "History survey"

    missing_job_response = client.get("/api/jobs/999999")
    assert missing_job_response.status_code == 404
    assert missing_job_response.json() == {"message": "Job not found"}


# The pipeline needs ultralytics and the checkpoints. Skipped rather than
# failed where they are absent: a missing optional dependency reported as
# `assert 'failed' == 'done'` sends you looking for a bug in the API.
needs_model = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None,
    reason="ultralytics is not installed - end-to-end inference cannot run",
)


@needs_model
def test_upload_processing_and_exports(client: TestClient) -> None:
    survey_response = client.post("/api/surveys", json={"name": "Demo survey"})
    assert survey_response.status_code == 201
    survey_id = survey_response.json()["id"]

    # Generate a minimal valid JPEG in memory — avoids any filesystem dependency
    # so this test runs identically locally and inside the Docker container.
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(30, 30, 30)).save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    upload_response = client.post(
        f"/api/surveys/{survey_id}/upload",
        files={"file": ("000346.jpg", image_bytes, "image/jpeg")},
    )
    assert upload_response.status_code == 202
    job_id = upload_response.json()["job_id"]

    job_response = client.get(f"/api/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "done", job_response.json().get("error")
    assert job_response.json()["progress"] == 100

    detections_response = client.get(
        f"/api/jobs/{job_id}/detections", params={"class": "debris", "min_conf": 0.8}
    )
    assert detections_response.status_code == 200
    # Assert response shape — not prediction count, which depends on the ML model.
    det_body = detections_response.json()
    assert "total" in det_body
    assert "items" in det_body
    assert isinstance(det_body["total"], int)
    assert isinstance(det_body["items"], list)

    summary_response = client.get(f"/api/jobs/{job_id}/summary")
    assert summary_response.status_code == 200
    # Assert response shape — not specific class counts.
    summary_body = summary_response.json()
    assert "by_class" in summary_body
    assert isinstance(summary_body["by_class"], dict)

    image_response = client.get(f"/api/jobs/{job_id}/image")
    assert image_response.status_code == 200
    # Whatever the pipeline drew - the real one writes a PNG - the declared
    # type has to match the bytes, or the browser refuses to render it.
    content_type = image_response.headers["content-type"]
    assert content_type.startswith("image/")
    if content_type == "image/png":
        assert image_response.content[:4] == bytes.fromhex("89504e47")
    elif content_type == "image/svg+xml":
        assert image_response.content.lstrip()[:4] == b"<svg"

    json_response = client.get(f"/api/jobs/{job_id}/export", params={"format": "json"})
    assert json_response.status_code == 200
    assert json_response.json()["job_id"] == job_id

    csv_response = client.get(f"/api/jobs/{job_id}/export", params={"format": "csv"})
    assert csv_response.status_code == 200
    assert "confidence" in csv_response.text


def test_upload_validation(client: TestClient) -> None:
    survey_id = client.post("/api/surveys", json={"name": "Validation survey"}).json()["id"]

    unsupported = client.post(
        f"/api/surveys/{survey_id}/upload",
        files={"file": ("sample.exe", b"data", "application/octet-stream")},
    )
    assert unsupported.status_code == 400

    empty = client.post(
        f"/api/surveys/{survey_id}/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert empty.status_code == 400


def test_registry_listing_is_paged(client: TestClient) -> None:
    """The registry is the one table with no ceiling - an entry per hazard per
    survey, never pruned - so the endpoint must not select all of it."""
    response = client.get("/api/registry", params={"limit": 10})
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"entries", "total", "limit", "offset"}
    assert body["limit"] == 10
    assert len(body["entries"]) <= 10

    assert client.get("/api/registry", params={"limit": 0}).status_code == 422
    assert client.get("/api/registry", params={"limit": 99999}).status_code == 422


def test_survey_report_csv_is_built_without_touching_disk(client: TestClient) -> None:
    import tempfile
    from pathlib import Path

    survey_id = client.post("/api/surveys", json={"name": "Report survey"}).json()["id"]
    before = set(Path(tempfile.gettempdir()).glob("*.csv"))

    response = client.get(f"/api/surveys/{survey_id}/report", params={"format": "csv"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "anomaly_id" in response.text

    # It used to write a NamedTemporaryFile(delete=False) and unlink it by hand.
    assert set(Path(tempfile.gettempdir()).glob("*.csv")) == before


def test_health_is_cheap_and_readiness_checks_the_database(client: TestClient) -> None:
    """Two different questions. The container HEALTHCHECK and compose's
    depends_on hang off /api/health, so it must not fail when the database
    does - restarting the API would not fix that. /api/ready is what a load
    balancer should ask."""
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/ready").json() == {"status": "ready"}


def test_readiness_reports_503_when_the_database_is_gone(client: TestClient) -> None:
    from app.db import get_db
    from app.main import app

    def broken_db():
        class Dead:
            def execute(self, *_a, **_k):
                raise RuntimeError("connection refused")
        yield Dead()

    app.dependency_overrides[get_db] = broken_db
    try:
        response = client.get("/api/ready")
        assert response.status_code == 503
        # Liveness is unaffected: the process itself is fine.
        assert client.get("/api/health").status_code == 200
    finally:
        app.dependency_overrides.pop(get_db, None)
