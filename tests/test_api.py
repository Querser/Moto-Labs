from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.database import get_db
from app.domain import LapTimingService, PassageCandidate
from app.main import app
from app.services.camera_runtime import camera_runtime
from app.services.vision_runtime import vision_runtime


def test_minimal_api_lifecycle_corrections_results_and_export(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def db_override() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_settings] = lambda: Settings(export_dir=tmp_path)
    monkeypatch.setattr("app.main._restore_active_runtime", lambda: None)
    monkeypatch.setattr("app.main._pause_running_race_on_shutdown", lambda: None)
    try:
        with TestClient(app) as client:
            assert client.get("/api/health").json()["local_only"] is True
            cameras = client.get("/api/cameras").json()
            assert any(item["identifier"] == "synthetic" for item in cameras)
            selected = client.put("/api/camera", json={"camera_identifier": "synthetic"})
            assert selected.status_code == 200
            assert selected.json()["selected_camera"] == "synthetic"
            selected_again = client.put(
                "/api/camera", json={"camera_identifier": "synthetic"}
            )
            assert selected_again.status_code == 200
            assert selected_again.json()["status"] == "running"
            snapshot = client.get("/api/camera/snapshot")
            assert snapshot.status_code == 200
            assert snapshot.headers["content-type"] == "image/jpeg"
            assert snapshot.headers["cache-control"] == "no-store, no-cache, must-revalidate"
            snapshot_sequence = int(snapshot.headers["x-frame-sequence"])
            assert snapshot_sequence >= 0
            assert snapshot.content.startswith(b"\xff\xd8")
            assert snapshot.content.endswith(b"\xff\xd9")
            next_snapshot = client.get(
                "/api/camera/snapshot",
                params={"after": snapshot_sequence},
            )
            assert int(next_snapshot.headers["x-frame-sequence"]) > snapshot_sequence
            line = client.put(
                "/api/camera/line",
                json={"x1": 0.1, "y1": 0.7, "x2": 0.9, "y2": 0.65},
            )
            assert line.status_code == 200
            assert line.json()["finish_line"] == {
                "x1": 0.1,
                "y1": 0.7,
                "x2": 0.9,
                "y2": 0.65,
            }

            created = client.post(
                "/api/races",
                json={
                    "name": "API race",
                    "description": "minimal",
                    "required_laps": 2,
                    "camera_identifier": "synthetic",
                },
            )
            assert created.status_code == 201
            assert created.json()["description"] == "minimal"
            race_id = created.json()["id"]
            assert client.post(f"/api/races/{race_id}/start").json()["status"] == "running"
            assert client.post(f"/api/races/{race_id}/pause").json()["status"] == "paused"
            assert client.post(f"/api/races/{race_id}/resume").json()["status"] == "running"

            # Direct domain insertion keeps this integration test deterministic while
            # the separate runtime test covers automatic synthetic recognition.
            with session_factory() as session:
                race = LapTimingService(session).get_race(race_id)
                start = race.monotonic_start_reference_ns
                assert start is not None
                decision = LapTimingService(session).record_passage(
                    race_id,
                    PassageCandidate(
                        "007",
                        start + race.total_paused_ns + 1_000_000_000,
                        datetime.now(timezone.utc),
                        0.99,
                        idempotency_key="api-lap",
                    ),
                )
                lap_id = decision.lap.id  # type: ignore[union-attr]
                session.commit()

            laps = client.get(f"/api/races/{race_id}/laps").json()
            assert laps[0]["racing_number"] == "007"
            corrected = client.patch(
                f"/api/races/{race_id}/laps/{lap_id}", json={"racing_number": "0007"}
            )
            assert corrected.json()["racing_number"] == "0007"
            summary = client.get(f"/api/races/{race_id}/results").json()["summary"]
            assert summary[0]["completed_laps"] == 1
            exported = client.get(f"/api/races/{race_id}/export")
            assert exported.status_code == 200
            assert exported.content.startswith(b"PK")
            assert client.delete(f"/api/races/{race_id}/laps/{lap_id}").status_code == 204

            # Starting another named race must not require a page reload.  The
            # previous race is finalized and its rows remain isolated.
            second = client.post(
                "/api/races",
                json={
                    "name": "Second API race",
                    "required_laps": 4,
                    "camera_identifier": "synthetic",
                },
            ).json()
            second_started = client.post(f"/api/races/{second['id']}/start")
            assert second_started.status_code == 200
            assert second_started.json()["status"] == "running"
            assert client.get(f"/api/races/{race_id}").json()["status"] == "finished"
            assert client.get(f"/api/races/{second['id']}/laps").json() == []
            assert client.post(f"/api/races/{second['id']}/finish").json()["status"] == "finished"
    finally:
        vision_runtime.stop()
        camera_runtime.stop()
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("camera_error", "expected_code"),
    [
        ("denied", "camera_unavailable"),
        (
            "dshow: GoPro Webcam returned only a black placeholder",
            "gopro_not_streaming",
        ),
    ],
)
def test_camera_unavailable_returns_structured_error(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    camera_error: str,
    expected_code: str,
) -> None:
    def db_override() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_settings] = lambda: Settings(export_dir=tmp_path)
    monkeypatch.setattr("app.main._restore_active_runtime", lambda: None)
    monkeypatch.setattr("app.main._pause_running_race_on_shutdown", lambda: None)
    monkeypatch.setattr(
        "app.api.routes._start_camera_ready",
        lambda *_args, **_kwargs: camera_error,
    )
    try:
        with TestClient(app) as client:
            response = client.put("/api/camera", json={"camera_identifier": "webcam:0"})
            assert response.status_code == 409
            assert response.json()["error"]["code"] == expected_code
    finally:
        app.dependency_overrides.clear()
