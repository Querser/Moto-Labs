from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.errors import ApiError, api_error_handler
from app.api.routes import router
from app.camera import CameraConfig, EndOfStream, VideoFileSource
from app.config import Settings
from app.models import Race
from app.services.video_runtime import UploadedVideoRuntime
from app.video_uploads import VideoCatalog, VideoUploadError


def _make_video(path: Path, *, frame_count: int = 6, fps: float = 12.0) -> bytes:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore[attr-defined]
        fps,
        (96, 64),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index * 5, 16), (index * 5 + 20, 48), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_video_catalog_accepts_valid_file_and_rejects_unsafe_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    payload = _make_video(source)
    settings = Settings(video_upload_dir=tmp_path / "uploads", video_upload_max_bytes=5_000_000)
    catalog = VideoCatalog(settings)

    uploaded = catalog.save(io.BytesIO(payload), "race.avi", "video/x-msvideo")
    assert uploaded.width == 96
    assert uploaded.height == 64
    assert uploaded.frame_count == 6
    assert uploaded.identifier == f"video:{uploaded.id}"
    assert catalog.get(uploaded.id) == uploaded
    assert catalog.path_for(uploaded).name != "race.avi"

    with pytest.raises(VideoUploadError, match="MP4"):
        catalog.save(io.BytesIO(payload), "race.exe", "application/octet-stream")
    with pytest.raises(VideoUploadError, match="пуст"):
        catalog.save(io.BytesIO(b""), "empty.avi", "video/x-msvideo")
    with pytest.raises(VideoUploadError, match=r"повреждено|открывается"):
        catalog.save(io.BytesIO(b"not-a-video"), "broken.avi", "video/x-msvideo")


def test_video_upload_api_validates_valid_unsupported_and_corrupt_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "api.avi"
    payload = _make_video(source)
    settings = Settings(video_upload_dir=tmp_path / "uploads", video_upload_max_bytes=5_000_000)
    application = FastAPI()
    application.include_router(router)
    application.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    from app.config import get_settings

    application.dependency_overrides[get_settings] = lambda: settings
    with TestClient(application) as client:
        valid = client.post(
            "/api/videos",
            files={"file": ("race.avi", payload, "video/x-msvideo")},
        )
        assert valid.status_code == 201
        assert valid.json()["frame_count"] == 6
        video_id = valid.json()["id"]
        media = client.get(f"/api/videos/{video_id}/media")
        assert media.status_code == 200
        assert media.headers["content-type"].startswith("video/")
        assert media.content == payload
        media_range = client.get(
            f"/api/videos/{video_id}/media",
            headers={"Range": "bytes=0-99"},
        )
        assert media_range.status_code == 206
        assert media_range.headers["content-range"].startswith("bytes 0-99/")
        assert media_range.content == payload[:100]
        unsupported = client.post(
            "/api/videos",
            files={"file": ("race.txt", payload, "text/plain")},
        )
        assert unsupported.status_code == 422
        corrupt = client.post(
            "/api/videos",
            files={"file": ("broken.avi", b"broken", "video/x-msvideo")},
        )
        assert corrupt.status_code == 422


def test_video_source_uses_source_timeline_instead_of_processing_speed(tmp_path: Path) -> None:
    path = tmp_path / "timeline.avi"
    _make_video(path, frame_count=3, fps=10.0)
    origin_ns = 4_000_000_000
    origin_utc = datetime(2026, 1, 2, tzinfo=timezone.utc)
    source = VideoFileSource(
        path,
        config=CameraConfig(target_fps=10),
        realtime=False,
        timeline_origin_ns=origin_ns,
        timeline_origin_utc=origin_utc,
    )
    source.open()
    try:
        frames = [source.read(), source.read(), source.read()]
        with pytest.raises(EndOfStream):
            source.read()
    finally:
        source.close()
    assert frames[0].captured_monotonic_ns == origin_ns
    assert frames[1].captured_monotonic_ns - frames[0].captured_monotonic_ns == 100_000_000
    assert frames[2].metadata["source_type"] == "uploaded_video"
    assert frames[2].metadata["original_resolution"] == (96, 64)


def test_video_source_can_resume_from_a_source_timeline_position(tmp_path: Path) -> None:
    path = tmp_path / "resume.avi"
    _make_video(path, frame_count=8, fps=10.0)
    source = VideoFileSource(
        path,
        config=CameraConfig(target_fps=10),
        realtime=False,
        timeline_origin_ns=5_000_000_000,
        timeline_origin_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        start_position_ms=300.0,
    )
    source.open()
    try:
        frame = source.read()
    finally:
        source.close()
    assert frame.sequence == 3
    assert frame.captured_monotonic_ns == 5_300_000_000


def test_uploaded_runtime_scans_then_processes_candidate_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "runtime.avi"
    payload = _make_video(source, frame_count=5, fps=30.0)
    settings = Settings(video_upload_dir=tmp_path / "uploads", video_upload_max_bytes=5_000_000)
    uploaded = VideoCatalog(settings).save(
        io.BytesIO(payload), "runtime.avi", "video/x-msvideo"
    )

    class FakeVisionRuntime:
        def __init__(self) -> None:
            self.frames: list[Any] = []
            self.analyzed: list[int] = []
            self.evidence_only: list[int] = []

        def configure_source(self, source_identifier: str, *, race_id: int | None) -> None:
            assert source_identifier == uploaded.identifier
            assert race_id == 1

        def process_frame_sync(self, frame: Any) -> None:
            self.frames.append(frame)
            self.analyzed.append(frame.sequence)

        def scan_frame_sync(self, frame: Any) -> bool:
            return frame.sequence == 0

        def collect_evidence_frame_sync(self, frame: Any) -> None:
            self.frames.append(frame)
            self.evidence_only.append(frame.sequence)

        def status(self) -> dict[str, Any]:
            return {
                "state": "vision-state-must-not-replace-video-state",
                "race_id": 999,
                "tracks": [],
                "boards": [],
                "digit_regions": [],
            }

        def stop(self) -> None:
            return None

    fake = FakeVisionRuntime()
    monkeypatch.setattr("app.services.video_runtime.get_settings", lambda: settings)
    monkeypatch.setattr("app.services.video_runtime.vision_runtime", fake)
    race = Race(
        id=1,
        name="Uploaded",
        required_laps=2,
        camera_identifier=uploaded.identifier,
        # SQLite returns this persisted UTC value without tzinfo on Windows.
        started_at_utc=datetime(2026, 1, 2),
    )
    runtime = UploadedVideoRuntime()
    runtime.start(uploaded.id, race, timeline_origin_ns=8_000_000_000)
    deadline = time.monotonic() + 5
    while runtime.status()["state"] not in {"completed", "error"} and time.monotonic() < deadline:
        time.sleep(0.01)
    status = runtime.status()
    assert status["state"] == "completed"
    assert status["race_id"] == 1
    assert status["processed_frames"] == 5
    assert len(fake.frames) == 5
    assert [frame.sequence for frame in fake.frames] == [0, 1, 2, 3, 4]
    # Pass one only locates a broad window. Pass two preserves every source
    # frame inside that window for exact tracking and timestamp interpolation.
    assert fake.analyzed == [0, 1, 2, 3, 4]
    assert fake.evidence_only == []
    runtime.restart()
    deadline = time.monotonic() + 5
    while runtime.status()["state"] not in {"completed", "error"} and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runtime.status()["processed_frames"] == 5
    runtime.stop()
    assert len(fake.frames) == 10
    assert fake.analyzed == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert fake.evidence_only == []
