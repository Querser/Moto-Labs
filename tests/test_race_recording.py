from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.camera import Frame
from app.services.race_recording import RaceRecordingService


def test_live_race_recording_writes_readable_video_without_frame_loss(
    tmp_path: Path,
) -> None:
    service = RaceRecordingService(queue_size=16)
    output = service.start(7, "webcam:0", target_fps=30, directory=tmp_path)
    for sequence in range(24):
        image = np.full((96, 128, 3), sequence * 8, dtype=np.uint8)
        service.process_frame(
            Frame(
                image=image,
                sequence=sequence,
                source_id="webcam:0",
                captured_monotonic_ns=sequence + 1,
                captured_at_utc=datetime.now(timezone.utc),
            )
        )
        time.sleep(0.002)
    service.stop()

    capture = cv2.VideoCapture(str(output))
    try:
        assert output.is_file() and output.stat().st_size > 0
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
    finally:
        capture.release()
    status = service.status()
    assert status["active"] is False
    assert status["frames_written"] == 24
    assert status["dropped_frames"] == 0
    assert service.resolve_recording(7, output.name, directory=tmp_path) == output
    assert service.resolve_recording(7, "../escape.mp4", directory=tmp_path) is None
