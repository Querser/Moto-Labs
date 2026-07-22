from __future__ import annotations

import cv2
import numpy as np
import pytest

from app import main
from app.camera import CameraInfo
from app.services.camera_runtime import _preview_image


def test_preview_downscale_does_not_modify_full_resolution_source() -> None:
    source = np.zeros((1080, 1920, 3), dtype=np.uint8)

    preview = _preview_image(cv2, source)

    assert source.shape == (1080, 1920, 3)
    assert preview.shape == (720, 1280, 3)


def test_preview_keeps_smaller_frames_without_an_extra_copy() -> None:
    source = np.zeros((480, 640, 3), dtype=np.uint8)

    assert _preview_image(cv2, source) is source


def test_saved_webcam_restore_retries_after_virtual_driver_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[str] = []
    readiness = iter((False, True))
    sleeps: list[float] = []

    monkeypatch.setattr(
        main.camera_runtime,
        "start",
        lambda identifier, _config: starts.append(identifier),
    )
    monkeypatch.setattr(
        main,
        "resolve_camera",
        lambda _identifier: CameraInfo(
            identifier="webcam:1",
            label="GoPro Webcam",
            source_type="webcam",
            camera_index=1,
            device_name="GoPro Webcam",
        ),
    )
    monkeypatch.setattr(
        main.camera_runtime,
        "wait_until_ready",
        lambda timeout: next(readiness),
    )
    monkeypatch.setattr(main.camera_runtime, "stop", lambda: None)
    monkeypatch.setattr(main.time, "sleep", sleeps.append)

    assert main._start_restored_camera("webcam:1") is True
    assert starts == ["webcam:1", "webcam:1"]
    assert sleeps == [2.0]
