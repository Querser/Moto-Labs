from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.camera import CameraConfig, CameraUnavailableError, WebcamCameraSource, sources


class _FakeCapture:
    def __init__(self) -> None:
        self.opened = True
        self.set_calls: list[tuple[int, float]] = []
        self.values = {1: 1920.0, 2: 1080.0, 3: 30.0}

    def isOpened(self) -> bool:
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        self.values[property_id] = value
        return True

    def get(self, property_id: int) -> float:
        return self.values.get(property_id, 0.0)

    def read(self) -> tuple[bool, object]:
        return True, object()

    def release(self) -> None:
        self.opened = False


class _FakeCv2:
    CAP_DSHOW = 700
    CAP_MSMF = 1400
    CAP_PROP_FRAME_WIDTH = 1
    CAP_PROP_FRAME_HEIGHT = 2
    CAP_PROP_FPS = 3
    CAP_PROP_FOURCC = 4
    CAP_PROP_BUFFERSIZE = 5
    CAP_PROP_EXPOSURE = 6
    CAP_PROP_AUTOFOCUS = 7

    def __init__(self) -> None:
        self.captures: list[_FakeCapture] = []

    def VideoCapture(self, *_args: Any) -> _FakeCapture:
        capture = _FakeCapture()
        self.captures.append(capture)
        return capture

    @staticmethod
    def VideoWriter_fourcc(*_args: str) -> int:
        return 1234


class _BlackCapture(_FakeCapture):
    def read(self) -> tuple[bool, object]:
        return True, np.zeros((24, 32, 3), dtype=np.uint8)


class _BlackCv2(_FakeCv2):
    def VideoCapture(self, *_args: Any) -> _FakeCapture:
        capture = _BlackCapture()
        self.captures.append(capture)
        return capture


def test_gopro_uses_driver_native_directshow_format(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cv2 = _FakeCv2()
    monkeypatch.setattr(sources, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(sources.sys, "platform", "win32")
    source = WebcamCameraSource(
        1,
        CameraConfig(width=640, height=480, target_fps=30, backend="dshow"),
        device_name="GoPro Webcam",
    )

    source.open()

    property_ids = [item[0] for item in fake_cv2.captures[0].set_calls]
    assert _FakeCv2.CAP_PROP_BUFFERSIZE in property_ids
    assert _FakeCv2.CAP_PROP_FOURCC not in property_ids
    assert _FakeCv2.CAP_PROP_FRAME_WIDTH not in property_ids
    assert "Capture profile: GoPro driver-native" in source.diagnostics
    source.close()


def test_physical_webcam_keeps_low_bandwidth_mjpg_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cv2 = _FakeCv2()
    monkeypatch.setattr(sources, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(sources.sys, "platform", "win32")
    source = WebcamCameraSource(
        0,
        CameraConfig(width=640, height=480, target_fps=30, backend="dshow"),
        device_name="HD Webcam",
    )

    source.open()

    property_ids = [item[0] for item in fake_cv2.captures[0].set_calls]
    assert _FakeCv2.CAP_PROP_FOURCC in property_ids
    assert _FakeCv2.CAP_PROP_FRAME_WIDTH in property_ids
    source.close()


def test_idle_gopro_black_placeholder_is_not_reported_as_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cv2 = _BlackCv2()
    monkeypatch.setattr(sources, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(sources.sys, "platform", "win32")
    source = WebcamCameraSource(
        1,
        CameraConfig(backend="dshow"),
        device_name="GoPro Webcam",
    )

    with pytest.raises(CameraUnavailableError, match="black placeholder"):
        source.open()
