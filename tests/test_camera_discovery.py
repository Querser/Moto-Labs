from __future__ import annotations

import pytest

from app.camera import discovery


def test_backend_enumeration_preserves_real_index_and_identifies_gopro(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discovery.sys, "platform", "win32")
    monkeypatch.setattr(
        discovery,
        "_named_camera_devices",
        lambda _backend: [
            discovery._NamedCamera(0, "Integrated Camera", "dshow", "usb-path"),
            discovery._NamedCamera(1, "GoPro Webcam", "dshow", ""),
        ],
    )

    cameras = discovery.discover_cameras(max_devices=2, include_synthetic=False)

    assert [camera.identifier for camera in cameras] == ["webcam:0", "webcam:1"]
    assert cameras[0].label == "Встроенная камера — Integrated Camera"
    assert cameras[1].label == "GoPro Webcam (виртуальная камера) — GoPro Webcam"
    assert cameras[1].camera_index == 1
    assert cameras[1].backend == "dshow"
    assert cameras[1].device_name == "GoPro Webcam"


def test_windows_pnp_names_remain_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery.sys, "platform", "win32")
    monkeypatch.setattr(discovery, "_named_camera_devices", lambda _backend: [])
    monkeypatch.setattr(discovery, "_windows_camera_names", lambda: ["HD Webcam"])

    cameras = discovery.discover_cameras(max_devices=1, include_synthetic=False)

    assert cameras[0].identifier == "webcam:0"
    assert cameras[0].device_name == "HD Webcam"
    assert cameras[0].backend == "dshow"
