"""Best-effort local camera discovery with backend-correct OpenCV indexes."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from .sources import _backend_code


def _windows_camera_names() -> list[str]:
    """Fallback enumeration for installations missing the native helper."""

    if sys.platform != "win32":
        return []
    command = (
        "Get-PnpDevice -PresentOnly -Class Camera,Image -ErrorAction SilentlyContinue | "
        "Where-Object Status -eq 'OK' | "
        "Select-Object -ExpandProperty FriendlyName | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return []
        decoded = json.loads(completed.stdout)
        values = [decoded] if isinstance(decoded, str) else decoded
        return [str(value).strip() for value in values if str(value).strip()]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []


def _is_integrated_camera(name: str) -> bool:
    normalized = name.casefold()
    return any(marker in normalized for marker in ("integrated", "built-in", "builtin"))


def _is_gopro_camera(name: str) -> bool:
    return "gopro" in name.casefold()


def _camera_display_label(name: str) -> str:
    if _is_gopro_camera(name):
        return f"GoPro Webcam (виртуальная камера) — {name}"
    if _is_integrated_camera(name):
        return f"Встроенная камера — {name}"
    # Names such as "HD Webcam" do not reveal whether a device is built in.
    # Keeping the exact driver name is more truthful than guessing from order.
    return f"Камера — {name}"


@dataclass(frozen=True, slots=True)
class CameraInfo:
    identifier: str
    label: str
    source_type: str
    camera_index: int | None = None
    backend: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    device_name: str | None = None
    device_path: str | None = None


@dataclass(frozen=True, slots=True)
class _NamedCamera:
    index: int
    name: str
    backend: str
    path: str = ""


def _default_backend() -> str | None:
    if sys.platform == "win32":
        # GoPro Webcam is exposed as a DirectShow virtual camera on Windows.
        return "dshow"
    if sys.platform == "darwin":
        return "avfoundation"
    if sys.platform.startswith("linux"):
        return "v4l2"
    return None


def _backend_name(cv2: Any, backend_code: int) -> str:
    known = {
        int(getattr(cv2, "CAP_DSHOW", -1)): "dshow",
        int(getattr(cv2, "CAP_MSMF", -2)): "msmf",
        int(getattr(cv2, "CAP_AVFOUNDATION", -3)): "avfoundation",
        int(getattr(cv2, "CAP_V4L2", -4)): "v4l2",
        int(getattr(cv2, "CAP_GSTREAMER", -5)): "gstreamer",
    }
    return known.get(int(backend_code), "auto")


def _named_camera_devices(backend: str | None) -> list[_NamedCamera]:
    """Return names paired with indexes from the same OpenCV backend."""

    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras
    except (ImportError, OSError):
        return []

    try:
        backend_code = _backend_code(cv2, backend)
        devices = enumerate_cameras() if backend_code is None else enumerate_cameras(backend_code)
    except Exception:
        return []

    found: list[_NamedCamera] = []
    seen: set[tuple[int, str]] = set()
    for device in devices:
        name = str(getattr(device, "name", "")).strip()
        index = int(device.index)
        actual_backend = _backend_name(cv2, int(getattr(device, "backend", backend_code or 0)))
        key = (index, actual_backend)
        if not name or key in seen:
            continue
        seen.add(key)
        found.append(
            _NamedCamera(
                index=index,
                name=name,
                backend=actual_backend,
                path=str(getattr(device, "path", "") or ""),
            )
        )
    return found


def discover_cameras(
    *,
    max_devices: int = 8,
    backend: str | None = None,
    include_synthetic: bool = True,
) -> list[CameraInfo]:
    """Enumerate cameras without opening them or guessing Windows indexes."""

    if not 0 <= max_devices <= 64:
        raise ValueError("max_devices must be between 0 and 64")
    found: list[CameraInfo] = []
    selected_backend = backend or _default_backend()

    named_devices = _named_camera_devices(selected_backend)
    if named_devices:
        for device in named_devices[:max_devices]:
            found.append(
                CameraInfo(
                    identifier=f"webcam:{device.index}",
                    label=_camera_display_label(device.name),
                    source_type="webcam",
                    camera_index=device.index,
                    backend=device.backend,
                    device_name=device.name,
                    device_path=device.path or None,
                )
            )
        found.sort(
            key=lambda item: (
                not _is_integrated_camera(item.device_name or ""),
                not _is_gopro_camera(item.device_name or ""),
                (item.device_name or "").casefold(),
            )
        )
    else:
        # PnP is a name-only fallback. It cannot guarantee name/index mapping,
        # so exact backend enumeration above is always preferred.
        windows_names = _windows_camera_names()
        if windows_names:
            found.extend(
                CameraInfo(
                    identifier=f"webcam:{index}",
                    label=_camera_display_label(name),
                    source_type="webcam",
                    camera_index=index,
                    backend=selected_backend,
                    device_name=name,
                )
                for index, name in enumerate(windows_names[:max_devices])
            )

    cv2: Any
    try:
        import cv2 as cv2_module
    except ImportError:
        cv2 = None
    else:
        cv2 = cv2_module
    if cv2 is not None and not found:
        backend_code = _backend_code(cv2, selected_backend)
        for index in range(max_devices):
            capture: Any | None = None
            try:
                capture = (
                    cv2.VideoCapture(index)
                    if backend_code is None
                    else cv2.VideoCapture(index, backend_code)
                )
                if not capture.isOpened():
                    continue
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
                fps_value = float(capture.get(cv2.CAP_PROP_FPS))
                found.append(
                    CameraInfo(
                        identifier=f"webcam:{index}",
                        label=f"Камера {index + 1}" + (
                            f" — {width}x{height}" if width and height else ""
                        ),
                        source_type="webcam",
                        camera_index=index,
                        backend=selected_backend,
                        width=width,
                        height=height,
                        fps=fps_value if fps_value > 0 else None,
                    )
                )
            except Exception:
                continue
            finally:
                if capture is not None:
                    capture.release()
    if include_synthetic:
        found.append(
            CameraInfo(
                identifier="synthetic",
                label="Демо-анимация (не камера)",
                source_type="synthetic",
            )
        )
    return found


def resolve_camera(identifier: str, *, max_devices: int = 64) -> CameraInfo | None:
    """Resolve a saved identifier to its current backend/name metadata."""

    return next(
        (
            camera
            for camera in discover_cameras(
                max_devices=max_devices,
                include_synthetic=False,
            )
            if camera.identifier == identifier
        ),
        None,
    )
