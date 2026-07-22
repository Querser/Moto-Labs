"""Concrete webcam, video-file, and dependency-free synthetic sources."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any

from .base import (
    CameraConfig,
    CameraReadError,
    CameraSource,
    CameraUnavailableError,
    EndOfStream,
    Frame,
)


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise CameraUnavailableError(
            "OpenCV is required for webcam/video capture; install opencv-python-headless"
        ) from exc
    return cv2


def _backend_code(cv2: Any, backend: str | None) -> int | None:
    if backend is None or backend.lower() in {"", "auto", "any"}:
        return None
    normalized = backend.upper().removeprefix("CAP_")
    value = getattr(cv2, f"CAP_{normalized}", None)
    if value is None:
        raise CameraUnavailableError(f"Unknown OpenCV camera backend: {backend}")
    return int(value)


def _has_visible_content(image: Any) -> bool:
    """Reject the exact-black placeholder emitted by an idle GoPro driver."""

    if image is None:
        return False
    if not hasattr(image, "max") or not hasattr(image, "std"):
        return True
    try:
        return float(image.max()) > 4.0 and float(image.std()) > 0.5
    except (TypeError, ValueError):
        return True


class WebcamCameraSource(CameraSource):
    """OpenCV-backed USB or built-in webcam source."""

    def __init__(
        self,
        camera_id: int | str = 0,
        config: CameraConfig | None = None,
        *,
        device_name: str | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = config or CameraConfig()
        self.device_name = device_name
        self._capture: Any | None = None
        self._sequence = 0
        self._lock = Lock()
        self._diagnostics: list[str] = []
        self._prefetched_image: Any | None = None

    @property
    def identifier(self) -> str:
        return f"webcam:{self.camera_id}"

    @property
    def is_open(self) -> bool:
        with self._lock:
            return bool(self._capture is not None and self._capture.isOpened())

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def open(self) -> None:
        self.close()
        cv2 = _load_cv2()
        is_gopro_webcam = "gopro" in (self.device_name or "").casefold()
        if self.config.backend:
            candidates = [(self.config.backend, _backend_code(cv2, self.config.backend))]
        elif sys.platform == "win32":
            # DirectShow opens and applies format changes much faster on the
            # tested USB webcam. Media Foundation and CAP_ANY remain fallbacks
            # because Windows camera drivers vary substantially.
            candidates = [
                ("dshow", int(cv2.CAP_DSHOW)),
                ("msmf", int(cv2.CAP_MSMF)),
                ("auto", None),
            ]
        else:
            candidates = [("auto", None)]

        capture: Any | None = None
        prefetched_image: Any | None = None
        failures: list[str] = []
        for backend_name, backend_code in candidates:
            candidate: Any | None = None
            try:
                candidate = (
                    cv2.VideoCapture(self.camera_id)
                    if backend_code is None
                    else cv2.VideoCapture(self.camera_id, backend_code)
                )
                if candidate.isOpened():
                    if not is_gopro_webcam:
                        # Physical USB webcams usually need MJPG to sustain FPS.
                        # GoPro Webcam is a virtual DirectShow source and rejects
                        # format requests that are not exposed by its driver, so
                        # it deliberately stays in the driver-native mode.
                        if sys.platform == "win32" and hasattr(cv2, "CAP_PROP_FOURCC"):
                            candidate.set(
                                cv2.CAP_PROP_FOURCC,
                                float(cv2.VideoWriter_fourcc(*"MJPG")),
                            )
                        requested = (
                            (cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width)),
                            (cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height)),
                            (cv2.CAP_PROP_FPS, float(self.config.target_fps)),
                        )
                        for property_id, desired in requested:
                            current = float(candidate.get(property_id))
                            if current <= 0 or abs(current - desired) > 0.5:
                                candidate.set(property_id, desired)
                    candidate.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
                    first_image = None
                    # When the desktop utility is installed but the physical
                    # camera is absent, its DirectShow device still opens and
                    # emits exact-black 1920x1080 frames. Wait briefly for a
                    # real signal instead of reporting that placeholder as a
                    # working camera.
                    initial_reads = 180 if is_gopro_webcam else 1
                    first_read_started = time.monotonic()
                    black_gopro_frame = False
                    for _ in range(initial_reads):
                        ok, image = candidate.read()
                        if ok and image is not None:
                            if not is_gopro_webcam or _has_visible_content(image):
                                first_image = image
                                break
                            black_gopro_frame = True
                        if is_gopro_webcam and time.monotonic() - first_read_started >= 4.0:
                            break
                    if first_image is not None:
                        capture = candidate
                        prefetched_image = first_image
                        actual_width = int(candidate.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(candidate.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = float(candidate.get(cv2.CAP_PROP_FPS))
                        self._diagnostics = [
                            f"Capture device: {self.device_name or self.camera_id}",
                            f"Capture backend: {backend_name}",
                            (
                                f"Capture format: {actual_width}x{actual_height} "
                                f"@ {actual_fps:.2f} FPS"
                            ),
                        ]
                        if is_gopro_webcam:
                            self._diagnostics.append("Capture profile: GoPro driver-native")
                        break
                    if black_gopro_frame:
                        failures.append(
                            f"{backend_name}: GoPro Webcam returned only a black placeholder; "
                            "connect and power on the GoPro in GoPro Connect/Webcam mode"
                        )
                        continue
                    failures.append(f"{backend_name}: opened but returned no frame")
                    continue
                failures.append(f"{backend_name}: unavailable")
            except Exception as exc:
                failures.append(f"{backend_name}: {exc}")
            finally:
                if candidate is not None and candidate is not capture:
                    candidate.release()
        if capture is None:
            details = "; ".join(failures)
            raise CameraUnavailableError(
                f"Camera {self.camera_id!r} is disconnected, busy, or denied ({details})"
            )

        if self.config.exposure is not None and not capture.set(
            cv2.CAP_PROP_EXPOSURE, float(self.config.exposure)
        ):
            self._diagnostics.append("Camera backend rejected exposure setting")
        if self.config.autofocus is not None and not capture.set(
            cv2.CAP_PROP_AUTOFOCUS, 1.0 if self.config.autofocus else 0.0
        ):
            self._diagnostics.append("Camera backend rejected autofocus setting")
        with self._lock:
            self._capture = capture
            self._sequence = 0
            self._prefetched_image = prefetched_image

    def read(self) -> Frame:
        with self._lock:
            capture = self._capture
            image, self._prefetched_image = self._prefetched_image, None
        if capture is None or not capture.isOpened():
            raise CameraReadError("Camera is not open")
        if image is None:
            try:
                ok, image = capture.read()
            except Exception as exc:
                raise CameraReadError(f"Camera read failed: {exc}") from exc
        else:
            ok = True
        if not ok or image is None:
            raise CameraReadError("Camera returned no frame; it may have disconnected")
        # Timestamp immediately after the blocking driver acquisition returns.
        frame = Frame.captured(
            image,
            sequence=self._sequence,
            source_id=self.identifier,
            metadata={
                "requested_size": (self.config.width, self.config.height),
                "device_name": self.device_name,
                "backend": self.config.backend,
            },
        )
        self._sequence += 1
        return frame

    def close(self) -> None:
        with self._lock:
            capture, self._capture = self._capture, None
            self._prefetched_image = None
        if capture is not None:
            capture.release()


# Short public alias used by setup/API code.
WebcamSource = WebcamCameraSource


class VideoFileSource(CameraSource):
    """Prerecorded video source for repeatable development and testing."""

    def __init__(
        self,
        path: str | Path,
        *,
        config: CameraConfig | None = None,
        loop: bool = False,
        realtime: bool = True,
        source_id: str | None = None,
        timeline_origin_ns: int | None = None,
        timeline_origin_utc: datetime | None = None,
        start_position_ms: float = 0.0,
    ) -> None:
        self.path = Path(path).expanduser()
        self.config = config or CameraConfig()
        self.loop = loop
        self.realtime = realtime
        self._source_id = source_id
        self.timeline_origin_ns = timeline_origin_ns
        self.timeline_origin_utc = timeline_origin_utc
        self.start_position_ms = max(0.0, float(start_position_ms))
        self._capture: Any | None = None
        self._sequence = 0
        self._next_due_ns: int | None = None
        self._fps = self.config.target_fps
        self._closed = Event()

    @property
    def identifier(self) -> str:
        return self._source_id or f"video:{self.path.name}"

    @property
    def is_open(self) -> bool:
        return bool(self._capture is not None and self._capture.isOpened())

    def open(self) -> None:
        self.close()
        if not self.path.is_file():
            raise CameraUnavailableError(f"Video file does not exist: {self.path}")
        cv2 = _load_cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise CameraUnavailableError(f"Could not open video file: {self.path.name}")
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._fps = reported_fps if reported_fps > 0 else self.config.target_fps
        start_frame = max(0, round(self.start_position_ms * self._fps / 1000.0))
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        self._capture = capture
        self._sequence = start_frame
        self._next_due_ns = time.perf_counter_ns()
        self._closed.clear()

    def read(self) -> Frame:
        capture = self._capture
        if capture is None or not capture.isOpened():
            raise CameraReadError("Video source is not open")
        if self.realtime and self._next_due_ns is not None:
            delay_ns = self._next_due_ns - time.perf_counter_ns()
            if delay_ns > 0 and self._closed.wait(delay_ns / 1_000_000_000):
                raise CameraReadError("Video source was closed")
        ok, image = capture.read()
        if not ok or image is None:
            if not self.loop:
                raise EndOfStream(f"End of video: {self.path.name}")
            cv2 = _load_cv2()
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = capture.read()
            if not ok or image is None:
                raise CameraReadError(f"Video cannot be rewound: {self.path.name}")
        cv2 = _load_cv2()
        reported_position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        fallback_position_ms = self._sequence * 1000.0 / max(self._fps, 0.001)
        # OpenCV's H.264 backends may report the following presentation time or
        # even briefly move POS_MSEC backwards around reordered frames.  A frame
        # index divided by the stream FPS is strictly monotonic and is therefore
        # the authoritative race clock.  Keep the decoder value only as debug
        # metadata so it can be compared on unusual variable-frame-rate files.
        position_ms = fallback_position_ms
        origin_ns = self.timeline_origin_ns
        origin_utc = self.timeline_origin_utc
        if origin_ns is None or origin_utc is None:
            frame = Frame.captured(
                image,
                sequence=self._sequence,
                source_id=self.identifier,
                metadata={
                    "video_path": self.path.name,
                    "source_fps": self._fps,
                    "source_type": "uploaded_video",
                    "video_position_ms": position_ms,
                    "decoder_position_ms": reported_position_ms,
                    "original_resolution": (int(image.shape[1]), int(image.shape[0])),
                },
            )
        else:
            offset_ns = max(0, round(position_ms * 1_000_000))
            frame = Frame(
                image=image,
                sequence=self._sequence,
                source_id=self.identifier,
                captured_monotonic_ns=origin_ns + offset_ns,
                captured_at_utc=origin_utc + timedelta(microseconds=offset_ns / 1_000),
                metadata={
                    "video_path": self.path.name,
                    "source_fps": self._fps,
                    "source_type": "uploaded_video",
                    "video_position_ms": position_ms,
                    "decoder_position_ms": reported_position_ms,
                    "original_resolution": (int(image.shape[1]), int(image.shape[0])),
                },
            )
        self._sequence += 1
        if self.realtime:
            interval_ns = max(1, int(1_000_000_000 / max(self._fps, 0.001)))
            self._next_due_ns = max(
                (self._next_due_ns or frame.captured_monotonic_ns) + interval_ns,
                frame.captured_monotonic_ns,
            )
        return frame

    def close(self) -> None:
        self._closed.set()
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


VideoSource = VideoFileSource


@dataclass(frozen=True, slots=True)
class SyntheticFrameSpec:
    image: Any = None
    metadata: Mapping[str, Any] | None = None
    captured_monotonic_ns: int | None = None
    captured_at_utc: datetime | None = None


SyntheticItem = Any | Frame | SyntheticFrameSpec


class SyntheticCameraSource(CameraSource):
    """Deterministic source with no OpenCV or physical camera requirement."""

    def __init__(
        self,
        frames: Iterable[SyntheticItem] | Callable[[int], SyntheticItem | None],
        *,
        source_id: str = "synthetic",
        fps: float = 0,
        repeat: bool = False,
    ) -> None:
        if fps < 0:
            raise ValueError("fps cannot be negative")
        self._factory = frames if callable(frames) else None
        self._items = None if callable(frames) else tuple(frames)
        self._iterator: Iterator[SyntheticItem] | None = None
        self._identifier = source_id
        self._fps = fps
        self._repeat = repeat
        self._open = False
        self._sequence = 0
        self._next_due_ns: int | None = None
        self._closed = Event()

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._iterator = iter(self._items or ())
        self._open = True
        self._closed.clear()
        self._sequence = 0
        self._next_due_ns = time.perf_counter_ns()

    def _next_item(self) -> SyntheticItem:
        if self._factory is not None:
            value = self._factory(self._sequence)
            if value is None:
                raise EndOfStream("Synthetic generator completed")
            return value
        assert self._iterator is not None
        try:
            return next(self._iterator)
        except StopIteration:
            if not self._repeat or not self._items:
                raise EndOfStream("Synthetic sequence completed") from None
            self._iterator = iter(self._items)
            return next(self._iterator)

    def read(self) -> Frame:
        if not self._open:
            raise CameraReadError("Synthetic source is not open")
        if self._fps > 0 and self._next_due_ns is not None:
            delay_ns = self._next_due_ns - time.perf_counter_ns()
            if delay_ns > 0 and self._closed.wait(delay_ns / 1_000_000_000):
                raise CameraReadError("Synthetic source was closed")
        item = self._next_item()
        if isinstance(item, Frame):
            frame = item
        elif isinstance(item, SyntheticFrameSpec):
            monotonic_ns = item.captured_monotonic_ns
            wall_clock = item.captured_at_utc
            frame = Frame(
                image=item.image,
                sequence=self._sequence,
                source_id=self.identifier,
                captured_monotonic_ns=(
                    time.perf_counter_ns() if monotonic_ns is None else monotonic_ns
                ),
                captured_at_utc=(datetime.now(timezone.utc) if wall_clock is None else wall_clock),
                metadata=item.metadata or {},
            )
        else:
            frame = Frame.captured(
                item,
                sequence=self._sequence,
                source_id=self.identifier,
                metadata={"synthetic": True},
            )
        self._sequence += 1
        if self._fps > 0:
            interval_ns = int(1_000_000_000 / self._fps)
            self._next_due_ns = (self._next_due_ns or time.perf_counter_ns()) + interval_ns
        return frame

    def close(self) -> None:
        self._closed.set()
        self._open = False
        self._iterator = None


class MockCameraSource(SyntheticCameraSource):
    """Semantic alias for tests and dependency injection."""


MockSource = MockCameraSource
