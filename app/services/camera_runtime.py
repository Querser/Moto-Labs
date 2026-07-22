"""Application-level camera preview and frame dispatch orchestration."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from typing import Any

from app.camera import (
    CameraConfig,
    CameraManager,
    CameraSource,
    Frame,
    SyntheticCameraSource,
    SyntheticFrameSpec,
    WebcamCameraSource,
    resolve_camera,
)

logger = logging.getLogger(__name__)
FrameCallback = Callable[[Frame], None]
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 720
PREVIEW_JPEG_QUALITY = 80


def _preview_image(cv2: Any, image: Any) -> Any:
    """Downscale only the browser copy while preserving the CV source frame."""

    if not hasattr(image, "shape") or len(image.shape) < 2:
        return image
    height, width = int(image.shape[0]), int(image.shape[1])
    scale = min(1.0, PREVIEW_MAX_WIDTH / width, PREVIEW_MAX_HEIGHT / height)
    if scale >= 1.0:
        return image
    preview_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, preview_size, interpolation=cv2.INTER_AREA)


def _synthetic_frame(sequence: int) -> SyntheticFrameSpec:
    """Generate an endless, clearly labelled development camera feed."""

    import cv2
    import numpy as np

    width, height = 960, 540
    image = np.full((height, width, 3), (24, 32, 45), dtype=np.uint8)
    # A repeating approach/crossing path with a pause away from the line.
    phase = sequence % 240
    progress = min(1.0, max(0.0, (phase - 20) / 150))
    center_x = int(width * 0.5)
    center_y = int((-0.20 + 1.40 * progress) * height)
    box_width, box_height = 170, 210
    x1, y1 = center_x - box_width // 2, center_y - box_height // 2
    x2, y2 = center_x + box_width // 2, center_y + box_height // 2
    line_y = int(height * 0.68)
    cv2.line(image, (int(width * 0.10), line_y), (int(width * 0.90), line_y), (38, 90, 255), 4)
    cv2.rectangle(image, (x1, y1), (x2, y2), (54, 211, 153), -1)
    cv2.circle(image, (x1 + 30, center_y), 24, (10, 15, 25), -1)
    cv2.circle(image, (x2 - 30, center_y), 24, (10, 15, 25), -1)
    card_x1, card_y1 = center_x - 62, center_y + 22
    card_x2, card_y2 = center_x + 62, center_y + 94
    cv2.rectangle(image, (card_x1, card_y1), (card_x2, card_y2), (245, 245, 245), -1)
    cv2.putText(
        image,
        "777",
        (center_x - 57, center_y + 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.55,
        (5, 5, 5),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "SYNTHETIC DEMO - NOT A REAL CAMERA",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (230, 235, 245),
        2,
        cv2.LINE_AA,
    )
    visible = x2 > 0 and x1 < width and y2 > 0 and y1 < height
    bbox = (
        float(max(0, x1)),
        float(max(0, y1)),
        float(min(width, x2)),
        float(min(height, y2)),
    )
    return SyntheticFrameSpec(
        image=image,
        metadata={
            "synthetic": True,
            "detections": [
                {
                    "bbox": bbox,
                    "label": "motorcycle",
                    "confidence": 0.99,
                    "racing_number": "777",
                    "ocr_confidence": 0.98,
                    "synthetic_id": "demo-777",
                }
            ]
            if visible
            else [],
        },
    )


class CameraRuntime:
    """Own camera worker, latest JPEG, and non-blocking CV callbacks."""

    def __init__(self) -> None:
        self._manager: CameraManager | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest_frame: Frame | None = None
        self._latest_jpeg: bytes | None = None
        self._jpeg_sequence = -1
        self._callbacks: list[FrameCallback] = []
        self._lifecycle_lock = threading.RLock()
        self._processed_frames = 0
        self._encode_errors = 0
        self._processing_latency_ms = 0.0
        self._source_identifier: str | None = None
        self._standby_manager: CameraManager | None = None
        self._standby_identifier: str | None = None

    @property
    def source_identifier(self) -> str | None:
        return self._source_identifier

    @property
    def is_running(self) -> bool:
        return bool(self._manager and self._manager.is_running)

    @property
    def latest_frame(self) -> Frame | None:
        with self._condition:
            return self._latest_frame

    @property
    def frame_age_ms(self) -> float | None:
        """Return the age of the newest captured frame on the monotonic clock."""

        with self._condition:
            frame = self._latest_frame
        if frame is None:
            return None
        return max(0.0, (time.perf_counter_ns() - frame.captured_monotonic_ns) / 1_000_000)

    def latest_jpeg_snapshot(self) -> tuple[int, bytes] | None:
        """Return one complete immutable JPEG without exposing the capture buffer."""

        with self._condition:
            if self._latest_jpeg is None:
                return None
            return self._jpeg_sequence, self._latest_jpeg

    def wait_for_jpeg_snapshot(
        self,
        after_sequence: int = -1,
        *,
        timeout: float = 1.0,
    ) -> tuple[int, bytes] | None:
        """Wait briefly for a newer preview and otherwise return the latest one.

        A browser therefore receives no historical queue: a slow tab always
        jumps straight to the newest encoded frame instead of accumulating lag.
        """

        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while (
                (self._latest_jpeg is None or self._jpeg_sequence <= after_sequence)
                and time.monotonic() < deadline
                and not self._stop.is_set()
            ):
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
            if self._latest_jpeg is None:
                return None
            return self._jpeg_sequence, self._latest_jpeg

    def wait_until_running(self, timeout: float | None = None) -> bool:
        manager = self._manager
        return manager.wait_until_running(timeout) if manager is not None else False

    def wait_until_ready(self, timeout: float = 12.0) -> bool:
        """Wait for an opened source and the first successfully encoded frame."""

        deadline = time.monotonic() + max(0.0, timeout)
        manager = self._manager
        if manager is None:
            return False
        while time.monotonic() < deadline:
            if manager.wait_until_running(min(0.1, max(0.0, deadline - time.monotonic()))):
                break
            if manager.metrics().last_error is not None:
                return False
        else:
            return False
        with self._condition:
            while self._latest_jpeg is None and time.monotonic() < deadline:
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
            return self._latest_jpeg is not None

    def add_callback(self, callback: FrameCallback) -> None:
        with self._condition:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: FrameCallback) -> None:
        with self._condition:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def start(self, source_identifier: str, config: CameraConfig | None = None) -> None:
        """Start a known local source; arbitrary paths are intentionally rejected."""

        with self._lifecycle_lock:
            manager = self._manager
            if (
                source_identifier == self._source_identifier
                and manager is not None
                and manager.state.value == "running"
                and self.frame_age_ms is not None
                and self.frame_age_ms < 1_500
            ):
                return
            selected_config = config or CameraConfig(width=960, height=540, target_fps=30)

            if (
                source_identifier.startswith("webcam:")
                and source_identifier == self._standby_identifier
                and self._standby_manager is not None
                and self._standby_manager.is_running
            ):
                standby = self._standby_manager
                self._standby_manager = None
                self._standby_identifier = None
                self._stop_active_manager(timeout=1.0)
                self._activate_manager(standby, source_identifier)
                logger.info("Warm camera preview restored", extra={"source": source_identifier})
                return

            if (
                source_identifier == "synthetic"
                and manager is not None
                and (self._source_identifier or "").startswith("webcam:")
            ):
                if self._standby_manager is not None:
                    self._standby_manager.stop(timeout=1.0)
                self._stop_dispatch(timeout=1.0)
                self._standby_manager = manager
                self._standby_identifier = self._source_identifier
                self._manager = None
                self._source_identifier = None
            else:
                self._stop_active_manager(timeout=1.5)
                if source_identifier.startswith("webcam:") and self._standby_manager is not None:
                    self._standby_manager.stop(timeout=1.5)
                    self._standby_manager = None
                    self._standby_identifier = None

            if source_identifier == "synthetic":
                source: CameraSource = SyntheticCameraSource(
                    _synthetic_frame,
                    source_id="synthetic",
                    fps=selected_config.target_fps,
                )
            elif source_identifier.startswith("webcam:"):
                raw_index = source_identifier.partition(":")[2]
                if not raw_index.isdigit() or int(raw_index) > 63:
                    raise ValueError("Invalid webcam identifier")
                # Resolve the name and backend together. Windows does not
                # guarantee that PnP list order matches OpenCV camera indexes;
                # GoPro Webcam in particular exists only as a DirectShow source.
                camera_info = resolve_camera(source_identifier)
                camera_index = (
                    camera_info.camera_index
                    if camera_info is not None and camera_info.camera_index is not None
                    else int(raw_index)
                )
                source_config = selected_config
                if (
                    camera_info is not None
                    and camera_info.backend
                    and not selected_config.backend
                ):
                    source_config = replace(selected_config, backend=camera_info.backend)
                source = WebcamCameraSource(
                    camera_index,
                    source_config,
                    device_name=camera_info.device_name if camera_info is not None else None,
                )
            else:
                raise ValueError("Only discovered webcam identifiers and synthetic are allowed")

            new_manager = CameraManager(source, selected_config, reconnect=True)
            new_manager.start()
            self._activate_manager(new_manager, source_identifier)
            logger.info("Camera preview started", extra={"source": source_identifier})

    def _activate_manager(self, manager: CameraManager, source_identifier: str) -> None:
        self._manager = manager
        self._source_identifier = source_identifier
        self._latest_frame = None
        self._latest_jpeg = None
        self._jpeg_sequence = -1
        self._processed_frames = 0
        self._encode_errors = 0
        self._stop = threading.Event()
        stop_event = self._stop
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            args=(manager, stop_event),
            name="camera-frame-dispatch",
            daemon=True,
        )
        self._dispatch_thread.start()

    def _stop_dispatch(self, timeout: float) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._dispatch_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        self._dispatch_thread = None

    def _stop_active_manager(self, timeout: float) -> None:
        manager = self._manager
        self._stop_dispatch(timeout=min(timeout, 1.0))
        self._manager = None
        self._source_identifier = None
        if manager is not None:
            manager.stop(timeout=max(0.0, timeout))

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_active_manager(timeout=5.0)
            standby = self._standby_manager
            self._standby_manager = None
            self._standby_identifier = None
            if standby is not None:
                standby.stop(timeout=5.0)
            with self._condition:
                self._condition.notify_all()

    def metrics(self) -> dict[str, Any]:
        manager = self._manager
        if manager is None:
            return {
                "state": "stopped",
                "source": None,
                "measured_fps": 0.0,
                "processed_fps": 0.0,
                "captured_frames": 0,
                "dropped_frames": 0,
                "frame_queue_size": 0,
                "processing_latency_ms": 0.0,
                "frame_age_ms": None,
                "last_error": None,
                "encode_errors": self._encode_errors,
                "diagnostics": [],
            }
        values = asdict(manager.metrics())
        buffer = values.pop("buffer")
        measured_fps = float(values.get("measured_fps") or 0.0)
        return {
            **values,
            "state": manager.state.value,
            "source": self._source_identifier,
            "dropped_frames": buffer["dropped_frames"],
            "frame_queue_size": buffer["queue_size"],
            "processed_fps": min(measured_fps, self._processed_frames and measured_fps),
            "processing_latency_ms": round(self._processing_latency_ms, 3),
            "frame_age_ms": round(self.frame_age_ms, 3) if self.frame_age_ms is not None else None,
            "encode_errors": self._encode_errors,
            "diagnostics": list(getattr(manager.source, "diagnostics", ())),
        }

    def mjpeg_stream(self) -> Iterator[bytes]:
        """Yield only newly encoded frames so slow clients cannot grow memory."""

        last_sequence = -1
        while True:
            with self._condition:
                if self._jpeg_sequence == last_sequence and not self._stop.is_set():
                    self._condition.wait(timeout=2)
                jpeg = self._latest_jpeg
                sequence = self._jpeg_sequence
                stopped = self._stop.is_set() and not self.is_running
            if jpeg is not None and sequence != last_sequence:
                last_sequence = sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            elif stopped:
                return

    def _dispatch_loop(self, manager: CameraManager, stop_event: threading.Event) -> None:
        import cv2

        while not stop_event.is_set():
            frame = manager.get_frame(timeout=0.5, latest=True)
            if frame is None:
                if not manager.is_running:
                    break
                continue
            start_ns = time.perf_counter_ns()
            # CV and recording callbacks receive the original full-resolution
            # image before any preview-only resize or JPEG work.
            with self._condition:
                callbacks = tuple(self._callbacks)
                self._latest_frame = frame
            for callback in callbacks:
                try:
                    callback(frame)
                except Exception:
                    logger.exception("Frame callback failed")
            preview_image = _preview_image(cv2, frame.image)
            ok, encoded = cv2.imencode(
                ".jpg",
                preview_image,
                [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY],
            )
            if not ok:
                self._encode_errors += 1
                continue
            with self._condition:
                self._latest_jpeg = bytes(encoded)
                self._jpeg_sequence = frame.sequence
                self._condition.notify_all()
            self._processed_frames += 1
            self._processing_latency_ms = (
                time.perf_counter_ns() - frame.captured_monotonic_ns
            ) / 1_000_000
            # The encoder cost is included in latency but never affects capture timestamps.
            _ = start_ns


camera_runtime = CameraRuntime()
