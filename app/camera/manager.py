"""Background capture worker with reconnect and bounded buffering."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Event, RLock, Thread, current_thread

from .base import CameraConfig, CameraSource, EndOfStream, Frame
from .buffer import FrameBufferMetrics, LatestFrameBuffer
from .transforms import transform_frame

logger = logging.getLogger(__name__)


class CameraState(str, Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CameraMetrics:
    state: CameraState
    measured_fps: float
    captured_frames: int
    reconnect_count: int
    read_errors: int
    last_error: str | None
    last_frame_monotonic_ns: int | None
    buffer: FrameBufferMetrics


class CameraManager:
    """Own a source in a daemon worker and expose recent frames safely."""

    def __init__(
        self,
        source: CameraSource,
        config: CameraConfig | None = None,
        *,
        reconnect: bool = True,
    ) -> None:
        self.source = source
        self.config = config or getattr(source, "config", None) or CameraConfig()
        self.reconnect = reconnect
        self._buffer = LatestFrameBuffer(self.config.queue_size)
        self._stop = Event()
        self._running = Event()
        self._thread: Thread | None = None
        self._lock = RLock()
        self._state = CameraState.STOPPED
        self._captured_frames = 0
        self._reconnect_count = 0
        self._read_errors = 0
        self._last_error: str | None = None
        self._last_frame_ns: int | None = None
        self._arrival_times: deque[int] = deque(maxlen=240)

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._buffer = LatestFrameBuffer(self.config.queue_size)
            self._stop.clear()
            self._running.clear()
            self._state = CameraState.CONNECTING
            self._last_error = None
            self._thread = Thread(
                target=self._capture_loop,
                name=f"camera-{self.source.identifier}",
                daemon=True,
            )
            self._thread.start()

    def wait_until_running(self, timeout: float | None = None) -> bool:
        """Wait until the first successful open, not merely thread creation."""

        return self._running.wait(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Request shutdown, release the driver, and join the worker."""

        self._stop.set()
        # Releasing here helps unblock a driver read during USB disconnect.
        try:
            self.source.close()
        except Exception:
            logger.exception("Failed to release camera source during shutdown")
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(max(timeout, 0))
            if thread.is_alive():
                logger.warning("Camera worker did not stop within %.1f seconds", timeout)
        self._buffer.close()
        with self._lock:
            worker_alive = bool(thread and thread.is_alive())
            if worker_alive:
                self._state = CameraState.ERROR
                self._last_error = "Camera worker did not stop cleanly"
            else:
                self._state = CameraState.STOPPED
                self._thread = None
            self._running.clear()

    def get_frame(self, timeout: float | None = None, *, latest: bool = True) -> Frame | None:
        return self._buffer.get(timeout, latest=latest)

    def metrics(self) -> CameraMetrics:
        with self._lock:
            fps = self._measure_fps_locked()
            return CameraMetrics(
                state=self._state,
                measured_fps=fps,
                captured_frames=self._captured_frames,
                reconnect_count=self._reconnect_count,
                read_errors=self._read_errors,
                last_error=self._last_error,
                last_frame_monotonic_ns=self._last_frame_ns,
                buffer=self._buffer.metrics(),
            )

    def _set_state(self, state: CameraState, error: BaseException | None = None) -> None:
        with self._lock:
            self._state = state
            if error is not None:
                self._last_error = f"{type(error).__name__}: {error}"
            elif state is CameraState.RUNNING:
                # Preserve the previous diagnostic while reconnecting; clear it
                # only after a source has actually opened successfully.
                self._last_error = None

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._set_state(
                        CameraState.RECONNECTING
                        if self._reconnect_count
                        else CameraState.CONNECTING
                    )
                    self.source.open()
                    self._set_state(CameraState.RUNNING)
                    self._running.set()
                    logger.info("Camera source connected: %s", self.source.identifier)
                    self._read_open_source()
                    break
                except EndOfStream:
                    logger.info("Finite camera source completed: %s", self.source.identifier)
                    break
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    with self._lock:
                        self._read_errors += 1
                    self._set_state(CameraState.ERROR, exc)
                    logger.warning("Camera source error (%s): %s", self.source.identifier, exc)
                    try:
                        self.source.close()
                    except Exception:
                        logger.exception("Failed to close camera source after an error")
                    if not self.reconnect:
                        break
                    with self._lock:
                        self._reconnect_count += 1
                    self._set_state(CameraState.RECONNECTING, exc)
                    if self._stop.wait(self.config.reconnect_interval_s):
                        break
        finally:
            try:
                self.source.close()
            except Exception:
                logger.exception("Failed to close camera source")
            self._buffer.close()
            self._running.clear()
            with self._lock:
                if self._state is not CameraState.ERROR or self._stop.is_set():
                    self._state = CameraState.STOPPED

    def _read_open_source(self) -> None:
        while not self._stop.is_set():
            frame = self.source.read()
            transformed = transform_frame(frame, self.config)
            arrival_ns = time.perf_counter_ns()
            self._buffer.put(transformed)
            with self._lock:
                self._captured_frames += 1
                self._last_frame_ns = frame.captured_monotonic_ns
                self._arrival_times.append(arrival_ns)
                cutoff = arrival_ns - 2_000_000_000
                while self._arrival_times and self._arrival_times[0] < cutoff:
                    self._arrival_times.popleft()

    def _measure_fps_locked(self) -> float:
        if len(self._arrival_times) < 2:
            return 0.0
        duration_ns = self._arrival_times[-1] - self._arrival_times[0]
        return (
            0.0
            if duration_ns <= 0
            else (len(self._arrival_times) - 1) * 1_000_000_000 / duration_ns
        )

    def __enter__(self) -> CameraManager:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


CameraCaptureWorker = CameraManager
