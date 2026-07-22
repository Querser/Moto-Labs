"""Thread-safe bounded latest-frame buffer."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Condition

from .base import Frame


@dataclass(frozen=True, slots=True)
class FrameBufferMetrics:
    capacity: int
    queue_size: int
    enqueued_frames: int
    dequeued_frames: int
    dropped_frames: int


class LatestFrameBuffer:
    """A bounded queue that drops stale frames under processing backpressure."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._frames: deque[Frame] = deque()
        self._condition = Condition()
        self._closed = False
        self._enqueued = 0
        self._dequeued = 0
        self._dropped = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def put(self, frame: Frame) -> bool:
        """Enqueue a frame; return ``True`` when an old frame was dropped."""

        with self._condition:
            if self._closed:
                return False
            dropped = len(self._frames) >= self._capacity
            if dropped:
                self._frames.popleft()
                self._dropped += 1
            self._frames.append(frame)
            self._enqueued += 1
            self._condition.notify()
            return dropped

    def get(self, timeout: float | None = None, *, latest: bool = True) -> Frame | None:
        """Return a frame or ``None`` after timeout/closure.

        ``latest=True`` drains stale queued frames so slow inference catches up
        immediately instead of processing historical video.
        """

        deadline = None if timeout is None else time.monotonic() + max(timeout, 0)
        with self._condition:
            while not self._frames and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if not self._frames:
                return None
            if latest and len(self._frames) > 1:
                stale = len(self._frames) - 1
                for _ in range(stale):
                    self._frames.popleft()
                self._dropped += stale
            frame = self._frames.popleft()
            self._dequeued += 1
            return frame

    def clear(self) -> None:
        with self._condition:
            self._dropped += len(self._frames)
            self._frames.clear()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def metrics(self) -> FrameBufferMetrics:
        with self._condition:
            return FrameBufferMetrics(
                capacity=self._capacity,
                queue_size=len(self._frames),
                enqueued_frames=self._enqueued,
                dequeued_frames=self._dequeued,
                dropped_frames=self._dropped,
            )
