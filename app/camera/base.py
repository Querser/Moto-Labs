"""Camera contracts and timestamped frame types.

The capture timestamp belongs to a frame and must survive every downstream
transformation.  Consumers must never replace it with an inference-completion
timestamp.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any


class CameraError(RuntimeError):
    """Base error for camera operations."""


class CameraUnavailableError(CameraError):
    """The selected source cannot be opened."""


class CameraReadError(CameraError):
    """A source was opened but a frame could not be acquired."""


class EndOfStream(CameraError):
    """A finite source reached its normal end."""


class Rotation(int, Enum):
    NONE = 0
    CLOCKWISE_90 = 90
    UPSIDE_DOWN = 180
    COUNTERCLOCKWISE_90 = 270


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Cross-platform capture settings.

    Unsupported optional properties are reported in source diagnostics rather
    than making an otherwise usable camera fail to open.
    """

    width: int = 1280
    height: int = 720
    target_fps: float = 30.0
    backend: str | None = None
    exposure: float | None = None
    autofocus: bool | None = None
    rotation: Rotation = Rotation.NONE
    mirror_horizontal: bool = False
    mirror_vertical: bool = False
    queue_size: int = 2
    reconnect_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera resolution must be positive")
        if not 0 < self.target_fps <= 1000:
            raise ValueError("target_fps must be between 0 and 1000")
        if not 1 <= self.queue_size <= 1024:
            raise ValueError("queue_size must be between 1 and 1024")
        if self.reconnect_interval_s < 0:
            raise ValueError("reconnect_interval_s cannot be negative")
        try:
            rotation = Rotation(self.rotation)
        except ValueError as exc:
            raise ValueError("rotation must be one of 0, 90, 180, 270") from exc
        object.__setattr__(self, "rotation", rotation)


@dataclass(frozen=True, slots=True)
class Frame:
    """One image plus clocks sampled immediately after acquisition."""

    image: Any
    sequence: int
    source_id: str
    captured_monotonic_ns: int
    captured_at_utc: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("Frame sequence cannot be negative")
        if self.captured_monotonic_ns < 0:
            raise ValueError("Monotonic timestamp cannot be negative")
        timestamp = self.captured_at_utc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("captured_at_utc must be timezone-aware")
        object.__setattr__(self, "captured_at_utc", timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def captured(
        cls,
        image: Any,
        *,
        sequence: int,
        source_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Frame:
        # perf_counter_ns is intentionally sampled first and as close to the
        # acquisition call as Python permits.
        monotonic_ns = time.perf_counter_ns()
        wall_clock = datetime.now(timezone.utc)
        return cls(
            image=image,
            sequence=sequence,
            source_id=source_id,
            captured_monotonic_ns=monotonic_ns,
            captured_at_utc=wall_clock,
            metadata=metadata or {},
        )


class CameraSource(ABC):
    """Replaceable frame source used by the camera worker."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Stable identifier suitable for configuration and diagnostics."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """Whether the source currently owns an open capture resource."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the source or raise :class:`CameraUnavailableError`."""

    @abstractmethod
    def read(self) -> Frame:
        """Acquire and timestamp one frame.

        Finite sources raise :class:`EndOfStream` at normal completion.
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources. The operation must be idempotent."""

    def __enter__(self) -> CameraSource:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
