"""Dependency-free data types shared by detector, tracker, OCR, and pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space axis-aligned bounding box using exclusive max edges."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(float("-inf") < value < float("inf") for value in values):
            raise ValueError("Bounding box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)

    def centroid_distance(self, other: BoundingBox) -> float:
        first, second = self.center, other.center
        return hypot(first[0] - second[0], first[1] - second[1])

    def iou(self, other: BoundingBox) -> float:
        intersection_width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        intersection_height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = intersection_width * intersection_height
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def translated(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def clipped(self, width: float, height: float) -> BoundingBox:
        x1 = min(max(self.x1, 0.0), width)
        y1 = min(max(self.y1, 0.0), height)
        x2 = min(max(self.x2, 0.0), width)
        y2 = min(max(self.y2, 0.0), height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box lies outside the frame")
        return BoundingBox(x1, y1, x2, y2)


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    label: str = "motorcycle_candidate"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        if not self.label:
            raise ValueError("Detection label cannot be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    x: float
    y: float
    captured_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    bbox: BoundingBox
    confidence: float
    hits: int
    age: int
    missed_frames: int
    observed: bool
    captured_monotonic_ns: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trajectory: tuple[TrajectoryPoint, ...] = ()

    def __post_init__(self) -> None:
        if self.track_id <= 0:
            raise ValueError("track_id must be positive")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class CandidateRegion:
    image: Any
    bbox: BoundingBox
    kind: str = "number_candidate"
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Candidate-region confidence must be between 0 and 1")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class OcrPrediction:
    text: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("OCR confidence must be between 0 and 1")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class OcrResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN = "unknown"
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING = "conflicting"
    INVALID = "invalid"
    NOT_IN_WHITELIST = "not_in_whitelist"


@dataclass(frozen=True, slots=True)
class OcrResolution:
    status: OcrResolutionStatus
    racing_number: str | None
    confidence: float
    observation_count: int
    alternatives: tuple[tuple[str, float], ...] = ()
    nearby_whitelist_numbers: tuple[str, ...] = ()

    @property
    def is_resolved(self) -> bool:
        return self.status is OcrResolutionStatus.RESOLVED
