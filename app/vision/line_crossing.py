"""Geometry and state for authoritative motorcycle finish-line crossings."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .types import BoundingBox, Track


@dataclass(frozen=True, slots=True)
class FinishLine:
    """A two-point finish line stored independently of preview resolution."""

    x1: float = 0.10
    y1: float = 0.68
    x2: float = 0.90
    y2: float = 0.68

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Finish-line coordinates must be normalized to [0, 1]")
        if hypot(self.x2 - self.x1, self.y2 - self.y1) < 0.05:
            raise ValueError("Finish line is too short")

    def pixels(self, frame_size: tuple[int, int]) -> tuple[tuple[float, float], ...]:
        width, height = frame_size
        return (
            (self.x1 * width, self.y1 * height),
            (self.x2 * width, self.y2 * height),
        )

    def as_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    track_id: int
    captured_monotonic_ns: int
    point: tuple[float, float]
    interpolation: float


@dataclass(slots=True)
class _TrackLineState:
    anchor_point: tuple[float, float] | None = None
    anchor_side: float | None = None
    anchor_timestamp_ns: int | None = None
    locked: bool = False
    clear_frames: int = 0
    missing_frames: int = 0


class LineCrossingDetector:
    """Detect genuine side changes of a tracked motorcycle leading edge.

    The line is treated as a finite segment. A hysteresis band rejects bbox
    noise, while interpolation preserves the capture-time estimate even when
    a fast motorcycle moves across the line between two frames.
    """

    def __init__(
        self,
        line: FinishLine | None = None,
        *,
        hysteresis_px: float = 4.0,
        exit_distance_px: float = 18.0,
        exit_frames: int = 3,
        disappearance_frames: int = 12,
        segment_margin: float = 0.05,
    ) -> None:
        if hysteresis_px <= 0 or exit_distance_px <= hysteresis_px:
            raise ValueError("Line hysteresis/exit distances are invalid")
        if exit_frames < 1 or disappearance_frames < 1:
            raise ValueError("Line state frame thresholds must be positive")
        self.line = line or FinishLine()
        self.hysteresis_px = hysteresis_px
        self.exit_distance_px = exit_distance_px
        self.exit_frames = exit_frames
        self.disappearance_frames = disappearance_frames
        self.segment_margin = segment_margin
        self._states: dict[int, _TrackLineState] = {}

    def set_line(self, line: FinishLine) -> None:
        self.line = line
        self.reset()

    def reset(self) -> None:
        self._states.clear()

    def update(
        self,
        tracks: tuple[Track, ...] | list[Track],
        frame_size: tuple[int, int],
    ) -> tuple[CrossingEvent, ...]:
        observed_ids = {track.track_id for track in tracks if track.observed}
        events: list[CrossingEvent] = []
        for track_id, state in tuple(self._states.items()):
            if track_id in observed_ids:
                state.missing_frames = 0
                continue
            state.missing_frames += 1
            if state.missing_frames >= self.disappearance_frames:
                del self._states[track_id]

        for track in tracks:
            if not track.observed:
                continue
            state = self._states.setdefault(track.track_id, _TrackLineState())
            point = leading_point(track)
            side = self.signed_distance(point, frame_size)
            if state.locked:
                if self.is_fully_clear(track.bbox, frame_size):
                    state.clear_frames += 1
                    if state.clear_frames >= self.exit_frames:
                        state.locked = False
                        state.clear_frames = 0
                        state.anchor_point = point
                        state.anchor_side = side if abs(side) >= self.hysteresis_px else None
                        state.anchor_timestamp_ns = track.captured_monotonic_ns
                else:
                    state.clear_frames = 0
                continue

            if abs(side) < self.hysteresis_px:
                continue
            if state.anchor_side is None or state.anchor_point is None:
                state.anchor_point = point
                state.anchor_side = side
                state.anchor_timestamp_ns = track.captured_monotonic_ns
                continue
            if state.anchor_side * side < 0:
                event = self._interpolate_crossing(
                    track.track_id,
                    state.anchor_point,
                    point,
                    state.anchor_side,
                    side,
                    state.anchor_timestamp_ns or track.captured_monotonic_ns,
                    track.captured_monotonic_ns,
                    frame_size,
                )
                if event is not None:
                    events.append(event)
                    state.locked = True
                    state.clear_frames = 0
                    continue
            state.anchor_point = point
            state.anchor_side = side
            state.anchor_timestamp_ns = track.captured_monotonic_ns
        return tuple(events)

    def signed_distance(
        self, point: tuple[float, float], frame_size: tuple[int, int]
    ) -> float:
        start, end = self.line.pixels(frame_size)
        dx, dy = end[0] - start[0], end[1] - start[1]
        return (dx * (point[1] - start[1]) - dy * (point[0] - start[0])) / hypot(dx, dy)

    def is_fully_clear(self, bbox: BoundingBox, frame_size: tuple[int, int]) -> bool:
        corners = (
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y1),
            (bbox.x2, bbox.y2),
            (bbox.x1, bbox.y2),
        )
        distances = [self.signed_distance(point, frame_size) for point in corners]
        return min(distances) > self.exit_distance_px or max(distances) < -self.exit_distance_px

    def _interpolate_crossing(
        self,
        track_id: int,
        previous_point: tuple[float, float],
        current_point: tuple[float, float],
        previous_side: float,
        current_side: float,
        previous_timestamp_ns: int,
        current_timestamp_ns: int,
        frame_size: tuple[int, int],
    ) -> CrossingEvent | None:
        denominator = abs(previous_side) + abs(current_side)
        interpolation = 0.5 if denominator <= 0 else abs(previous_side) / denominator
        crossing = (
            previous_point[0] + (current_point[0] - previous_point[0]) * interpolation,
            previous_point[1] + (current_point[1] - previous_point[1]) * interpolation,
        )
        start, end = self.line.pixels(frame_size)
        line_dx, line_dy = end[0] - start[0], end[1] - start[1]
        along = (
            (crossing[0] - start[0]) * line_dx + (crossing[1] - start[1]) * line_dy
        ) / (line_dx * line_dx + line_dy * line_dy)
        if not -self.segment_margin <= along <= 1 + self.segment_margin:
            return None
        timestamp = round(
            previous_timestamp_ns
            + (current_timestamp_ns - previous_timestamp_ns) * interpolation
        )
        return CrossingEvent(track_id, timestamp, crossing, interpolation)


def leading_point(track: Track) -> tuple[float, float]:
    """Return the bbox edge that is foremost along the recent trajectory."""

    center_x, center_y = track.bbox.center
    if len(track.trajectory) < 2:
        return track.bbox.bottom_center
    previous, current = track.trajectory[-2], track.trajectory[-1]
    velocity_x, velocity_y = current.x - previous.x, current.y - previous.y
    if abs(velocity_x) >= abs(velocity_y):
        return (track.bbox.x2 if velocity_x >= 0 else track.bbox.x1, center_y)
    return (center_x, track.bbox.y2 if velocity_y >= 0 else track.bbox.y1)


__all__ = ["CrossingEvent", "FinishLine", "LineCrossingDetector", "leading_point"]
