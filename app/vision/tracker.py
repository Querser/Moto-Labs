"""Multi-object tracking adapters with trajectory preservation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .interfaces import ObjectTracker
from .types import BoundingBox, Detection, Track, TrajectoryPoint


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: BoundingBox
    confidence: float
    metadata: dict[str, object]
    hits: int = 1
    age: int = 1
    missed_frames: int = 0
    velocity_x: float = 0
    velocity_y: float = 0
    trajectory: deque[TrajectoryPoint] = field(default_factory=deque)


class CentroidIoUTracker(ObjectTracker):
    """Greedy IoU/centroid tracker suitable for a static single camera.

    Constant-velocity prediction keeps tracks alive across brief occlusions.
    Predicted-only tracks are exposed with ``observed=False`` and therefore do
    not trigger line crossings by themselves.
    """

    def __init__(
        self,
        *,
        minimum_iou: float = 0.02,
        maximum_centroid_distance: float = 180.0,
        iou_weight: float = 0.6,
        max_missed_frames: int = 8,
        trajectory_size: int = 32,
        velocity_smoothing: float = 0.65,
    ) -> None:
        if not 0 <= minimum_iou <= 1:
            raise ValueError("minimum_iou must be between 0 and 1")
        if maximum_centroid_distance <= 0:
            raise ValueError("maximum_centroid_distance must be positive")
        if not 0 <= iou_weight <= 1:
            raise ValueError("iou_weight must be between 0 and 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        if trajectory_size < 2:
            raise ValueError("trajectory_size must be at least two")
        if not 0 <= velocity_smoothing < 1:
            raise ValueError("velocity_smoothing must be in [0, 1)")
        self.minimum_iou = minimum_iou
        self.maximum_centroid_distance = maximum_centroid_distance
        self.iou_weight = iou_weight
        self.max_missed_frames = max_missed_frames
        self.trajectory_size = trajectory_size
        self.velocity_smoothing = velocity_smoothing
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Detection], *, captured_monotonic_ns: int) -> list[Track]:
        states = list(self._tracks.values())
        candidate_pairs: list[tuple[float, int, int]] = []
        for state_index, state in enumerate(states):
            for detection_index, detection in enumerate(detections):
                iou = state.bbox.iou(detection.bbox)
                distance = state.bbox.centroid_distance(detection.bbox)
                if iou < self.minimum_iou and distance > self.maximum_centroid_distance:
                    continue
                proximity = max(0.0, 1 - distance / self.maximum_centroid_distance)
                score = self.iou_weight * iou + (1 - self.iou_weight) * proximity
                candidate_pairs.append((score, state_index, detection_index))

        assigned_states: set[int] = set()
        assigned_detections: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, state_index, detection_index in sorted(candidate_pairs, reverse=True):
            if state_index in assigned_states or detection_index in assigned_detections:
                continue
            assigned_states.add(state_index)
            assigned_detections.add(detection_index)
            matches.append((state_index, detection_index))

        observed_ids: set[int] = set()
        for state_index, detection_index in matches:
            state, detection = states[state_index], detections[detection_index]
            old_center = state.bbox.center
            new_center = detection.bbox.center
            instantaneous_x = new_center[0] - old_center[0]
            instantaneous_y = new_center[1] - old_center[1]
            smoothing = self.velocity_smoothing
            state.velocity_x = smoothing * state.velocity_x + (1 - smoothing) * instantaneous_x
            state.velocity_y = smoothing * state.velocity_y + (1 - smoothing) * instantaneous_y
            state.bbox = detection.bbox
            state.confidence = detection.confidence
            state.metadata = dict(detection.metadata)
            state.hits += 1
            state.age += 1
            state.missed_frames = 0
            center = detection.bbox.center
            state.trajectory.append(TrajectoryPoint(center[0], center[1], captured_monotonic_ns))
            observed_ids.add(state.track_id)

        for state_index, state in enumerate(states):
            if state_index in assigned_states:
                continue
            state.age += 1
            state.missed_frames += 1
            state.bbox = state.bbox.translated(state.velocity_x, state.velocity_y)
            state.velocity_x *= 0.85
            state.velocity_y *= 0.85

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            state = _TrackState(
                track_id=self._next_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                metadata=dict(detection.metadata),
                trajectory=deque(maxlen=self.trajectory_size),
            )
            center = detection.bbox.center
            state.trajectory.append(TrajectoryPoint(center[0], center[1], captured_monotonic_ns))
            self._tracks[state.track_id] = state
            observed_ids.add(state.track_id)
            self._next_id += 1

        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if state.missed_frames > self.max_missed_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

        return [
            Track(
                track_id=state.track_id,
                bbox=state.bbox,
                confidence=state.confidence,
                hits=state.hits,
                age=state.age,
                missed_frames=state.missed_frames,
                observed=state.track_id in observed_ids,
                captured_monotonic_ns=captured_monotonic_ns,
                metadata=state.metadata,
                trajectory=tuple(state.trajectory),
            )
            for state in sorted(self._tracks.values(), key=lambda item: item.track_id)
        ]


BaselineObjectTracker = CentroidIoUTracker


class SupervisionByteTrack(ObjectTracker):
    """Adapter for Supervision's MIT-licensed ByteTrack implementation.

    ByteTrack associates both high- and low-confidence detections and keeps a
    Kalman-predicted lost-track buffer. This adapter exposes project-native
    :class:`Track` values and capture-timestamped trajectories.
    """

    def __init__(
        self,
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 12,
        minimum_matching_threshold: float = 0.75,
        frame_rate: float = 30.0,
        trajectory_size: int = 40,
    ) -> None:
        try:
            import numpy as np
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError("ByteTrack requires supervision and NumPy") from exc
        self._np = np
        self._sv: Any = sv
        self._settings: dict[str, Any] = {
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
            "frame_rate": frame_rate,
            "minimum_consecutive_frames": 1,
        }
        self.trajectory_size = trajectory_size
        self._tracker: Any = self._new_tracker()
        self._trajectories: dict[int, deque[TrajectoryPoint]] = {}
        self._hits: dict[int, int] = {}
        self._ages: dict[int, int] = {}
        self._missed: dict[int, int] = {}

    def _new_tracker(self) -> Any:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return self._sv.ByteTrack(**self._settings)

    def reset(self) -> None:
        reset = getattr(self._tracker, "reset", None)
        if callable(reset):
            reset()
        else:
            self._tracker = self._new_tracker()
        self._trajectories.clear()
        self._hits.clear()
        self._ages.clear()
        self._missed.clear()

    def update(self, detections: Sequence[Detection], *, captured_monotonic_ns: int) -> list[Track]:
        for track_id in tuple(self._ages):
            self._ages[track_id] += 1
            self._missed[track_id] = self._missed.get(track_id, 0) + 1
        if detections:
            xyxy = self._np.asarray(
                [
                    [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2]
                    for item in detections
                ],
                dtype=self._np.float32,
            )
            confidence = self._np.asarray(
                [item.confidence for item in detections], dtype=self._np.float32
            )
            class_ids = self._np.full(len(detections), 3, dtype=int)
            source_indices = self._np.arange(len(detections), dtype=int)
            values = self._sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_ids,
                data={"source_index": source_indices},
            )
        else:
            values = self._sv.Detections.empty()
            values.confidence = self._np.asarray([], dtype=self._np.float32)
        tracked = self._tracker.update_with_detections(values)
        output: list[Track] = []
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else ()
        source_indices = tracked.data.get("source_index", ())
        confidences = tracked.confidence if tracked.confidence is not None else ()
        for index, raw_track_id in enumerate(tracker_ids):
            track_id = int(raw_track_id)
            box = tracked.xyxy[index]
            bbox = BoundingBox(*(float(value) for value in box))
            source_index = int(source_indices[index]) if len(source_indices) else 0
            metadata = (
                dict(detections[source_index].metadata)
                if 0 <= source_index < len(detections)
                else {}
            )
            trajectory = self._trajectories.setdefault(
                track_id, deque(maxlen=self.trajectory_size)
            )
            center = bbox.center
            trajectory.append(
                TrajectoryPoint(center[0], center[1], captured_monotonic_ns)
            )
            self._hits[track_id] = self._hits.get(track_id, 0) + 1
            self._ages.setdefault(track_id, 1)
            self._missed[track_id] = 0
            output.append(
                Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=float(confidences[index]),
                    hits=self._hits[track_id],
                    age=self._ages[track_id],
                    missed_frames=0,
                    observed=True,
                    captured_monotonic_ns=captured_monotonic_ns,
                    metadata=metadata,
                    trajectory=tuple(trajectory),
                )
            )
        expired = [track_id for track_id, missed in self._missed.items() if missed > 60]
        for track_id in expired:
            self._trajectories.pop(track_id, None)
            self._hits.pop(track_id, None)
            self._ages.pop(track_id, None)
            self._missed.pop(track_id, None)
        return sorted(output, key=lambda item: item.track_id)


__all__ = ["CentroidIoUTracker", "SupervisionByteTrack"]
