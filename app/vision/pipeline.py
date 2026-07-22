"""Modular motorcycle-to-lap-event computer-vision pipeline."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

from app.camera import Frame

from .frame_selector import measure_region_quality, select_diverse_evidence
from .interfaces import NumberRegionExtractor, ObjectDetector, ObjectTracker, OcrEngine
from .line_crossing import CrossingEvent, FinishLine, LineCrossingDetector
from .ocr import OcrAggregator
from .types import BoundingBox, CandidateRegion, OcrPrediction, OcrResolution, Track
from .verifier import NumberVerifier


@dataclass(frozen=True, slots=True)
class StablePassage:
    racing_number: str
    confidence: float
    track_id: int
    captured_monotonic_ns: int
    detected_at_utc: Any
    frame_sequence: int
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class VisionStageRegion:
    """One observable board or digit localization result in frame coordinates."""

    track_id: int
    bbox: BoundingBox
    kind: str
    confidence: float
    text: str | None = None


@dataclass(frozen=True, slots=True)
class VisionPipelineResult:
    recognized_number: str | None
    tracks: tuple[Track, ...]
    track_numbers: tuple[tuple[int, str, float], ...]
    passages: tuple[StablePassage, ...]
    crossings: tuple[CrossingEvent, ...]
    number_board_crop: Any | None = None
    board_regions: tuple[VisionStageRegion, ...] = ()
    digit_regions: tuple[VisionStageRegion, ...] = ()


@dataclass(slots=True)
class _ParticipantLock:
    missing_frames: int = 0
    clear_frames: int = 0


@dataclass(frozen=True, slots=True)
class _OcrEvidence:
    """One short-lived board candidate retained for crossing-time recovery."""

    frame: Frame
    track: Track
    region: CandidateRegion
    quality: float


@dataclass(frozen=True, slots=True)
class _RecoveredNumber:
    racing_number: str
    confidence: float
    support: int


class ParticipantPassageGuard:
    """Suppress duplicate track/crossing events until a physical stay has ended."""

    def __init__(self, *, clear_frames: int = 3, disappearance_frames: int = 12) -> None:
        self.clear_frames = clear_frames
        self.disappearance_frames = disappearance_frames
        self._locked: dict[str, _ParticipantLock] = {}

    def reset(self) -> None:
        self._locked.clear()

    def accept(self, number: str) -> bool:
        if number in self._locked:
            return False
        self._locked[number] = _ParticipantLock()
        return True

    def update(
        self,
        tracks: tuple[Track, ...],
        numbers_by_track: dict[int, tuple[str, float]],
        line_detector: LineCrossingDetector,
        frame_size: tuple[int, int],
    ) -> None:
        by_number: dict[str, list[Track]] = {}
        for track in tracks:
            value = numbers_by_track.get(track.track_id)
            if value is not None and track.observed:
                by_number.setdefault(value[0], []).append(track)
        for racing_number, state in tuple(self._locked.items()):
            observed = by_number.get(racing_number, [])
            if not observed:
                state.missing_frames += 1
                state.clear_frames = 0
                if state.missing_frames >= self.disappearance_frames:
                    del self._locked[racing_number]
                continue
            state.missing_frames = 0
            if all(
                line_detector.is_fully_clear(track.bbox, frame_size) for track in observed
            ):
                state.clear_frames += 1
                if state.clear_frames >= self.clear_frames:
                    del self._locked[racing_number]
            else:
                state.clear_frames = 0


class MotorcycleVisionPipeline:
    """Associate local detector, tracker, front-board OCR, and line geometry."""

    def __init__(
        self,
        *,
        detector: ObjectDetector,
        tracker: ObjectTracker,
        region_extractor: NumberRegionExtractor,
        ocr_engine: OcrEngine,
        ocr_aggregator: OcrAggregator,
        finish_line: FinishLine | None = None,
        line_detector: LineCrossingDetector | None = None,
        passage_guard: ParticipantPassageGuard | None = None,
        pending_crossing_ns: int = 4_000_000_000,
        recovery_delay_ns: int = 0,
        evidence_limit: int = 28,
        number_verifier: NumberVerifier | None = None,
        recover_identities_on_exit: bool = True,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.region_extractor = region_extractor
        self.ocr_engine = ocr_engine
        self.ocr_aggregator = ocr_aggregator
        self.line_detector = line_detector or LineCrossingDetector(finish_line)
        self.passage_guard = passage_guard or ParticipantPassageGuard()
        self.pending_crossing_ns = pending_crossing_ns
        self.recovery_delay_ns = max(0, recovery_delay_ns)
        self.evidence_limit = max(4, evidence_limit)
        self.number_verifier = number_verifier
        self.recover_identities_on_exit = recover_identities_on_exit
        self._numbers_by_track: dict[int, tuple[str, float]] = {}
        self._number_support_by_track: dict[int, int] = {}
        self._pending_crossings: dict[int, CrossingEvent] = {}
        self._recovery_attempts_by_track: dict[int, int] = {}
        self._track_missing: dict[int, int] = {}
        self._evidence_by_track: dict[int, list[_OcrEvidence]] = {}
        self._recovery_diagnostics: dict[int, tuple[dict[str, Any], ...]] = {}
        self._identity_recovered_tracks: set[int] = set()
        self._last_observed_tracks: tuple[Track, ...] = ()
        self._track_aliases: dict[int, int] = {}
        self._recent_track_snapshots: dict[int, Track] = {}

    @property
    def finish_line(self) -> FinishLine:
        return self.line_detector.line

    @property
    def recovery_diagnostics(self) -> dict[int, tuple[dict[str, Any], ...]]:
        """Return read-only-friendly candidate summaries for offline QA."""

        return {track_id: tuple(items) for track_id, items in self._recovery_diagnostics.items()}

    def set_finish_line(self, line: FinishLine) -> None:
        self.line_detector.set_line(line)
        self._pending_crossings.clear()
        self._recovery_attempts_by_track.clear()
        self.passage_guard.reset()

    def track_near_finish_line(
        self,
        frame_size: tuple[int, int],
        *,
        margin: float = 0.06,
    ) -> bool:
        """Return whether a recently observed motorcycle is entering the line band."""

        width, height = frame_size
        return any(
            _normalized_bbox_near_line(track.bbox, width, height, self.finish_line, margin)
            for track in self._last_observed_tracks
        )

    def reset(self) -> None:
        self.tracker.reset()
        self.ocr_aggregator.reset()
        self.line_detector.reset()
        self.passage_guard.reset()
        self._numbers_by_track.clear()
        self._number_support_by_track.clear()
        self._pending_crossings.clear()
        self._recovery_attempts_by_track.clear()
        self._track_missing.clear()
        self._evidence_by_track.clear()
        self._recovery_diagnostics.clear()
        self._identity_recovered_tracks.clear()
        self._last_observed_tracks = ()
        self._track_aliases.clear()
        self._recent_track_snapshots.clear()

    def close(self) -> None:
        self.detector.close()
        self.ocr_engine.close()
        if self.number_verifier is not None:
            self.number_verifier.close()

    def recover_pending_identities(self) -> tuple[tuple[int, str, float], ...]:
        """Resolve retained bursts without creating finish-line events.

        This diagnostic/finalization hook is useful at the end of an uploaded
        video evaluation window. It deliberately cannot create a lap: only a
        genuine line crossing in :meth:`process` may do that.
        """

        recovered_identities: list[tuple[int, str, float]] = []
        for track_id in tuple(self._evidence_by_track):
            recovered = self._recover_track_number(track_id)
            if recovered is None:
                continue
            recovered_identities.append(
                (track_id, recovered.racing_number, recovered.confidence)
            )
        return tuple(recovered_identities)

    def process(self, frame: Frame) -> VisionPipelineResult:
        detections = self.detector.detect(frame)
        tracks = self._stitch_tracklets(
            tuple(
                self.tracker.update(
                    detections,
                    captured_monotonic_ns=frame.captured_monotonic_ns,
                )
            )
        )
        self._last_observed_tracks = tuple(track for track in tracks if track.observed)
        frame_size = _frame_size(frame)
        crossings = self.line_detector.update(tracks, frame_size)
        recognized_number: str | None = None
        debug_crop: Any | None = None
        board_regions: list[VisionStageRegion] = []
        digit_regions: list[VisionStageRegion] = []
        observed_ids = {track.track_id for track in tracks if track.observed}
        recovered_on_exit = self._age_track_state(observed_ids)
        if recovered_on_exit:
            recognized_number = recovered_on_exit[-1][1]

        for track in tracks:
            if not track.observed:
                continue
            stable = self._numbers_by_track.get(track.track_id)
            # Identity is established from the approaching/front-facing part of
            # the trajectory. Never replace it with a larger rear panel that
            # becomes visible after the motorcycle has passed the camera.
            if stable is not None:
                stable_regions = sorted(
                    self.region_extractor.extract(frame, track),
                    key=lambda item: item.confidence,
                    reverse=True,
                )
                if stable_regions:
                    self._retain_evidence(frame, track, stable_regions)
                recognized_number = stable[0]
                continue
            if frame.metadata.get("deferred_ocr", False):
                # Uploaded races first establish geometry and retain sharp
                # per-track crops. Accurate OCR is then run only for a crossing
                # or track finalization, instead of blocking every video frame.
                deferred_regions = sorted(
                    self.region_extractor.extract(frame, track),
                    key=lambda item: item.confidence,
                    reverse=True,
                )
                if deferred_regions:
                    self._retain_evidence(frame, track, deferred_regions)
                    if debug_crop is None:
                        debug_crop = deferred_regions[0].image
                continue
            resolution, selected_crop, selected_region, predictions = self._recognize_track(
                frame, track
            )
            if selected_crop is not None and debug_crop is None:
                debug_crop = selected_crop
            # A candidate rectangle is only an internal proposal. Display it
            # after OCR has actually localized text, otherwise bright clothing
            # and background contours produce visibly jumping random boxes.
            if selected_region is not None and predictions:
                board_regions.append(
                    VisionStageRegion(
                        track_id=track.track_id,
                        bbox=selected_region.bbox,
                        kind="number_board",
                        confidence=selected_region.confidence,
                    )
                )
            for prediction in predictions:
                digit_region = _digit_region(track.track_id, selected_region, prediction)
                if digit_region is not None:
                    digit_regions.append(digit_region)
            if resolution is None or not resolution.is_resolved:
                continue
            if resolution.racing_number is None:
                continue
            stable = (resolution.racing_number, resolution.confidence)
            self._numbers_by_track[track.track_id] = stable
            self._number_support_by_track[track.track_id] = resolution.observation_count
            recognized_number = stable[0]

        self.passage_guard.update(
            tracks, self._numbers_by_track, self.line_detector, frame_size
        )
        for event in crossings:
            self._pending_crossings[event.track_id] = event
            self._recovery_attempts_by_track[event.track_id] = 0
        passages: list[StablePassage] = []
        for track_id, event in tuple(self._pending_crossings.items()):
            age_ns = frame.captured_monotonic_ns - event.captured_monotonic_ns
            if age_ns > self.pending_crossing_ns:
                del self._pending_crossings[track_id]
                self._recovery_attempts_by_track.pop(track_id, None)
                continue
            attempts = self._recovery_attempts_by_track.get(track_id, 0)
            # Keep collecting a short post-line burst. The event timestamp is
            # already authoritative, so this improves identity without adding
            # the OCR runtime to the measured lap time. A second bounded pass
            # is allowed when the first group was blurred or clipped.
            next_recovery_age_ns = self.recovery_delay_ns * (attempts + 1)
            if attempts < 2 and age_ns >= next_recovery_age_ns:
                recovered = self._recover_track_number(track_id)
                self._recovery_attempts_by_track[track_id] = attempts + 1
                if recovered is not None:
                    stable = self._numbers_by_track.get(track_id)
                    stable_support = self._number_support_by_track.get(track_id, 0)
                    if (
                        stable is None
                        or recovered.racing_number == stable[0]
                        or recovered.support > stable_support
                        or (
                            stable[0] in recovered.racing_number
                            and len(recovered.racing_number) > len(stable[0])
                            and recovered.support >= stable_support
                        )
                    ):
                        self._numbers_by_track[track_id] = (
                            recovered.racing_number,
                            recovered.confidence,
                        )
                        self._number_support_by_track[track_id] = recovered.support
                        recognized_number = recovered.racing_number
            # Do not publish a preliminary two-frame OCR reading before the
            # retained best-frame recovery had a chance to correct it.
            if self._recovery_attempts_by_track.get(track_id, 0) == 0:
                continue
            stable = self._numbers_by_track.get(track_id)
            if stable is None:
                continue
            del self._pending_crossings[track_id]
            self._recovery_attempts_by_track.pop(track_id, None)
            number, confidence = stable
            if not self.passage_guard.accept(number):
                continue
            identity = (
                f"{frame.source_id}|{event.captured_monotonic_ns}|{track_id}|{number}"
            )
            time_offset_ns = max(
                0, frame.captured_monotonic_ns - event.captured_monotonic_ns
            )
            passages.append(
                StablePassage(
                    racing_number=number,
                    confidence=confidence,
                    track_id=track_id,
                    captured_monotonic_ns=event.captured_monotonic_ns,
                    detected_at_utc=frame.captured_at_utc
                    - timedelta(microseconds=time_offset_ns / 1_000),
                    frame_sequence=frame.sequence,
                    idempotency_key=hashlib.sha256(identity.encode()).hexdigest()[:40],
                )
            )
        active_track_numbers = tuple(
            (track.track_id, value[0], value[1])
            for track in tracks
            if (value := self._numbers_by_track.get(track.track_id)) is not None
        )
        active_number_ids = {item[0] for item in active_track_numbers}
        track_numbers = active_track_numbers + tuple(
            item for item in recovered_on_exit if item[0] not in active_number_ids
        )
        return VisionPipelineResult(
            recognized_number,
            tracks,
            track_numbers,
            tuple(passages),
            crossings,
            debug_crop,
            tuple(board_regions),
            tuple(digit_regions),
        )

    def _stitch_tracklets(self, tracks: tuple[Track, ...]) -> tuple[Track, ...]:
        """Keep one logical ID across a short, spatially plausible tracker split.

        Tiny distant motorcycles are sometimes absent for a few detector frames.
        ByteTrack then creates a new raw ID even though the physical motorcycle
        continued smoothly. A conservative one-to-one handoff preserves the OCR
        burst; ambiguous close-together riders deliberately remain separate.
        """

        if not tracks:
            return ()
        now_ns = max(track.captured_monotonic_ns for track in tracks)
        current_raw_ids = {track.track_id for track in tracks}
        observed_canonical_ids = {
            self._track_aliases.get(track.track_id, track.track_id) for track in tracks
        }
        stitched: list[Track] = []
        claimed_canonical_ids: set[int] = set()
        for raw_track in sorted(tracks, key=lambda item: item.confidence, reverse=True):
            raw_id = raw_track.track_id
            canonical_id = self._track_aliases.get(raw_id)
            previous: Track | None = None
            if canonical_id is None:
                candidates: list[tuple[float, int, Track]] = []
                for candidate_id, candidate in self._recent_track_snapshots.items():
                    if (
                        candidate_id in observed_canonical_ids
                        or candidate_id in claimed_canonical_ids
                    ):
                        continue
                    if candidate_id in current_raw_ids:
                        continue
                    score = _tracklet_handoff_score(candidate, raw_track)
                    if score is not None:
                        candidates.append((score, candidate_id, candidate))
                candidates.sort(key=lambda item: item[0])
                unambiguous = (
                    candidates
                    and (
                        len(candidates) == 1
                        or candidates[1][0] - candidates[0][0] >= 0.16
                    )
                )
                if unambiguous:
                    _score, canonical_id, previous = candidates[0]
                else:
                    canonical_id = raw_id
                self._track_aliases[raw_id] = canonical_id
            elif canonical_id != raw_id:
                previous = self._recent_track_snapshots.get(canonical_id)
            claimed_canonical_ids.add(canonical_id)
            raw_metadata = {**raw_track.metadata, "raw_track_id": raw_id}
            if canonical_id == raw_id:
                logical = replace(raw_track, metadata=raw_metadata)
            else:
                logical = replace(
                    raw_track,
                    track_id=canonical_id,
                    hits=(previous.hits if previous is not None else 0) + raw_track.hits,
                    age=(previous.age if previous is not None else 0) + raw_track.age,
                    metadata=raw_metadata,
                )
            stitched.append(logical)
        by_canonical: dict[int, Track] = {}
        for track in stitched:
            existing = by_canonical.get(track.track_id)
            if existing is None or track.confidence > existing.confidence:
                by_canonical[track.track_id] = track
        for track in by_canonical.values():
            self._recent_track_snapshots[track.track_id] = track
        for track_id, snapshot in tuple(self._recent_track_snapshots.items()):
            if now_ns - snapshot.captured_monotonic_ns > 1_000_000_000:
                self._recent_track_snapshots.pop(track_id, None)
        return tuple(sorted(by_canonical.values(), key=lambda item: item.track_id))

    def collect_evidence(self, frame: Frame) -> None:
        """Retain an in-between source frame without another detector pass.

        Uploaded 30 FPS files run semantic detection at roughly 15 FPS. The
        intervening frame is still valuable OCR evidence, so the last tracked
        geometry is extrapolated for at most 120 ms and only cheap board crops
        are retained. This restores temporal detail without doubling YOLO work.
        """

        for track in self._last_observed_tracks:
            delta_ns = frame.captured_monotonic_ns - track.captured_monotonic_ns
            if not 0 < delta_ns <= 120_000_000:
                continue
            predicted = _extrapolate_track(track, frame.captured_monotonic_ns)
            regions = sorted(
                self.region_extractor.extract(frame, predicted),
                key=lambda item: item.confidence,
                reverse=True,
            )
            if regions:
                self._retain_evidence(frame, predicted, regions)

    def _recognize_track(
        self, frame: Frame, track: Track
    ) -> tuple[
        OcrResolution | None,
        Any | None,
        CandidateRegion | None,
        tuple[OcrPrediction, ...],
    ]:
        regions = sorted(
            self.region_extractor.extract(frame, track),
            key=lambda item: item.confidence,
            reverse=True,
        )
        if not regions:
            return None, None, None, ()
        self._retain_evidence(frame, track, regions)
        # Process one candidate per track/frame to keep live latency bounded.
        # Alternate the strongest localized board with every fallback in turn:
        # this catches a plate at the single sharp close-pass frame, while the
        # directional anchors still recover when the strongest bright contour
        # is a helmet or fairing.
        fallback = next(
            (item for item in regions if item.kind == "front_board_fallback"), None
        )
        anchors = [item for item in regions if "_anchor_" in item.kind]
        if fallback is not None and frame.sequence % 3 == 0:
            selected = fallback
        elif anchors and frame.sequence % 5 == 0:
            selected = anchors[(frame.sequence // 5) % len(anchors)]
        elif len(regions) == 1 or frame.sequence % 2:
            selected = regions[0]
        else:
            selected = regions[1 + (frame.sequence // 2) % (len(regions) - 1)]
        # The normal per-frame path uses recognition-only shortcuts. Expensive
        # text detection belongs to the bounded best-frame recovery near a real
        # crossing, rather than running on every visible motorcycle frame.
        fast_frame = Frame(
            image=frame.image,
            sequence=frame.sequence,
            source_id=frame.source_id,
            captured_monotonic_ns=frame.captured_monotonic_ns,
            captured_at_utc=frame.captured_at_utc,
            metadata={**frame.metadata, "fast_ocr_only": True},
        )
        predictions = list(
            self.ocr_engine.recognize(selected, frame=fast_frame, track=track)
        )
        if not predictions:
            return None, selected.image, selected, ()
        resolution = self.ocr_aggregator.observe_predictions(
            track.track_id,
            predictions,
            captured_monotonic_ns=frame.captured_monotonic_ns,
            frame_sequence=frame.sequence,
        )
        return resolution, selected.image, selected, tuple(predictions)

    def _retain_evidence(
        self,
        frame: Frame,
        track: Track,
        regions: list[CandidateRegion],
    ) -> None:
        """Keep a bounded set of sharp board crops without blocking capture."""

        scored = sorted(
            ((_candidate_quality(region), region) for region in regions),
            key=lambda item: item[0],
            reverse=True,
        )
        chosen: list[tuple[float, CandidateRegion]] = scored[:1]
        fallback = next(
            (item for item in scored if item[1].kind == "front_board_fallback"), None
        )
        anchors = [item for item in scored if "_anchor_" in item[1].kind]
        if fallback is not None and all(fallback[1] is not item[1] for item in chosen):
            chosen.append(fallback)
        if anchors:
            anchor = anchors[frame.sequence % len(anchors)]
            if all(anchor[1] is not item[1] for item in chosen):
                chosen.append(anchor)
        retained = self._evidence_by_track.setdefault(track.track_id, [])
        for quality, region in chosen:
            image = region.image
            if image is None or not hasattr(image, "shape") or getattr(image, "size", 0) == 0:
                continue
            copied = image.copy()
            retained.append(
                _OcrEvidence(
                    frame=Frame(
                        image=copied,
                        sequence=frame.sequence,
                        source_id=frame.source_id,
                        captured_monotonic_ns=frame.captured_monotonic_ns,
                        captured_at_utc=frame.captured_at_utc,
                        metadata=frame.metadata,
                    ),
                    track=track,
                    region=CandidateRegion(
                        image=copied,
                        bbox=region.bbox,
                        kind=region.kind,
                        confidence=region.confidence,
                        metadata=region.metadata,
                    ),
                    quality=quality,
                )
            )
        localized = _select_diverse_localized_evidence(retained, limit=8)
        fallbacks = sorted(
            (item for item in retained if item.region.kind == "front_board_fallback"),
            key=lambda item: item.quality,
            reverse=True,
        )[:6]
        anchored = sorted(
            (item for item in retained if "_anchor_" in item.region.kind),
            key=lambda item: item.quality,
            reverse=True,
        )[:6]
        # A logical motorcycle can span several raw ByteTrack IDs. Preserve a
        # couple of sharp crops from each recent fragment so the handoff does
        # not let early distant frames crowd out the close number-board view.
        by_raw_track: defaultdict[int, list[_OcrEvidence]] = defaultdict(list)
        for item in retained:
            by_raw_track[int(item.track.metadata.get("raw_track_id", item.track.track_id))].append(
                item
            )
        recent_segments = sorted(
            by_raw_track.values(),
            key=lambda values: max(item.frame.sequence for item in values),
            reverse=True,
        )[:4]
        segment_representatives = [
            item
            for values in recent_segments
            for item in sorted(values, key=lambda value: value.quality, reverse=True)[:2]
        ]
        retained[:] = _unique_evidence(
            segment_representatives + localized + fallbacks + anchored,
            limit=self.evidence_limit,
        )

    def _recover_track_number(self, track_id: int) -> _RecoveredNumber | None:
        """Resolve a fast pass from the best distinct frames captured earlier."""

        # Keep an unresolved burst for the second post-line attempt. Earlier
        # versions popped it before OCR, so a clipped pre-line ``43`` could
        # never be combined with the later ``35`` from the same motorcycle.
        evidence = list(self._evidence_by_track.get(track_id, []))
        if not evidence:
            return None
        # Preserve candidate diversity, but cap expensive OCR work. Several
        # regions from one source frame are combined into one temporal vote.
        localized = _select_diverse_localized_evidence(evidence, limit=4)
        fallbacks = sorted(
            (item for item in evidence if item.region.kind == "front_board_fallback"),
            key=lambda item: item.quality,
            reverse=True,
        )[:4]
        anchored_pool = sorted(
            (item for item in evidence if "_anchor_" in item.region.kind),
            key=lambda item: item.quality,
            reverse=True,
        )
        # Prefer one sharp observation from each normalized anchor instead of
        # spending every recovery slot on the same, occasionally wrong crop.
        anchored: list[_OcrEvidence] = []
        seen_anchor_kinds: set[str] = set()
        for item in anchored_pool:
            if item.region.kind in seen_anchor_kinds:
                continue
            anchored.append(item)
            seen_anchor_kinds.add(item.region.kind)
            if len(anchored) >= 3:
                break
        selected = localized + fallbacks + anchored
        by_raw_track: defaultdict[int, list[_OcrEvidence]] = defaultdict(list)
        for item in evidence:
            by_raw_track[int(item.track.metadata.get("raw_track_id", item.track.track_id))].append(
                item
            )
        recent_segments = sorted(
            by_raw_track.values(),
            key=lambda values: max(item.frame.sequence for item in values),
            reverse=True,
        )[:2]
        segment_representatives = [
            item
            for values in recent_segments
            for item in _select_segment_families(values)
        ]
        selected = _unique_evidence(
            segment_representatives + selected,
            limit=16,
        )
        # Most retained crops use the 256-pixel text detector with no repeated
        # preprocessing. One best crop per candidate family receives the
        # 480-pixel pass; only the overall best gets one alternate preprocessing
        # view. This preserves temporal/crop diversity without the previous
        # worst case of 12 crops x 4 high-resolution detector invocations.
        high_resolution_ids = {
            id(group[0])
            for group in (localized, fallbacks, anchored)
            if group
        }
        high_resolution_ids.update(id(item) for item in segment_representatives)
        v5_recovery_ids: set[int] = set()
        v5_frame_sequences: set[int] = set()
        direction_ready_fallbacks = sorted(
            (item for item in fallbacks if len(item.track.trajectory) >= 3),
            key=lambda item: (len(item.track.trajectory), item.track.hits, item.quality),
            reverse=True,
        )
        for item in direction_ready_fallbacks:
            if item.frame.sequence in v5_frame_sequences:
                continue
            v5_recovery_ids.add(id(item))
            v5_frame_sequences.add(item.frame.sequence)
            if len(v5_recovery_ids) >= 3:
                break
        # One localized crop from each of the last two raw tracker fragments
        # receives the alternate recognizer. This is the bounded path that can
        # pair a pre-handoff partial reading with a post-handoff partial.
        for item in segment_representatives:
            if (
                "_anchor_" in item.region.kind
                or item.region.kind == "front_board_fallback"
            ):
                continue
            v5_recovery_ids.add(id(item))
        # PP-OCRv6 receives only a small, temporally diverse subset. This keeps
        # uploaded-video throughput high while giving ambiguous boards the
        # strongest available local recognizer.
        v6_recovery_ids = {
            id(item)
            for item in select_diverse_evidence(
                selected,
                limit=min(6, len(selected)),
                minimum_frame_gap=1,
            )
        }
        predictions_by_frame: defaultdict[int, list[tuple[OcrPrediction, float]]] = (
            defaultdict(list)
        )
        candidate_images: dict[str, tuple[float, Any]] = {}
        for item in selected:
            try:
                is_high_resolution = id(item) in high_resolution_ids
                recovery_frame = Frame(
                    image=item.frame.image,
                    sequence=item.frame.sequence,
                    source_id=item.frame.source_id,
                    captured_monotonic_ns=item.frame.captured_monotonic_ns,
                    captured_at_utc=item.frame.captured_at_utc,
                    metadata={
                        **item.frame.metadata,
                        "force_full_ocr": True,
                        "high_resolution_ocr": is_high_resolution,
                        "preprocessing_variant_limit": 1 if is_high_resolution else 0,
                        "v5_recovery_ocr": id(item) in v5_recovery_ids,
                        "v6_recovery_ocr": id(item) in v6_recovery_ids,
                    },
                )
                predictions = self.ocr_engine.recognize(
                    item.region,
                    frame=recovery_frame,
                    track=item.track,
                )
            except Exception:
                continue
            for prediction in predictions:
                predictions_by_frame[item.frame.sequence].append(
                    (prediction, item.quality)
                )
                image_score = prediction.confidence * item.quality
                existing_image = candidate_images.get(prediction.text)
                if existing_image is None or image_score > existing_image[0]:
                    candidate_images[prediction.text] = (image_score, item.region.image)
        votes: defaultdict[str, list[tuple[float, float, int]]] = defaultdict(list)
        engines_by_text: defaultdict[str, set[str]] = defaultdict(set)
        whole_board_texts: set[str] = set()
        localized_texts: set[str] = set()
        for frame_sequence, frame_predictions in predictions_by_frame.items():
            strongest: dict[str, tuple[OcrPrediction, float]] = {}
            for prediction, quality in frame_predictions:
                engines_by_text[prediction.text].add(
                    str(prediction.metadata.get("engine", "unknown"))
                )
                if prediction.metadata.get("text_region_localized") is True:
                    localized_texts.add(prediction.text)
                elif prediction.metadata.get("engine") == "paddleocr_ppocrv6_medium_rec":
                    whole_board_texts.add(prediction.text)
                existing = strongest.get(prediction.text)
                if existing is None or prediction.confidence > existing[0].confidence:
                    strongest[prediction.text] = (prediction, quality)
            for text, (prediction, quality) in strongest.items():
                votes[text].append((prediction.confidence, quality, frame_sequence))
        if not votes:
            return None
        temporally_stitched_texts = _add_temporal_overlap_candidates(
            votes,
            engines_by_text,
            whole_board_texts,
            localized_texts,
        )
        ranked: list[tuple[str, float, int, float, float, int]] = []
        for text, values in votes.items():
            weighted = sum(
                confidence * (0.65 + 0.35 * quality)
                for confidence, quality, _ in values
            )
            if text in whole_board_texts:
                # The project localizer already supplies a board hypothesis.
                # A full-board PP-OCRv6 result gets a small prior over a nested
                # text-detector crop, whose first digit is easier to clip.
                weighted *= 1.04
            ranked.append(
                (
                    text,
                    weighted,
                    len(values),
                    max(confidence for confidence, _quality, _frame in values),
                    max(quality for _confidence, quality, _frame in values),
                    len(engines_by_text[text]),
                )
            )
        ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        self._recovery_diagnostics[track_id] = tuple(
            {
                "text": item[0],
                "score": round(item[1], 4),
                "frames": item[2],
                "maximum_confidence": round(item[3], 4),
                "maximum_quality": round(item[4], 4),
                "engines": item[5],
                "whole_board": item[0] in whole_board_texts,
                "text_localized": item[0] in localized_texts,
                "temporal_stitch": item[0] in temporally_stitched_texts,
            }
            for item in ranked
        )
        # OCR often loses the first or last digit at a crop edge. When a longer
        # high-confidence candidate contains the leading short candidate and
        # has comparable evidence, prefer the complete reading. No roster or
        # list of expected racing numbers is used here.
        leading = ranked[0]
        for candidate in ranked[1:]:
            short, long = leading[0], candidate[0]
            if len(long) <= len(short) or short not in long:
                continue
            comparable = candidate[1] >= leading[1] * 0.60
            trustworthy = candidate[3] >= 0.90 or candidate[2] >= leading[2]
            if comparable and trustworthy:
                ranked.remove(candidate)
                ranked.insert(0, candidate)
                break
        text, score, count, maximum_confidence, maximum_quality, _engine_count = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
        incompatible_runner_score = max(
            (
                candidate[1]
                for candidate in ranked[1:]
                if candidate[0] not in text and text not in candidate[0]
            ),
            default=0.0,
        )
        total_score = sum(item[1] for item in ranked)
        required_count = 3 if len(text) == 1 else 2
        truncated_confirmation = (
            len(text) > 1
            # Leading-zero boards are especially prone to losing their last
            # digit against the white panel. Keep the generic alternating
            # ``10``/``1`` conflict unresolved; only preserve a repeated,
            # complete leading-zero string with its shorter prefix here.
            and text.startswith("0")
            and count >= 2
            and maximum_confidence >= 0.90
            and any(
                text.startswith(candidate[0])
                and len(candidate[0]) < len(text)
                and score >= candidate[1] * 1.45
                for candidate in ranked[1:]
            )
        )
        has_truncation_conflict = any(
            candidate[0] != text
            and (
                candidate[0] in text
                or text in candidate[0]
            )
            for candidate in ranked[1:]
        )
        repeated = (
            count >= required_count
            and (len(text) > 1 or maximum_confidence >= 0.94)
            and (
                score / max(0.001, total_score) >= 0.70
                or (
                    score - runner_score >= 0.25
                    and not has_truncation_conflict
                    and maximum_confidence >= 0.978
                )
                or (
                    len(text) >= 3
                    and maximum_confidence >= 0.978
                    and score - incompatible_runner_score >= 0.50
                )
                or (
                    count >= 3
                    and maximum_confidence >= 0.94
                    and score - runner_score >= 0.60
                )
                or (
                    count >= 2
                    and maximum_confidence >= 0.95
                    and score - runner_score >= 0.50
                    and not has_truncation_conflict
                )
                or (
                    text in temporally_stitched_texts
                    and count >= 2
                    and maximum_confidence >= 0.85
                    and score - runner_score >= 0.20
                )
            )
        ) or truncated_confirmation
        pristine_single = maximum_confidence >= (
            0.995 if len(text) == 1 else 0.99
        ) and maximum_quality >= (0.75 if len(text) == 1 else 0.62)
        leading_zero_single = (
            len(text) >= 3
            and text.startswith("0")
            and count == 1
            and maximum_confidence >= 0.80
            and maximum_quality >= 0.55
            and incompatible_runner_score <= 0.25
        )
        # A complete three/four-digit OCR result may compete with its truncated
        # prefix from another preprocessing view of the same source frame. The
        # complete result needs exceptional confidence and a real score lead;
        # unrelated alternatives still prevent acceptance.
        complete_over_truncated_single = (
            len(text) >= 3
            and count == 1
            and maximum_confidence >= 0.978
            and maximum_quality >= 0.68
            and incompatible_runner_score <= 0.25
            and any(
                candidate[0] != text
                and candidate[0] in text
                and score >= candidate[1] * 1.05
                for candidate in ranked[1:]
            )
        )
        strong_single = (
            pristine_single
            or leading_zero_single
            or complete_over_truncated_single
        )
        comparison_score = incompatible_runner_score if strong_single else runner_score
        required_margin = 0.14 if strong_single else 0.18
        if not (repeated or strong_single) or score - comparison_score < required_margin:
            verifier = self.number_verifier
            repeated_candidates = [
                item
                for item in ranked[:4]
                if item[2] >= 2 and item[1] >= ranked[0][1] * 0.55
            ]
            verification_candidates = repeated_candidates
            if verifier is not None and len(verification_candidates) >= 2 and selected:
                allowed_candidates = tuple(
                    item[0] for item in verification_candidates[:4]
                )
                candidate_image = max(
                    (
                        candidate_images[item]
                        for item in allowed_candidates
                        if item in candidate_images
                    ),
                    key=lambda item: item[0],
                    default=None,
                )
                verification = (
                    verifier.verify(candidate_image[1], candidates=allowed_candidates)
                    if candidate_image is not None
                    else None
                )
                raw_text = str(getattr(verifier, "last_raw_text", ""))
                raw_texts = [raw_text] if raw_text else []
                if raw_texts:
                    self._recovery_diagnostics[track_id] += (
                        {
                            "verifier": "florence_2_base_ft",
                            "verifier_raw_texts": raw_texts,
                            "verifier_choice": (
                                verification.racing_number
                                if verification is not None
                                else None
                            ),
                        },
                    )
                if verification is not None:
                    verified = next(
                        (item for item in ranked if item[0] == verification.racing_number),
                        None,
                    )
                    if (
                        verified is not None
                        and verified[2] >= 2
                        and verified[1] >= ranked[0][1] * 0.80
                    ):
                        result = _RecoveredNumber(
                            verification.racing_number,
                            min(0.88, max(verification.confidence, verified[3] * 0.80)),
                            verified[2],
                        )
                        self._evidence_by_track.pop(track_id, None)
                        return result
            return None
        confidence = min(
            0.98,
            maximum_confidence if count == 1 else 0.70 + 0.08 * min(count, 3),
        )
        result = _RecoveredNumber(text, confidence, count)
        self._evidence_by_track.pop(track_id, None)
        return result

    def _age_track_state(
        self,
        observed_ids: set[int],
    ) -> tuple[tuple[int, str, float], ...]:
        """Finalize a short best-frame burst when a motorcycle leaves view.

        Uploaded test footage does not always contain a meaningful finish-line
        crossing. Previously those tracks lost all retained sharp crops after
        disappearing, so the interface could see the motorcycle but never its
        number. Exit recovery identifies the track for the live recognition
        list only; it cannot create a lap without a geometric line crossing.
        """

        recovered_identities: list[tuple[int, str, float]] = []
        known_ids = set(self._numbers_by_track) | set(self._evidence_by_track)
        for track_id in known_ids:
            if track_id in observed_ids:
                self._track_missing[track_id] = 0
                continue
            missing = self._track_missing.get(track_id, 0) + 1
            self._track_missing[track_id] = missing
            if missing >= 8 and track_id not in self._identity_recovered_tracks:
                self._identity_recovered_tracks.add(track_id)
                stable = self._numbers_by_track.get(track_id)
                evidence = self._evidence_by_track.get(track_id, [])
                enough_evidence = (
                    len(evidence) >= 4
                    and max((item.track.hits for item in evidence), default=0) >= 3
                )
                recovered = (
                    self._recover_track_number(track_id)
                    if (
                        stable is None
                        and enough_evidence
                        and self.recover_identities_on_exit
                    )
                    else None
                )
                if recovered is not None:
                    stable = (recovered.racing_number, recovered.confidence)
                    self._numbers_by_track[track_id] = stable
                    self._number_support_by_track[track_id] = recovered.support
                if stable is not None:
                    recovered_identities.append((track_id, stable[0], stable[1]))
            if missing >= 60:
                self._numbers_by_track.pop(track_id, None)
                self._number_support_by_track.pop(track_id, None)
                self._track_missing.pop(track_id, None)
                self._evidence_by_track.pop(track_id, None)
                self._identity_recovered_tracks.discard(track_id)
                self.ocr_aggregator.reset(track_id)
        return tuple(recovered_identities)


def _add_temporal_overlap_candidates(
    votes: dict[str, list[tuple[float, float, int]]],
    engines_by_text: dict[str, set[str]],
    whole_board_texts: set[str],
    localized_texts: set[str],
) -> set[str]:
    """Reconstruct a number clipped differently in nearby source frames.

    For example, a left-clipped/right-clipped passage can yield ``43`` and
    ``35``. Their ordered overlap supplies independent evidence for ``435``.
    No roster or expected-number list participates in this operation.
    """

    stitched: set[str] = set()
    originals = list(votes.items())
    for left_text, left_values in originals:
        if len(left_text) < 2:
            continue
        for right_text, right_values in originals:
            if left_text == right_text or len(right_text) < 2:
                continue
            overlap = next(
                (
                    size
                    for size in range(min(len(left_text), len(right_text)), 0, -1)
                    if left_text[-size:] == right_text[:size]
                ),
                0,
            )
            if overlap == 0:
                continue
            merged = left_text + right_text[overlap:]
            if not 3 <= len(merged) <= 4 or merged in {left_text, right_text}:
                continue
            compatible_pairs = [
                (left, right)
                for left in left_values
                for right in right_values
                if 0 < right[2] - left[2] <= 45
                and left[0] >= 0.68
                and right[0] >= 0.68
            ]
            if not compatible_pairs:
                continue
            left, right = max(
                compatible_pairs,
                key=lambda pair: pair[0][0] * pair[0][1] + pair[1][0] * pair[1][1],
            )
            combined = {value[2]: value for value in votes.get(merged, [])}
            combined[left[2]] = left
            combined[right[2]] = right
            votes[merged] = list(combined.values())
            engines_by_text.setdefault(merged, set()).update(
                engines_by_text.get(left_text, set())
                | engines_by_text.get(right_text, set())
            )
            if left_text in whole_board_texts or right_text in whole_board_texts:
                whole_board_texts.add(merged)
            if left_text in localized_texts or right_text in localized_texts:
                localized_texts.add(merged)
            stitched.add(merged)
    return stitched


def _candidate_quality(region: CandidateRegion) -> float:
    """Rank crop sharpness, useful pixel size, and localization confidence."""

    return measure_region_quality(region).score


def _tracklet_handoff_score(previous: Track, current: Track) -> float | None:
    """Score a conservative short-gap handoff; lower values are better."""

    gap_ns = current.captured_monotonic_ns - previous.captured_monotonic_ns
    if not 0 < gap_ns <= 350_000_000:
        return None
    vertical_overlap = max(
        0.0,
        min(previous.bbox.y2, current.bbox.y2)
        - max(previous.bbox.y1, current.bbox.y1),
    )
    overlap_ratio = vertical_overlap / max(
        1.0,
        min(previous.bbox.height, current.bbox.height),
    )
    if overlap_ratio < 0.45:
        return None
    area_ratio = current.bbox.area / max(1.0, previous.bbox.area)
    if not 0.22 <= area_ratio <= 4.5:
        return None
    center_distance = previous.bbox.centroid_distance(current.bbox)
    scale = max(
        1.0,
        math.hypot(previous.bbox.width, previous.bbox.height),
        math.hypot(current.bbox.width, current.bbox.height),
    )
    normalized_distance = center_distance / scale
    if normalized_distance > 0.82:
        return None
    return (
        normalized_distance
        + (gap_ns / 350_000_000) * 0.20
        + abs(math.log(area_ratio)) * 0.12
        + (1.0 - overlap_ratio) * 0.18
    )


def _select_diverse_localized_evidence(
    evidence: list[_OcrEvidence],
    *,
    limit: int,
) -> list[_OcrEvidence]:
    """Keep both colour-mask and edge-rectangle board hypotheses.

    The sharpest bright component is often a white helmet or fairing, while a
    slightly lower-ranked edge rectangle contains the complete number board.
    Reserving one slot per localization family prevents several near-identical
    crops from crowding the useful alternative out of the bounded OCR burst.
    """

    localized = [
        item
        for item in evidence
        if "_anchor_" not in item.region.kind
        and item.region.kind != "front_board_fallback"
    ]
    ranked = select_diverse_evidence(
        localized,
        limit=max(limit * 2, limit),
        minimum_frame_gap=1,
    )
    selected: list[_OcrEvidence] = []
    selected_ids: set[int] = set()
    for prefix in ("front_board_bright_", "front_board_rect_"):
        candidate = next(
            (item for item in ranked if item.region.kind.startswith(prefix)),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(id(candidate))
    for item in ranked:
        if id(item) in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(id(item))
        if len(selected) >= limit:
            break
    return selected[:limit]


def _unique_evidence(
    evidence: list[_OcrEvidence],
    *,
    limit: int,
) -> list[_OcrEvidence]:
    """Keep insertion order while removing the same retained crop object."""

    result: list[_OcrEvidence] = []
    seen: set[int] = set()
    for item in evidence:
        marker = id(item)
        if marker in seen:
            continue
        result.append(item)
        seen.add(marker)
        if len(result) >= limit:
            break
    return result


def _select_segment_families(evidence: list[_OcrEvidence]) -> list[_OcrEvidence]:
    """Keep localized, contextual, and anchored views of one raw tracklet."""

    localized = max(
        (
            item
            for item in evidence
            if "_anchor_" not in item.region.kind
            and item.region.kind != "front_board_fallback"
        ),
        key=lambda item: item.quality,
        default=None,
    )
    fallback = max(
        (item for item in evidence if item.region.kind == "front_board_fallback"),
        key=lambda item: item.quality,
        default=None,
    )
    anchor = max(
        (item for item in evidence if "_anchor_" in item.region.kind),
        key=lambda item: item.quality,
        default=None,
    )
    return [item for item in (localized, fallback, anchor) if item is not None]


def _extrapolate_track(track: Track, captured_monotonic_ns: int) -> Track:
    """Move the latest bbox by its recent centroid velocity for one skipped frame."""

    dx = 0.0
    dy = 0.0
    if len(track.trajectory) >= 2:
        previous, latest = track.trajectory[-2:]
        trajectory_delta_ns = latest.captured_monotonic_ns - previous.captured_monotonic_ns
        if trajectory_delta_ns > 0:
            ahead_ns = max(0, captured_monotonic_ns - latest.captured_monotonic_ns)
            ratio = min(1.5, ahead_ns / trajectory_delta_ns)
            dx = (latest.x - previous.x) * ratio
            dy = (latest.y - previous.y) * ratio
    return Track(
        track_id=track.track_id,
        bbox=track.bbox.translated(dx, dy),
        confidence=track.confidence,
        hits=track.hits,
        age=track.age,
        missed_frames=track.missed_frames,
        observed=True,
        captured_monotonic_ns=captured_monotonic_ns,
        metadata=track.metadata,
        trajectory=track.trajectory,
    )


def _frame_size(frame: Frame) -> tuple[int, int]:
    image = frame.image
    if hasattr(image, "shape") and len(image.shape) >= 2:
        return int(image.shape[1]), int(image.shape[0])
    configured = frame.metadata.get("frame_size")
    if configured:
        return int(configured[0]), int(configured[1])
    return 1280, 720


def _normalized_bbox_near_line(
    bbox: BoundingBox,
    width: int,
    height: int,
    line: FinishLine,
    margin: float,
) -> bool:
    """Test a normalized bbox against a finite-width band around the line."""

    center_x = (bbox.x1 + bbox.x2) / (2 * max(1, width))
    center_y = (bbox.y1 + bbox.y2) / (2 * max(1, height))
    half_diagonal = math.hypot(
        bbox.width / (2 * max(1, width)),
        bbox.height / (2 * max(1, height)),
    )
    line_dx = line.x2 - line.x1
    line_dy = line.y2 - line.y1
    line_length = max(1e-9, math.hypot(line_dx, line_dy))
    distance = abs(
        line_dy * center_x
        - line_dx * center_y
        + line.x2 * line.y1
        - line.y2 * line.x1
    ) / line_length
    projection = (
        (center_x - line.x1) * line_dx + (center_y - line.y1) * line_dy
    ) / (line_length * line_length)
    return -0.15 <= projection <= 1.15 and distance <= half_diagonal + margin


def _digit_region(
    track_id: int,
    board: CandidateRegion | None,
    prediction: OcrPrediction,
) -> VisionStageRegion | None:
    if board is None:
        return None
    raw = prediction.metadata.get("digit_bbox")
    if not isinstance(raw, (tuple, list)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
        bbox = BoundingBox(
            board.bbox.x1 + board.bbox.width * x1,
            board.bbox.y1 + board.bbox.height * y1,
            board.bbox.x1 + board.bbox.width * x2,
            board.bbox.y1 + board.bbox.height * y2,
        )
    except (TypeError, ValueError):
        return None
    return VisionStageRegion(
        track_id=track_id,
        bbox=bbox,
        kind="digit_region",
        confidence=prediction.confidence,
        text=prediction.text,
    )


__all__ = [
    "MotorcycleVisionPipeline",
    "ParticipantPassageGuard",
    "StablePassage",
    "VisionPipelineResult",
    "VisionStageRegion",
]
