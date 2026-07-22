"""Background bridge from motorcycle line crossings to lap persistence."""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import Condition, Event, RLock, Thread
from typing import Any

from app.camera import Frame
from app.database import SessionLocal
from app.domain import LapTimingService, PassageCandidate
from app.models import Race
from app.services.camera_runtime import camera_runtime
from app.services.live_events import live_event_hub
from app.vision import (
    BoundingBoxNumberRegionExtractor,
    CentroidIoUTracker,
    FinishLine,
    FrontNumberBoardRegionExtractor,
    HybridDigitOcrEngine,
    MetadataObjectDetector,
    MetadataOcrEngine,
    MotionAwareMotorcycleDetector,
    MotorcycleVisionPipeline,
    NumberRegionExtractor,
    ObjectDetector,
    ObjectTracker,
    OcrAggregationConfig,
    OcrAggregator,
    OcrEngine,
    OpenCVDigitOcrEngine,
    PaddleOcrV6DigitEngine,
    RapidOcrDigitEngine,
    StablePassage,
    SupervisionByteTrack,
    YoloXMotorcycleDetector,
)

logger = logging.getLogger(__name__)


class VisionRuntime:
    def __init__(self) -> None:
        self._lock = RLock()
        self._pipeline: MotorcycleVisionPipeline | None = None
        self._physical_ocr_engine: OcrEngine | None = None
        self._physical_motorcycle_detector: ObjectDetector | None = None
        self._race_id: int | None = None
        self._enabled = False
        self._recognized_number: str | None = None
        self._last_error: str | None = None
        self._last_processing_latency_ms = 0.0
        self._dropped_inference_frames = 0
        self._track_overlays: list[dict[str, Any]] = []
        self._board_overlays: list[dict[str, Any]] = []
        self._digit_overlays: list[dict[str, Any]] = []
        self._board_overlay_cache: dict[int, tuple[int, dict[str, Any]]] = {}
        self._digit_overlay_cache: dict[int, tuple[int, dict[str, Any]]] = {}
        self._track_overlay_cache: dict[int, tuple[int, dict[str, Any]]] = {}
        self._recognition_history: deque[dict[str, Any]] = deque(maxlen=20)
        self._stable_by_track: dict[int, str] = {}
        self._last_recognition_by_number: dict[str, int] = {}
        self._number_board_jpeg: bytes | None = None
        self._last_frame_size = (1280, 720)
        # Retain a short 30 FPS burst. A single "latest frame" slot can skip
        # the only sharp, fully visible number while OCR is busy on its predecessor.
        self._pending_frames: deque[Frame] = deque(maxlen=6)
        self._frame_condition = Condition()
        self._worker_stop = Event()
        self._worker = Thread(
            target=self._processing_loop,
            name="number-recognition-latest-frame",
            daemon=True,
        )
        self._worker.start()
        self.finish_line = FinishLine()
        camera_runtime.add_callback(self.process_frame)

    def configure(self, race: Race) -> None:
        self.configure_source(race.camera_identifier, race_id=race.id)

    def configure_source(self, source_identifier: str, *, race_id: int | None = None) -> None:
        """Build the same downstream pipeline for live cameras and uploaded video."""

        synthetic = source_identifier == "synthetic"
        uploaded_video = source_identifier.startswith("video:")
        detector: ObjectDetector
        ocr_engine: OcrEngine
        tracker: ObjectTracker
        region_extractor: NumberRegionExtractor
        if synthetic:
            detector = MetadataObjectDetector()
            ocr_engine = MetadataOcrEngine()
            tracker = CentroidIoUTracker(maximum_centroid_distance=260)
            region_extractor = BoundingBoxNumberRegionExtractor()
        else:
            self.prepare_models()
            assert self._physical_motorcycle_detector is not None
            assert self._physical_ocr_engine is not None
            detector = (
                self._physical_motorcycle_detector
                if uploaded_video
                else MotionAwareMotorcycleDetector(
                    self._physical_motorcycle_detector,
                    full_frame_interval=8,
                )
            )
            ocr_engine = self._physical_ocr_engine
            tracker = SupervisionByteTrack(
                track_activation_threshold=0.15,
                lost_track_buffer=30,
                minimum_matching_threshold=0.70,
                frame_rate=30,
            )
            region_extractor = FrontNumberBoardRegionExtractor(maximum_candidates=3)
        pipeline = MotorcycleVisionPipeline(
            detector=detector,
            tracker=tracker,
            region_extractor=region_extractor,
            ocr_engine=ocr_engine,
            ocr_aggregator=OcrAggregator(
                OcrAggregationConfig(
                    maximum_length=4,
                    resolution_confidence=0.62,
                    # Do not publish a possibly cropped number from one frame.
                    # A unique sharp frame is still usable by the dedicated
                    # crossing-time recovery pass over the retained burst.
                    instant_resolution_confidence=None,
                    minimum_observations=2,
                    minimum_consecutive=2,
                    maximum_observation_age_ns=3_000_000_000,
                )
            ),
            finish_line=self.finish_line,
            recovery_delay_ns=0 if synthetic else 300_000_000,
            # Conventional multi-frame OCR is deterministic and bounded. The
            # experimental autoregressive verifier was removed from production
            # because a worker failure could stall an otherwise short video.
            number_verifier=None,
            # Uploaded races only need expensive OCR for a real geometric
            # crossing. Recovering every short-lived tracker fragment on exit
            # made dense videos several times slower without creating laps.
            recover_identities_on_exit=not uploaded_video,
        )
        self._clear_pending_frames()
        with self._lock:
            previous = self._pipeline
            self._pipeline = pipeline
            self._race_id = race_id
            self._recognized_number = None
            self._last_error = None
            self._track_overlays = []
            self._board_overlays = []
            self._digit_overlays = []
            self._board_overlay_cache = {}
            self._digit_overlay_cache = {}
            self._track_overlay_cache = {}
            self._stable_by_track = {}
            self._last_recognition_by_number = {}
            self._recognition_history.clear()
            self._number_board_jpeg = None
            self._last_frame_size = (1280, 720)
            self._enabled = True
            if previous is not None:
                previous.close()

    def prepare_models(self) -> None:
        """Load and warm reusable detector/OCR sessions once per process."""

        with self._lock:
            if self._physical_motorcycle_detector is None:
                self._physical_motorcycle_detector = YoloXMotorcycleDetector(
                    confidence_threshold=0.18
                )
            if self._physical_ocr_engine is None:
                try:
                    fast_engine = RapidOcrDigitEngine()
                    try:
                        self._physical_ocr_engine = HybridDigitOcrEngine(
                            fast_engine,
                            PaddleOcrV6DigitEngine(),
                        )
                    except Exception:
                        logger.exception(
                            "PP-OCRv6 unavailable; keeping the RapidOCR fallback"
                        )
                        self._physical_ocr_engine = fast_engine
                except Exception:
                    logger.exception("RapidOCR unavailable; using OpenCV digit fallback")
                    self._physical_ocr_engine = OpenCVDigitOcrEngine()

    def prepare_recognizer(self) -> None:
        """Compatibility entry point used by the camera-selection background task."""

        self.prepare_models()

    def scan_frame_sync(self, frame: Frame) -> bool:
        """Run only the reusable motorcycle detector during video pass one."""

        self.prepare_models()
        assert self._physical_motorcycle_detector is not None
        return bool(self._physical_motorcycle_detector.detect(frame))

    def set_finish_line(self, line: FinishLine) -> None:
        """Apply normalized line geometry immediately and clear old crossing state."""

        with self._lock:
            self.finish_line = line
            pipeline = self._pipeline
            if pipeline is None:
                return
            pipeline.set_finish_line(line)
            self._recognized_number = None

    def pause(self) -> None:
        with self._lock:
            self._enabled = False
            if self._pipeline:
                self._pipeline.reset()
        self._clear_pending_frames()

    def resume(self) -> None:
        self._clear_pending_frames()
        with self._lock:
            if self._pipeline:
                self._pipeline.reset()
                self._enabled = True

    def stop(self) -> None:
        with self._lock:
            pipeline, self._pipeline = self._pipeline, None
            self._enabled = False
            self._race_id = None
            self._recognized_number = None
            self._last_error = None
            self._track_overlays = []
            self._board_overlays = []
            self._digit_overlays = []
            self._board_overlay_cache = {}
            self._digit_overlay_cache = {}
            self._track_overlay_cache = {}
            self._stable_by_track = {}
            self._last_recognition_by_number = {}
            self._recognition_history.clear()
            self._number_board_jpeg = None
            if pipeline:
                pipeline.close()
        self._clear_pending_frames()

    def _clear_pending_frames(self) -> None:
        with self._frame_condition:
            self._pending_frames.clear()

    def process_frame(self, frame: Frame) -> None:
        """Non-blocking camera callback: retain a short fast-passage burst."""

        with self._frame_condition:
            self._pending_frames.append(frame)
            self._frame_condition.notify()

    def process_frame_sync(self, frame: Frame) -> None:
        """Process every frame of an offline source without stale-frame dropping."""

        self._process_latest_frame(frame)

    def collect_evidence_frame_sync(self, frame: Frame) -> None:
        """Collect an offline in-between frame without another YOLO inference."""

        with self._lock:
            if self._enabled and self._pipeline is not None:
                self._pipeline.collect_evidence(frame)

    def track_near_finish_line(self) -> bool:
        """Tell the offline scheduler when exact per-frame geometry is valuable."""

        with self._lock:
            return bool(
                self._enabled
                and self._pipeline is not None
                and self._pipeline.track_near_finish_line(self._last_frame_size)
            )

    def begin_candidate_window_sync(self) -> None:
        """Drop transient tracking state between separate offline passages."""

        with self._lock:
            if self._enabled and self._pipeline is not None:
                self._pipeline.reset()
                self._stable_by_track = {}
                self._recognized_number = None

    def _processing_loop(self) -> None:
        while not self._worker_stop.is_set():
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: bool(self._pending_frames) or self._worker_stop.is_set(),
                    timeout=0.5,
                )
                if self._worker_stop.is_set():
                    return
                frame = self._pending_frames.pop() if self._pending_frames else None
                if frame is not None and self._pending_frames:
                    self._dropped_inference_frames += len(self._pending_frames)
                    self._pending_frames.clear()
            if frame is not None:
                self._process_latest_frame(frame)

    def _process_latest_frame(self, frame: Frame) -> None:
        with self._lock:
            if not self._enabled or self._pipeline is None:
                return
            started_ns = time.perf_counter_ns()
            try:
                result = self._pipeline.process(frame)
                self._last_frame_size = (int(frame.image.shape[1]), int(frame.image.shape[0]))
                self._recognized_number = result.recognized_number
                numbers = {
                    track_id: (number, confidence)
                    for track_id, number, confidence in result.track_numbers
                }
                height, width = frame.image.shape[:2]
                current_track_overlays = [
                    self._smoothed_track_overlay(
                        {
                        "track_id": track.track_id,
                        "x1": track.bbox.x1 / width,
                        "y1": track.bbox.y1 / height,
                        "x2": track.bbox.x2 / width,
                        "y2": track.bbox.y2 / height,
                        "detector_confidence": round(track.confidence, 3),
                        "racing_number": numbers.get(track.track_id, (None, 0))[0],
                        },
                        frame.sequence,
                    )
                    for track in result.tracks
                    if track.observed
                ]
                observed_track_ids = {
                    int(overlay["track_id"]) for overlay in current_track_overlays
                }
                for track_id, (seen_at, _overlay) in tuple(self._track_overlay_cache.items()):
                    if track_id not in observed_track_ids and frame.sequence - seen_at > 8:
                        self._track_overlay_cache.pop(track_id, None)
                self._track_overlays = current_track_overlays
                self._board_overlays = self._update_stage_overlays(
                    self._board_overlay_cache,
                    result.board_regions,
                    frame.sequence,
                    width,
                    height,
                )
                self._digit_overlays = self._update_stage_overlays(
                    self._digit_overlay_cache,
                    result.digit_regions,
                    frame.sequence,
                    width,
                    height,
                )
                for track_id, number, confidence in result.track_numbers:
                    if self._stable_by_track.get(track_id) == number:
                        continue
                    self._stable_by_track[track_id] = number
                    last_seen_ns = self._last_recognition_by_number.get(number)
                    self._last_recognition_by_number[number] = frame.captured_monotonic_ns
                    if (
                        last_seen_ns is not None
                        and frame.captured_monotonic_ns - last_seen_ns < 3_000_000_000
                    ):
                        continue
                    self._recognition_history.appendleft(
                        {
                            "racing_number": number,
                            "timestamp": frame.captured_at_utc.isoformat(),
                            "state": "recognized",
                            "track_id": track_id,
                            "confidence": round(confidence, 3),
                            "crossed": False,
                            "lap_number": None,
                        }
                    )
                self._encode_debug_crop(result.number_board_crop)
                self._last_error = None
                for passage in result.passages:
                    self._save_passage(passage)
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Number recognition failed")
            finally:
                self._last_processing_latency_ms = (
                    time.perf_counter_ns() - started_ns
                ) / 1_000_000

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "race_id": self._race_id,
                "recognized_number": self._recognized_number,
                "finish_line": self.finish_line.as_dict(),
                "tracks": list(self._track_overlays),
                "boards": list(self._board_overlays),
                "digit_regions": list(self._digit_overlays),
                "recognition_history": list(self._recognition_history),
                "processing_latency_ms": round(self._last_processing_latency_ms, 1),
                "dropped_inference_frames": self._dropped_inference_frames,
                "error": self._last_error,
            }

    @staticmethod
    def _update_stage_overlays(
        cache: dict[int, tuple[int, dict[str, Any]]],
        regions: Any,
        frame_sequence: int,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        for region in regions:
            overlay = _normalized_region(region, width, height)
            previous = cache.get(region.track_id)
            if previous is not None and frame_sequence - previous[0] <= 3:
                overlay = _smooth_normalized_box(previous[1], overlay, alpha=0.72)
            cache[region.track_id] = (frame_sequence, overlay)
        for track_id, (seen_at, _overlay) in tuple(cache.items()):
            if frame_sequence - seen_at > 6:
                cache.pop(track_id, None)
        return [value[1] for value in cache.values()]

    def _smoothed_track_overlay(
        self, overlay: dict[str, Any], frame_sequence: int
    ) -> dict[str, Any]:
        track_id = int(overlay["track_id"])
        previous = self._track_overlay_cache.get(track_id)
        if previous is not None and frame_sequence - previous[0] <= 3:
            overlay = _smooth_normalized_box(previous[1], overlay, alpha=0.72)
        self._track_overlay_cache[track_id] = (frame_sequence, overlay)
        return overlay

    def number_board_snapshot(self) -> bytes | None:
        with self._lock:
            return self._number_board_jpeg

    def _encode_debug_crop(self, image: Any | None) -> None:
        if image is None or not hasattr(image, "shape") or getattr(image, "size", 0) == 0:
            return
        try:
            import cv2

            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                self._number_board_jpeg = bytes(encoded)
        except Exception:
            logger.debug("Could not encode number-board debug crop", exc_info=True)

    def _save_passage(self, passage: StablePassage) -> None:
        race_id = self._race_id
        if race_id is None:
            return
        with SessionLocal() as session:
            decision = LapTimingService(session).record_passage(
                race_id,
                PassageCandidate(
                    racing_number=passage.racing_number,
                    captured_monotonic_ns=passage.captured_monotonic_ns,
                    detected_at_utc=passage.detected_at_utc,
                    recognition_confidence=passage.confidence,
                    track_id=str(passage.track_id),
                    raw_recognition=passage.racing_number,
                    idempotency_key=passage.idempotency_key,
                ),
            )
            session.commit()
            if decision.lap is not None:
                lap = decision.lap
                self._recognition_history.appendleft(
                    {
                        "racing_number": lap.racing_number,
                        "timestamp": lap.detected_at_utc.isoformat(),
                        "state": "lap_recorded",
                        "track_id": passage.track_id,
                        "confidence": round(passage.confidence, 3),
                        "crossed": True,
                        "lap_number": lap.lap_number,
                    }
                )
                live_event_hub.publish_threadsafe(
                    "lap_recorded",
                    {
                        "race_id": race_id,
                        "lap": {
                            "id": lap.id,
                            "racing_number": lap.racing_number,
                            "lap_number": lap.lap_number,
                            "lap_time_ns": lap.lap_time_ns,
                            "race_elapsed_ns": lap.race_elapsed_ns,
                            "detected_at_utc": lap.detected_at_utc.isoformat(),
                        },
                    },
                )
                logger.info(
                    "Lap recorded",
                    extra={
                        "race_id": race_id,
                        "racing_number": lap.racing_number,
                        "lap_number": lap.lap_number,
                    },
                )


vision_runtime = VisionRuntime()


def _normalized_region(region: Any, width: int, height: int) -> dict[str, Any]:
    return {
        "track_id": region.track_id,
        "x1": region.bbox.x1 / width,
        "y1": region.bbox.y1 / height,
        "x2": region.bbox.x2 / width,
        "y2": region.bbox.y2 / height,
        "confidence": round(region.confidence, 3),
        "kind": region.kind,
        "text": region.text,
    }


def _smooth_normalized_box(
    previous: dict[str, Any], current: dict[str, Any], *, alpha: float
) -> dict[str, Any]:
    """Visual-only exponential smoothing; timing keeps raw tracker geometry."""

    result = dict(current)
    for key in ("x1", "y1", "x2", "y2"):
        result[key] = alpha * float(current[key]) + (1.0 - alpha) * float(previous[key])
    return result
