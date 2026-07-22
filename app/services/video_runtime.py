"""Sequential uploaded-video processing with source-timeline preservation."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from threading import Condition, RLock, Thread, current_thread
from typing import Any

from app.camera import CameraConfig, EndOfStream, Frame, VideoFileSource
from app.config import get_settings
from app.models import Race
from app.services.vision_runtime import vision_runtime
from app.video_uploads import UploadedVideo, VideoCatalog
from app.vision import OpenCVMotionDetector

logger = logging.getLogger(__name__)


class UploadedVideoRuntime:
    """Run the shared vision pipeline over every source frame in order."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._thread: Thread | None = None
        self._source: VideoFileSource | None = None
        self._video: UploadedVideo | None = None
        self._race_id: int | None = None
        self._timeline_origin_ns = 0
        self._timeline_origin_utc = datetime.now(timezone.utc)
        self._state = "idle"
        self._stop_requested = False
        self._pause_requested = False
        self._restart_requested = False
        self._processed_frames = 0
        self._scan_processed_frames = 0
        self._analyzed_frames = 0
        self._analysis_frame_step = 1
        self._active_analysis_frame_step = 1
        self._motion_hold_until_sequence = 0
        self._motion_detector: OpenCVMotionDetector | None = None
        self._phase = "idle"
        self._candidate_windows: list[tuple[int, int]] = []
        self._scan_frame_step = 1
        self._latest_frame: Frame | None = None
        self._latest_jpeg: bytes | None = None
        self._last_error: str | None = None

    def start(
        self,
        video_id: str,
        race: Race,
        *,
        timeline_origin_ns: int,
        start_position_ns: int = 0,
        defer_processing: bool = False,
    ) -> None:
        catalog = VideoCatalog(get_settings())
        video = catalog.get(video_id)
        self.stop(stop_vision=False)
        vision_runtime.configure_source(video.identifier, race_id=race.id)
        started_at = race.started_at_utc or datetime.now(timezone.utc)
        # SQLite does not preserve timezone metadata even when SQLAlchemy's
        # column is declared with ``timezone=True``.  Treat persisted UTC
        # timestamps as UTC before deriving per-frame wall-clock timestamps.
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        source = VideoFileSource(
            catalog.path_for(video),
            config=CameraConfig(target_fps=max(video.fps, 1.0)),
            realtime=False,
            source_id=video.identifier,
            timeline_origin_ns=timeline_origin_ns,
            timeline_origin_utc=started_at,
            start_position_ms=max(0, start_position_ns) / 1_000_000,
        )
        source.open()
        with self._condition:
            self._video = video
            self._race_id = race.id
            self._timeline_origin_ns = timeline_origin_ns
            self._timeline_origin_utc = started_at
            self._source = source
            self._state = "processing"
            self._stop_requested = False
            self._pause_requested = False
            self._restart_requested = False
            self._processed_frames = max(
                0,
                round(max(0, start_position_ns) * video.fps / 1_000_000_000),
            )
            self._scan_processed_frames = 0
            self._analyzed_frames = 0
            # Idle video is sampled sparsely. Motion immediately opens a full-
            # cadence burst so every motorcycle keeps source-frame timing near
            # the line, including two riders separated by only a few frames.
            self._analysis_frame_step = max(1, round(video.fps / 6.0))
            self._active_analysis_frame_step = max(1, round(video.fps / 30.0))
            self._motion_hold_until_sequence = 0
            self._motion_detector = OpenCVMotionDetector(
                minimum_area=700,
                history=180,
                variance_threshold=24.0,
            )
            self._phase = "scanning"
            self._candidate_windows = []
            self._scan_frame_step = max(1, round(video.fps / 3.0))
            self._latest_frame = None
            self._latest_jpeg = None
            self._last_error = None
            self._thread = Thread(
                target=self._run,
                name="uploaded-video-processing",
                daemon=True,
            )
            if not defer_processing:
                self._thread.start()

    def activate(self) -> None:
        """Start a source prepared before a database transaction was committed."""

        with self._condition:
            if self._thread is None or self._source is None:
                raise RuntimeError("Uploaded video is not prepared")
            if not self._thread.is_alive():
                self._thread.start()

    def pause(self) -> None:
        with self._condition:
            if self._state == "processing":
                self._pause_requested = True
                self._state = "paused"

    def resume(self) -> None:
        with self._condition:
            if self._state == "paused":
                self._pause_requested = False
                self._state = "processing"
                self._condition.notify_all()

    def restart(self) -> None:
        with self._condition:
            if self._source is None or self._video is None:
                raise RuntimeError("Uploaded video is not active")
            was_paused = self._state == "paused"
            self._restart_requested = True
            self._pause_requested = was_paused
            self._state = "paused" if was_paused else "processing"
            if self._thread is None or not self._thread.is_alive():
                self._stop_requested = False
                self._thread = Thread(
                    target=self._run,
                    name="uploaded-video-processing",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()

    def stop(self, *, stop_vision: bool = True) -> None:
        with self._condition:
            self._stop_requested = True
            self._pause_requested = False
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=5.0)
        with self._condition:
            source, self._source = self._source, None
            self._thread = None
            self._state = "idle"
            self._phase = "idle"
            self._video = None
            self._race_id = None
        if source is not None:
            source.close()
        if stop_vision:
            vision_runtime.stop()

    def logical_now_ns(self) -> int:
        with self._lock:
            if self._phase == "analyzing" and self._latest_frame is not None:
                return self._latest_frame.captured_monotonic_ns
            source = self._source
            initial_offset_ns = (
                round(source.start_position_ms * 1_000_000) if source is not None else 0
            )
            return self._timeline_origin_ns + initial_offset_ns

    def status(self) -> dict[str, Any]:
        with self._lock:
            video = self._video
            frame = self._latest_frame
            vision_status = vision_runtime.status()
            return {
                **vision_status,
                "state": self._state,
                "phase": self._phase,
                "video": video.as_public_dict() if video else None,
                "race_id": self._race_id,
                "processed_frames": self._processed_frames,
                "scan_processed_frames": self._scan_processed_frames,
                "analyzed_frames": self._analyzed_frames,
                "analysis_frame_step": self._analysis_frame_step,
                "total_frames": video.frame_count if video else 0,
                "progress": (
                    self._progress(video.frame_count)
                    if video and video.frame_count
                    else 0.0
                ),
                "video_position_ms": (
                    frame.metadata.get("video_position_ms")
                    if frame is not None and self._phase == "analyzing"
                    else None
                ),
                "error": self._last_error,
            }

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: not self._pause_requested
                    or self._stop_requested
                    or self._restart_requested
                )
                if self._stop_requested:
                    return
                if self._restart_requested:
                    self._restart_source()
                    self._restart_requested = False
                    if self._pause_requested:
                        continue
                source = self._source
            if source is None:
                return
            try:
                frame = source.read()
                if self._phase == "scanning":
                    self._scan_frame(frame)
                    continue
                if not self._inside_candidate_window(frame.sequence):
                    with self._lock:
                        self._latest_frame = frame
                        self._processed_frames += 1
                        self._last_error = None
                    continue
                # Pass one already reduced the file to short motorcycle
                # windows. Preserve every source frame inside those windows so
                # two riders and the interpolated line timestamp retain the
                # original camera cadence.
                frame = replace(
                    frame,
                    metadata={
                        **frame.metadata,
                        "deferred_ocr": True,
                        "motorcycle_window": True,
                    },
                )
                vision_runtime.process_frame_sync(frame)
                jpeg = _encode_jpeg(frame)
                with self._lock:
                    self._latest_frame = frame
                    if jpeg is not None:
                        self._latest_jpeg = jpeg
                    self._processed_frames += 1
                    self._analyzed_frames += 1
                    self._last_error = None
            except EndOfStream:
                if self._phase == "scanning":
                    self._begin_analysis_pass(source)
                    if self._candidate_windows:
                        continue
                with self._lock:
                    self._state = "completed"
                return
            except Exception as exc:
                logger.exception("Uploaded video processing failed")
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._state = "error"
                return

    def _restart_source(self) -> None:
        source = self._source
        if source is None:
            return
        source.close()
        source.open()
        video = self._video
        self._processed_frames = max(
            0,
            round(
                source.start_position_ms
                * max(video.fps if video is not None else source.config.target_fps, 0.001)
                / 1000.0
            ),
        )
        self._analyzed_frames = 0
        self._scan_processed_frames = 0
        self._latest_frame = None
        self._latest_jpeg = None
        self._motion_hold_until_sequence = 0
        self._motion_detector = OpenCVMotionDetector(
            minimum_area=700,
            history=180,
            variance_threshold=24.0,
        )
        self._phase = "scanning"
        self._candidate_windows = []
        vision_runtime.configure_source(source.identifier, race_id=self._race_id)

    def _scan_frame(self, frame: Frame) -> None:
        """Find broad passage windows without running tracking or OCR."""

        motion = bool(self._motion_detector.detect(frame)) if self._motion_detector else True
        fps = max(1.0, float(frame.metadata.get("source_fps", 30.0)))
        sampled = frame.sequence % self._scan_frame_step == 0
        periodic_full_scan = frame.sequence % max(1, round(fps)) == 0
        should_detect = sampled and (motion or periodic_full_scan)
        detected = vision_runtime.scan_frame_sync(frame) if should_detect else False
        if detected:
            self._add_candidate_window(
                max(0, frame.sequence - round(fps * 1.5)),
                frame.sequence + round(fps * 2.5),
                merge_gap=round(fps * 0.75),
            )
        jpeg = _encode_jpeg(frame) if should_detect else None
        with self._lock:
            self._latest_frame = frame
            if jpeg is not None:
                self._latest_jpeg = jpeg
            self._scan_processed_frames += 1
            self._last_error = None

    def _add_candidate_window(self, start: int, end: int, *, merge_gap: int) -> None:
        if self._candidate_windows and start <= self._candidate_windows[-1][1] + merge_gap:
            previous_start, previous_end = self._candidate_windows[-1]
            self._candidate_windows[-1] = (previous_start, max(previous_end, end))
            return
        self._candidate_windows.append((start, end))

    def _inside_candidate_window(self, sequence: int) -> bool:
        return any(start <= sequence <= end for start, end in self._candidate_windows)

    def _begin_analysis_pass(self, source: VideoFileSource) -> None:
        """Rewind once and run the exact pipeline only inside detected windows."""

        source.close()
        source.open()
        with self._lock:
            self._phase = "analyzing"
            self._processed_frames = max(
                0,
                round(
                    source.start_position_ms
                    * max(source.config.target_fps, 0.001)
                    / 1000.0
                ),
            )
            self._analyzed_frames = 0
            self._latest_frame = None
            self._latest_jpeg = None
            self._motion_hold_until_sequence = 0
            self._motion_detector = OpenCVMotionDetector(
                minimum_area=700,
                history=180,
                variance_threshold=24.0,
            )

    def _progress(self, total_frames: int) -> float:
        if total_frames <= 0:
            return 0.0
        if self._phase == "scanning":
            return min(0.20, 0.20 * self._scan_processed_frames / total_frames)
        if self._phase == "analyzing":
            return min(1.0, 0.20 + 0.80 * self._processed_frames / total_frames)
        return 1.0 if self._state == "completed" else 0.0


def _encode_jpeg(frame: Frame) -> bytes | None:
    try:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return bytes(encoded) if ok else None
    except Exception:
        logger.debug("Could not encode uploaded-video preview frame", exc_info=True)
        return None


video_runtime = UploadedVideoRuntime()

__all__ = ["UploadedVideoRuntime", "video_runtime"]
