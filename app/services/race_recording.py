"""Non-blocking continuous recording for a live-camera race."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, RLock, Thread, current_thread
from typing import Any

from app.camera import Frame
from app.config import get_settings

logger = logging.getLogger(__name__)


class RaceRecordingService:
    """Write source frames on a dedicated thread without delaying capture/OCR."""

    def __init__(self, *, queue_size: int = 45) -> None:
        self._lock = RLock()
        self._condition = Condition(self._lock)
        # The first VideoWriter initialization can briefly be slower than the
        # camera. Keep at least one 30 FPS second so that startup does not lose
        # the first GoPro frames while retaining a strict memory bound.
        self._queue: deque[Frame] = deque(maxlen=max(32, queue_size))
        self._thread: Thread | None = None
        self._active = False
        self._stop_requested = False
        self._race_id: int | None = None
        self._source_identifier: str | None = None
        self._path: Path | None = None
        self._target_fps = 30.0
        self._frames_written = 0
        self._dropped_frames = 0
        self._error: str | None = None

    def start(
        self,
        race_id: int,
        source_identifier: str,
        *,
        target_fps: float = 30.0,
        directory: Path | None = None,
    ) -> Path:
        """Start one new immutable recording segment for a live race."""

        self.stop()
        root = (directory or get_settings().recording_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        path = root / f"race_{race_id:06d}_{stamp}.mp4"
        with self._condition:
            self._queue.clear()
            self._race_id = race_id
            self._source_identifier = source_identifier
            self._path = path
            self._target_fps = max(1.0, min(240.0, float(target_fps)))
            self._frames_written = 0
            self._dropped_frames = 0
            self._error = None
            self._stop_requested = False
            self._active = True
            self._thread = Thread(
                target=self._writer_loop,
                name=f"race-video-writer-{race_id}",
                daemon=True,
            )
            self._thread.start()
        logger.info(
            "Race video recording started",
            extra={"race_id": race_id, "recording_path": str(path)},
        )
        return path

    def process_frame(self, frame: Frame) -> None:
        """Camera callback that only enqueues a frame and returns immediately."""

        with self._condition:
            if not self._active or self._stop_requested:
                return
            if len(self._queue) == self._queue.maxlen:
                self._queue.popleft()
                self._dropped_frames += 1
            self._queue.append(frame)
            self._condition.notify()

    def stop(self, *, timeout: float = 10.0) -> None:
        """Flush queued frames and close the container cleanly."""

        with self._condition:
            thread = self._thread
            if thread is None:
                self._active = False
                return
            self._stop_requested = True
            self._condition.notify_all()
        if thread is not current_thread():
            thread.join(timeout=max(0.0, timeout))
        with self._condition:
            if self._thread is thread and not thread.is_alive():
                self._thread = None
                self._active = False

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "active": self._active and not self._stop_requested,
                "race_id": self._race_id,
                "source_identifier": self._source_identifier,
                "filename": self._path.name if self._path else None,
                "path": str(self._path) if self._path else None,
                "frames_written": self._frames_written,
                "dropped_frames": self._dropped_frames,
                "queue_size": len(self._queue),
                "error": self._error,
            }

    def list_recordings(
        self, race_id: int, *, directory: Path | None = None
    ) -> list[dict[str, Any]]:
        root = (directory or get_settings().recording_dir).expanduser().resolve()
        if not root.is_dir():
            return []
        prefix = f"race_{race_id:06d}_"
        files = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file()
                and path.name.startswith(prefix)
                and path.suffix.lower() in {".mp4", ".avi"}
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        return [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "created_at_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "download_url": f"/api/races/{race_id}/recordings/{path.name}",
            }
            for path in files
        ]

    def resolve_recording(
        self, race_id: int, filename: str, *, directory: Path | None = None
    ) -> Path | None:
        root = (directory or get_settings().recording_dir).expanduser().resolve()
        if Path(filename).name != filename:
            return None
        if not filename.startswith(f"race_{race_id:06d}_"):
            return None
        path = (root / filename).resolve()
        if path.parent != root or path.suffix.lower() not in {".mp4", ".avi"}:
            return None
        return path if path.is_file() else None

    def _writer_loop(self) -> None:
        import cv2

        writer: Any | None = None
        output_size: tuple[int, int] | None = None
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: bool(self._queue) or self._stop_requested,
                        timeout=0.5,
                    )
                    if self._queue:
                        frame = self._queue.popleft()
                    elif self._stop_requested:
                        break
                    else:
                        continue
                    path = self._path
                    fps = self._target_fps
                image = frame.image
                if image is None or not hasattr(image, "shape") or image.size == 0:
                    continue
                height, width = int(image.shape[0]), int(image.shape[1])
                if writer is None:
                    if path is None:
                        raise RuntimeError("Recording path is not configured")
                    output_size = (width, height)
                    writer = cv2.VideoWriter(
                        str(path),
                        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                        fps,
                        output_size,
                    )
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV could not open the MP4 video writer")
                if output_size != (width, height):
                    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
                writer.write(image)
                with self._condition:
                    self._frames_written += 1
        except Exception as exc:
            logger.exception("Race video recording failed")
            with self._condition:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            if writer is not None:
                writer.release()
            with self._condition:
                self._active = False
                self._stop_requested = False
                self._thread = None
                self._queue.clear()
                self._condition.notify_all()
            logger.info(
                "Race video recording stopped",
                extra={"race_id": self._race_id, "frames_written": self._frames_written},
            )


race_recording = RaceRecordingService()


__all__ = ["RaceRecordingService", "race_recording"]
