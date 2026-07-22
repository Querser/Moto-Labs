"""Safe repository-local storage and inspection for uploaded race videos."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from app.config import Settings

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_VIDEO_CONTENT_TYPES = {
    "application/octet-stream",
    "video/avi",
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/x-msvideo",
}


class VideoUploadError(ValueError):
    """An upload is unsafe, empty, corrupt, or unreadable."""


@dataclass(frozen=True, slots=True)
class UploadedVideo:
    id: str
    original_name: str
    stored_name: str
    extension: str
    content_type: str
    size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    uploaded_at_utc: str

    @property
    def identifier(self) -> str:
        return f"video:{self.id}"

    def as_public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["identifier"] = self.identifier
        return result


class VideoCatalog:
    """Persist upload metadata beside generated video filenames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.video_upload_dir.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, stream: BinaryIO, original_name: str, content_type: str | None) -> UploadedVideo:
        safe_original = Path(original_name or "video").name
        extension = Path(safe_original).suffix.lower()
        if extension not in ALLOWED_VIDEO_EXTENSIONS:
            raise VideoUploadError("Поддерживаются только MP4, MOV, AVI и MKV.")
        normalized_type = (content_type or "application/octet-stream").lower()
        if normalized_type not in ALLOWED_VIDEO_CONTENT_TYPES:
            raise VideoUploadError("Тип загружаемого файла не является поддерживаемым видео.")

        video_id = uuid.uuid4().hex
        stored_name = f"{video_id}{extension}"
        target = self.root / stored_name
        temporary = self.root / f".{video_id}.upload"
        size = 0
        try:
            with temporary.open("xb") as output:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.video_upload_max_bytes:
                        raise VideoUploadError(
                            "Видео превышает лимит "
                            f"{self.settings.video_upload_max_bytes // (1024 * 1024)} МБ."
                        )
                    output.write(chunk)
            if size == 0:
                raise VideoUploadError("Загруженный видеофайл пуст.")
            os.replace(temporary, target)
            probe = _probe_video(target)
            uploaded = UploadedVideo(
                id=video_id,
                original_name=safe_original[:255],
                stored_name=stored_name,
                extension=extension,
                content_type=normalized_type,
                size_bytes=size,
                uploaded_at_utc=datetime.now(timezone.utc).isoformat(),
                **probe,
            )
            self._metadata_path(video_id).write_text(
                json.dumps(asdict(uploaded), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return uploaded
        except Exception:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise

    def list(self) -> list[UploadedVideo]:
        videos: list[UploadedVideo] = []
        for metadata_path in self.root.glob("*.json"):
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                item = UploadedVideo(**data)
                if (self.root / item.stored_name).is_file():
                    videos.append(item)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(videos, key=lambda item: item.uploaded_at_utc, reverse=True)

    def get(self, video_id: str) -> UploadedVideo:
        if not _valid_id(video_id):
            raise VideoUploadError("Некорректный идентификатор видео.")
        path = self._metadata_path(video_id)
        try:
            item = UploadedVideo(**json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError as exc:
            raise VideoUploadError("Загруженное видео не найдено.") from exc
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VideoUploadError("Метаданные видео повреждены.") from exc
        if not self.path_for(item).is_file():
            raise VideoUploadError("Файл загруженного видео не найден.")
        return item

    def path_for(self, video: UploadedVideo) -> Path:
        candidate = (self.root / video.stored_name).resolve()
        if candidate.parent != self.root:
            raise VideoUploadError("Некорректный путь видео.")
        return candidate

    def _metadata_path(self, video_id: str) -> Path:
        return self.root / f"{video_id}.json"


def _valid_id(value: str) -> bool:
    return len(value) == 32 and all(character in "0123456789abcdef" for character in value)


def _probe_video(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise VideoUploadError("OpenCV недоступен для проверки видео.") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoUploadError("Видео не открывается: формат или кодек не поддерживается.")
        ok, frame = capture.read()
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            raise VideoUploadError("Видео повреждено или не содержит читаемых кадров.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(frame.shape[1])
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(frame.shape[0])
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise VideoUploadError(
                "Не удалось определить частоту или количество кадров видео."  # noqa: RUF001
            )
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)).strip()
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": frame_count / fps,
            "codec": codec or "unknown",
        }
    finally:
        capture.release()


__all__ = [
    "ALLOWED_VIDEO_EXTENSIONS",
    "UploadedVideo",
    "VideoCatalog",
    "VideoUploadError",
]
