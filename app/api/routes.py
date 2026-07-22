"""Minimal camera, race, lap, correction, and export API."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any, Literal, NoReturn

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import asc, desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.errors import ApiError
from app.camera import CameraConfig, discover_cameras, resolve_camera
from app.config import Settings, get_settings
from app.database import get_db
from app.domain import LapTimingService, RaceError, RaceNotFound
from app.exporter import ExcelLapExporter
from app.models import CameraSetting, LapRecord, Race, RaceStatus
from app.schemas import (
    CameraSelection,
    FinishLineUpdate,
    LapCorrection,
    RaceCreate,
    RaceRead,
    RaceUpdate,
)
from app.services.camera_runtime import camera_runtime
from app.services.live_events import live_event_hub
from app.services.race_recording import race_recording
from app.services.video_runtime import video_runtime
from app.services.vision_runtime import vision_runtime
from app.video_uploads import VideoCatalog, VideoUploadError
from app.vision import FinishLine

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
_camera_selection_lock = Lock()
camera_runtime.add_callback(race_recording.process_frame)


def _setting_line(setting: CameraSetting | None) -> FinishLine:
    if setting is None:
        return FinishLine()
    return FinishLine(
        x1=setting.finish_line_x1,
        y1=setting.finish_line_y1,
        x2=setting.finish_line_x2,
        y2=setting.finish_line_y2,
    )


def _camera_config(settings: Settings) -> CameraConfig:
    return CameraConfig(
        width=640,
        height=480,
        queue_size=settings.frame_queue_size,
        target_fps=30,
    )


def _start_camera_ready(
    identifier: str,
    settings: Settings,
    *,
    attempts: int = 1,
    ready_timeout: float = 5.0,
) -> str | None:
    """Start a source and verify a real frame, retrying transient driver release failures."""

    last_error = "camera did not produce a frame"
    camera_info = resolve_camera(identifier) if identifier.startswith("webcam:") else None
    if camera_info is not None and "gopro" in (camera_info.device_name or "").casefold():
        # The virtual device can open before the physical camera has delivered
        # a non-placeholder frame. The source validates that signal for 4 s.
        ready_timeout = max(ready_timeout, 8.0)
    # Multiple browser tabs must not restart a Windows driver while the first
    # request is still waiting for its first frame.
    with _camera_selection_lock:
        for attempt in range(attempts):
            camera_runtime.start(identifier, _camera_config(settings))
            if camera_runtime.wait_until_ready(timeout=ready_timeout):
                return None
            last_error = str(camera_runtime.metrics().get("last_error") or last_error)
            camera_runtime.stop()
            if attempt + 1 < attempts:
                time.sleep(1.0)
    return last_error


def _raise_not_found(exc: Exception) -> NoReturn:
    raise ApiError(404, "not_found", str(exc)) from exc


def _raise_race_error(exc: Exception) -> NoReturn:
    raise ApiError(409, "race_error", str(exc)) from exc


def _publish(background: BackgroundTasks, event: str, data: dict[str, Any]) -> None:
    background.add_task(live_event_hub.publish, event, data)


def _race_read(race: Race) -> RaceRead:
    now_ns = (
        video_runtime.logical_now_ns()
        if race.camera_identifier.startswith("video:")
        and video_runtime.status().get("race_id") == race.id
        else time.perf_counter_ns()
    )
    elapsed = LapTimingService(_object_session(race)).elapsed_ns(race, now_ns)
    return RaceRead.model_validate(race).model_copy(update={"elapsed_ns": elapsed})


def _object_session(race: Race) -> Session:
    from sqlalchemy import inspect

    session = inspect(race).session
    if session is None:
        raise RuntimeError("race is detached")
    return session


def _lap_payload(lap: LapRecord, required_laps: int) -> dict[str, Any]:
    return {
        "id": lap.id,
        "race_id": lap.race_id,
        "racing_number": lap.racing_number,
        "lap_number": lap.lap_number,
        "lap_time_ns": lap.lap_time_ns,
        "race_elapsed_ns": lap.race_elapsed_ns,
        "detected_at_utc": lap.detected_at_utc,
        "finished": lap.lap_number >= required_laps,
    }


@router.get("/health")
def health(session: Session = Depends(get_db)) -> dict[str, Any]:
    session.execute(select(1))
    return {"status": "ok", "version": get_settings().app_version, "local_only": True}


@router.get("/cameras")
def list_cameras(session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    cameras = [asdict(camera) for camera in discover_cameras(max_devices=8)]
    setting = session.get(CameraSetting, 1)
    selected = setting.camera_identifier if setting else None
    identifiers = {str(camera["identifier"]) for camera in cameras}
    if selected and selected.startswith("webcam:") and selected not in identifiers:
        raw_index = selected.partition(":")[2]
        active = camera_runtime.source_identifier == selected and camera_runtime.is_running
        cameras.insert(
            0,
            {
                "identifier": selected,
                "label": (
                    f"Встроенная / USB камера {int(raw_index) + 1} — подключена"
                    if active and raw_index.isdigit()
                    else "Сохранённая камера — не удалось проверить"
                ),
                "source_type": "webcam",
                "camera_index": int(raw_index) if raw_index.isdigit() else None,
                "backend": None,
                "width": None,
                "height": None,
                "fps": None,
            },
        )
    return cameras


@router.get("/camera")
def camera_state(session: Session = Depends(get_db)) -> dict[str, Any]:
    setting = session.get(CameraSetting, 1)
    runtime = vision_runtime.status()
    runtime["finish_line"] = _setting_line(setting).as_dict()
    metrics = camera_runtime.metrics()
    frame = camera_runtime.latest_frame
    resolution = None
    if frame is not None and hasattr(frame.image, "shape") and len(frame.image.shape) >= 2:
        resolution = f"{int(frame.image.shape[1])}x{int(frame.image.shape[0])}"
    return {
        "selected_camera": setting.camera_identifier if setting else None,
        "active_source": metrics.get("source"),
        "status": metrics["state"],
        "frame_age_ms": metrics.get("frame_age_ms"),
        "resolution": resolution,
        "recording": race_recording.status(),
        **runtime,
    }


@router.put("/camera")
def select_camera(
    payload: CameraSelection,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        previous_identifier = camera_runtime.source_identifier
        error = _start_camera_ready(payload.camera_identifier, settings)
        if error is not None:
            # A failed virtual/USB camera probe must not leave the existing
            # preview blank. Restore the last working source before reporting
            # the selection error to the browser.
            if previous_identifier and previous_identifier != payload.camera_identifier:
                recovery_error = _start_camera_ready(previous_identifier, settings)
                if recovery_error is not None:
                    logger.warning(
                        "Previous camera could not be restored",
                        extra={
                            "camera_identifier": previous_identifier,
                            "camera_error": recovery_error,
                        },
                    )
            if "black placeholder" in error:
                raise ApiError(
                    409,
                    "gopro_not_streaming",
                    "GoPro Webcam установлена, но камера отдаёт чёрную заглушку. "
                    "Включите GoPro, подключите её USB-C кабелем, выберите режим "
                    "GoPro Connect/Webcam и убедитесь, что предпросмотр GoPro Webcam активен.",
                )
            raise ApiError(
                409,
                "camera_unavailable",
                "Не удалось получить кадр с камеры. Закройте другие программы с видео "  # noqa: RUF001
                f"и повторите выбор. Диагностика: {error}",
            )
        setting = session.get(CameraSetting, 1)
        if setting is None:
            setting = CameraSetting(id=1, camera_identifier=payload.camera_identifier)
            session.add(setting)
        else:
            setting.camera_identifier = payload.camera_identifier
        session.flush()
        vision_runtime.set_finish_line(_setting_line(setting))
        session.commit()
        active_race = session.scalar(
            select(Race)
            .where(Race.status.in_([RaceStatus.RUNNING, RaceStatus.PAUSED]))
            .order_by(Race.created_at.desc())
        )
        needs_reconfigure = active_race is not None and (
            active_race.camera_identifier != payload.camera_identifier
            or vision_runtime.status().get("race_id") != active_race.id
        )
        if active_race is not None and active_race.camera_identifier != payload.camera_identifier:
            active_race.camera_identifier = payload.camera_identifier
            session.commit()
        if active_race is not None and needs_reconfigure:
            vision_runtime.configure(active_race)
            if payload.camera_identifier.startswith("webcam:"):
                race_recording.start(
                    active_race.id,
                    payload.camera_identifier,
                    directory=settings.recording_dir,
                )
            else:
                race_recording.stop()
            if active_race.status is RaceStatus.PAUSED:
                vision_runtime.pause()
    except ValueError as exc:
        session.rollback()
        raise ApiError(422, "invalid_camera", str(exc)) from exc
    result = camera_state(session)
    logger.info(
        "Camera selection applied",
        extra={
            "camera_identifier": payload.camera_identifier,
            "race_id": active_race.id if active_race else None,
        },
    )
    _publish(background, "camera_changed", result)
    if payload.camera_identifier.startswith("webcam:"):
        background.add_task(vision_runtime.prepare_recognizer)
    return result


@router.put("/camera/line")
def update_finish_line(
    payload: FinishLineUpdate,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    setting = session.get(CameraSetting, 1)
    if setting is None:
        setting = CameraSetting(id=1, camera_identifier="synthetic")
        session.add(setting)
    setting.finish_line_x1 = payload.x1
    setting.finish_line_y1 = payload.y1
    setting.finish_line_x2 = payload.x2
    setting.finish_line_y2 = payload.y2
    line = FinishLine(**payload.model_dump())
    vision_runtime.set_finish_line(line)
    session.commit()
    result = camera_state(session)
    logger.info("Finish line updated", extra={"finish_line": line.as_dict()})
    _publish(background, "finish_line_changed", {"finish_line": line.as_dict()})
    return result


@router.post("/camera/demo")
def start_camera_demo(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Start the placeholder animation without persisting it as a camera choice."""

    if _start_camera_ready("synthetic", settings, attempts=1) is not None:
        raise ApiError(
            409,
            "demo_unavailable",
            "Не удалось запустить демо-анимацию.",  # noqa: RUF001
        )
    return {"status": "running", "source": "synthetic"}


@router.get("/camera/frame")
def camera_frame() -> StreamingResponse:
    if not camera_runtime.is_running:
        raise ApiError(409, "camera_not_running", "Сначала выберите доступную камеру.")
    return StreamingResponse(
        camera_runtime.mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/camera/snapshot")
def camera_snapshot(after: int = Query(default=-1, ge=-1)) -> Response:
    """Serve one complete frame for browsers that render MJPEG unreliably."""

    if not camera_runtime.is_running:
        raise ApiError(409, "camera_not_running", "Сначала выберите доступную камеру.")
    snapshot = camera_runtime.wait_for_jpeg_snapshot(after, timeout=1.0)
    if snapshot is None:
        raise ApiError(503, "frame_not_ready", "Камера запускается, кадр ещё не готов.")
    sequence, jpeg = snapshot
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "X-Frame-Sequence": str(sequence),
        },
    )


@router.get("/camera/number-board", include_in_schema=False)
def number_board_snapshot() -> Response:
    """Expose the latest selected OCR crop for camera-alignment verification."""

    jpeg = vision_runtime.number_board_snapshot()
    if jpeg is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.post("/videos", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Validate and store one local race video under a generated filename."""

    catalog = VideoCatalog(settings)
    try:
        uploaded = await asyncio.to_thread(
            catalog.save,
            file.file,
            file.filename or "video",
            file.content_type,
        )
    except VideoUploadError as exc:
        raise ApiError(422, "invalid_video", str(exc)) from exc
    finally:
        await file.close()
    logger.info(
        "Race video uploaded",
        extra={"video_id": uploaded.id, "size_bytes": uploaded.size_bytes},
    )
    return uploaded.as_public_dict()


@router.get("/videos")
def list_uploaded_videos(settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    return [item.as_public_dict() for item in VideoCatalog(settings).list()]


@router.get("/videos/{video_id}/snapshot")
def uploaded_video_snapshot(
    video_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Return the first frame so a finish line can be edited before processing."""

    try:
        import cv2

        catalog = VideoCatalog(settings)
        item = catalog.get(video_id)
        capture = cv2.VideoCapture(str(catalog.path_for(item)))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise VideoUploadError("Не удалось прочитать первый кадр видео.")  # noqa: RUF001
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not encoded_ok:
            raise VideoUploadError("Не удалось подготовить предпросмотр видео.")  # noqa: RUF001
    except (VideoUploadError, ImportError) as exc:
        raise ApiError(422, "video_unavailable", str(exc)) from exc
    return Response(
        content=bytes(encoded),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/videos/{video_id}/media")
def uploaded_video_media(
    video_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Stream a validated local upload to the smooth browser preview."""

    try:
        catalog = VideoCatalog(settings)
        item = catalog.get(video_id)
        path = catalog.path_for(item)
    except VideoUploadError as exc:
        raise ApiError(404, "video_not_found", str(exc)) from exc
    media_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    return FileResponse(
        path,
        media_type=media_types.get(item.extension, "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/video/status")
def uploaded_video_status() -> dict[str, Any]:
    return video_runtime.status()


@router.get("/video/frame")
def uploaded_video_frame() -> Response:
    jpeg = video_runtime.snapshot()
    if jpeg is None:
        raise ApiError(503, "frame_not_ready", "Кадр загруженного видео ещё не готов.")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.post("/video/restart")
def restart_uploaded_video() -> dict[str, Any]:
    try:
        video_runtime.restart()
    except RuntimeError as exc:
        raise ApiError(409, "video_not_running", str(exc)) from exc
    return video_runtime.status()


@router.get("/races", response_model=list[RaceRead])
def list_races(session: Session = Depends(get_db)) -> list[RaceRead]:
    races = session.scalars(select(Race).order_by(Race.created_at.desc())).all()
    return [_race_read(race) for race in races]


@router.post("/races", response_model=RaceRead, status_code=201)
def create_race(payload: RaceCreate, session: Session = Depends(get_db)) -> RaceRead:
    service = LapTimingService(session)
    try:
        race = service.create_race(**payload.model_dump())
        if not payload.camera_identifier.startswith("video:"):
            setting = session.get(CameraSetting, 1)
            if setting is None:
                session.add(CameraSetting(id=1, camera_identifier=payload.camera_identifier))
            else:
                setting.camera_identifier = payload.camera_identifier
        session.commit()
    except (RaceError, IntegrityError) as exc:
        session.rollback()
        _raise_race_error(exc)
    logger.info(
        "Race created",
        extra={"race_id": race.id, "race_name": race.name, "required_laps": race.required_laps},
    )
    return _race_read(race)


@router.get("/races/{race_id}", response_model=RaceRead)
def get_race(race_id: int, session: Session = Depends(get_db)) -> RaceRead:
    try:
        return _race_read(LapTimingService(session).get_race(race_id))
    except RaceNotFound as exc:
        _raise_not_found(exc)


@router.patch("/races/{race_id}", response_model=RaceRead)
def update_race(
    race_id: int, payload: RaceUpdate, session: Session = Depends(get_db)
) -> RaceRead:
    try:
        race = LapTimingService(session).update_draft(
            race_id, **payload.model_dump(exclude_unset=True)
        )
        session.commit()
        return _race_read(race)
    except RaceNotFound as exc:
        _raise_not_found(exc)
    except RaceError as exc:
        session.rollback()
        _raise_race_error(exc)


def _transition(
    race_id: int,
    action: Literal["start", "pause", "resume", "finish"],
    session: Session,
    settings: Settings,
    background: BackgroundTasks,
) -> RaceRead:
    service = LapTimingService(session)
    activate_video = False
    try:
        if action == "start":
            race = service.get_race(race_id)
            is_uploaded_video = race.camera_identifier.startswith("video:")
            if not is_uploaded_video and (
                camera_runtime.source_identifier != race.camera_identifier
                or not camera_runtime.is_running
            ):
                error = _start_camera_ready(race.camera_identifier, settings)
                if error is not None:
                    raise RaceError(f"camera unavailable: {error}")
            now_ns = time.perf_counter_ns()
            # A single camera pipeline can belong to only one race.  Starting a
            # new race therefore closes any older running/paused race in the same
            # transaction, while preserving all of its laps and final time.
            closed_races = service.finish_other_active_races(race_id, now_ns)
            race = service.start(race_id, now_ns)
            vision_runtime.set_finish_line(_setting_line(session.get(CameraSetting, 1)))
            if is_uploaded_video:
                race_recording.stop()
                camera_runtime.stop()
                video_runtime.start(
                    race.camera_identifier.removeprefix("video:"),
                    race,
                    timeline_origin_ns=now_ns,
                    defer_processing=True,
                )
                activate_video = True
            else:
                video_runtime.stop(stop_vision=False)
                vision_runtime.configure(race)
                if race.camera_identifier.startswith("webcam:"):
                    race_recording.start(
                        race.id,
                        race.camera_identifier,
                        directory=settings.recording_dir,
                    )
                else:
                    race_recording.stop()
        elif action == "pause":
            existing = service.get_race(race_id)
            is_uploaded_video = existing.camera_identifier.startswith("video:")
            now_ns = video_runtime.logical_now_ns() if is_uploaded_video else time.perf_counter_ns()
            race = service.pause(race_id, now_ns)
            if is_uploaded_video:
                video_runtime.pause()
            vision_runtime.pause()
        elif action == "resume":
            existing = service.get_race(race_id)
            is_uploaded_video = existing.camera_identifier.startswith("video:")
            now_ns = video_runtime.logical_now_ns() if is_uploaded_video else time.perf_counter_ns()
            race = service.resume(race_id, now_ns)
            vision_runtime.resume()
            if is_uploaded_video:
                video_runtime.resume()
        else:
            existing = service.get_race(race_id)
            is_uploaded_video = existing.camera_identifier.startswith("video:")
            now_ns = video_runtime.logical_now_ns() if is_uploaded_video else time.perf_counter_ns()
            race = service.finish(race_id, now_ns)
            if is_uploaded_video:
                video_runtime.stop()
            else:
                vision_runtime.stop()
            race_recording.stop()
        session.commit()
        if activate_video:
            video_runtime.activate()
    except RaceNotFound as exc:
        session.rollback()
        _raise_not_found(exc)
    except RaceError as exc:
        session.rollback()
        _raise_race_error(exc)
    except Exception as exc:
        session.rollback()
        video_runtime.stop(stop_vision=False)
        vision_runtime.stop()
        race_recording.stop()
        raise ApiError(
            409,
            "runtime_unavailable",
            f"Could not {action} race runtime: {exc}",
        ) from exc
    _publish(background, "race_state", {"race_id": race_id, "status": race.status.value})
    if action == "start":
        for closed_race in closed_races:
            _publish(
                background,
                "race_state",
                {"race_id": closed_race.id, "status": closed_race.status.value},
            )
    logger.info(
        "Race state changed",
        extra={
            "race_id": race_id,
            "race_name": race.name,
            "race_action": action,
            "race_status": race.status.value,
            "auto_finished_race_ids": (
                [item.id for item in closed_races] if action == "start" else []
            ),
        },
    )
    return _race_read(race)


@router.get("/races/{race_id}/recordings")
def list_race_recordings(
    race_id: int,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[dict[str, Any]]:
    try:
        LapTimingService(session).get_race(race_id)
    except RaceNotFound as exc:
        _raise_not_found(exc)
    return race_recording.list_recordings(race_id, directory=settings.recording_dir)


@router.get("/races/{race_id}/recordings/{filename}")
def download_race_recording(
    race_id: int,
    filename: str,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    try:
        LapTimingService(session).get_race(race_id)
    except RaceNotFound as exc:
        _raise_not_found(exc)
    path = race_recording.resolve_recording(
        race_id, filename, directory=settings.recording_dir
    )
    if path is None:
        raise ApiError(404, "recording_not_found", "Race recording was not found")
    return FileResponse(path, filename=path.name, media_type="video/mp4")


@router.post("/races/{race_id}/start", response_model=RaceRead)
def start_race(
    race_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RaceRead:
    return _transition(race_id, "start", session, settings, background)


@router.post("/races/{race_id}/pause", response_model=RaceRead)
def pause_race(
    race_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RaceRead:
    return _transition(race_id, "pause", session, settings, background)


@router.post("/races/{race_id}/resume", response_model=RaceRead)
def resume_race(
    race_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RaceRead:
    return _transition(race_id, "resume", session, settings, background)


@router.post("/races/{race_id}/finish", response_model=RaceRead)
def finish_race(
    race_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RaceRead:
    return _transition(race_id, "finish", session, settings, background)


@router.get("/races/{race_id}/laps")
def list_laps(
    race_id: int,
    sort_by: Literal["number", "lap", "recorded"] = Query(default="recorded"),
    direction: Literal["asc", "desc"] = Query(default="asc"),
    session: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    try:
        race = LapTimingService(session).get_race(race_id)
    except RaceNotFound as exc:
        _raise_not_found(exc)
    columns = {
        "number": (LapRecord.racing_number, LapRecord.lap_number),
        "lap": (LapRecord.lap_number, LapRecord.racing_number),
        "recorded": (LapRecord.detected_at_utc, LapRecord.id),
    }[sort_by]
    order = [desc(column) if direction == "desc" else asc(column) for column in columns]
    laps = session.scalars(
        select(LapRecord).where(LapRecord.race_id == race_id).order_by(*order)
    ).all()
    return [_lap_payload(lap, race.required_laps) for lap in laps]


@router.get("/races/{race_id}/results")
def results(race_id: int, session: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        race = LapTimingService(session).get_race(race_id)
    except RaceNotFound as exc:
        _raise_not_found(exc)
    rows = session.execute(
        select(
            LapRecord.racing_number,
            func.count(LapRecord.id),
            func.sum(LapRecord.lap_time_ns),
        )
        .where(LapRecord.race_id == race_id)
        .group_by(LapRecord.racing_number)
        .order_by(LapRecord.racing_number)
    ).all()
    return {
        "race": _race_read(race),
        "summary": [
            {
                "racing_number": number,
                "completed_laps": count,
                "total_time_ns": total,
                "finished": count >= race.required_laps,
            }
            for number, count, total in rows
        ],
    }


@router.patch("/races/{race_id}/laps/{lap_id}")
def correct_lap(
    race_id: int,
    lap_id: int,
    payload: LapCorrection,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    service = LapTimingService(session)
    try:
        lap = service.correct_lap(race_id, lap_id, **payload.model_dump(exclude_none=True))
        race = service.get_race(race_id)
        session.commit()
    except RaceNotFound as exc:
        session.rollback()
        _raise_not_found(exc)
    except (RaceError, IntegrityError) as exc:
        session.rollback()
        _raise_race_error(exc)
    result = _lap_payload(lap, race.required_laps)
    logger.info("Lap corrected", extra={"race_id": race_id, "lap_id": lap_id})
    _publish(background, "lap_corrected", {"race_id": race_id, "lap": result})
    return result


@router.delete("/races/{race_id}/laps/{lap_id}", status_code=204)
def delete_lap(
    race_id: int,
    lap_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_db),
) -> None:
    try:
        LapTimingService(session).delete_lap(race_id, lap_id)
        session.commit()
    except RaceNotFound as exc:
        session.rollback()
        _raise_not_found(exc)
    logger.info("Lap deleted", extra={"race_id": race_id, "lap_id": lap_id})
    _publish(background, "lap_deleted", {"race_id": race_id, "lap_id": lap_id})


@router.get("/races/{race_id}/export")
def export_race(
    race_id: int,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    try:
        exported = ExcelLapExporter(settings.export_dir).export(session, race_id)
    except LookupError as exc:
        _raise_not_found(exc)
    logger.info(
        "Race results exported",
        extra={"race_id": race_id, "export_path": str(exported.path)},
    )
    return FileResponse(
        Path(exported.path),
        filename=exported.filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
