"""FastAPI application factory and local web entry point."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import __version__
from app.api.errors import ApiError, api_error_handler
from app.api.routes import router
from app.api.system import router as system_router
from app.backup import backup_sqlite_database
from app.camera import CameraConfig, resolve_camera
from app.config import get_settings
from app.database import DATABASE_URL, SessionLocal, engine, upgrade_database
from app.domain import LapTimingService
from app.logging_config import configure_logging
from app.models import CameraSetting, Race, RaceStatus, utc_now
from app.services.camera_runtime import camera_runtime
from app.services.live_events import live_event_hub
from app.services.race_recording import race_recording
from app.services.video_runtime import video_runtime
from app.services.vision_runtime import vision_runtime
from app.vision import FinishLine

logger = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")


def _start_restored_camera(source_identifier: str) -> bool:
    """Restore a physical preview, retrying a just-released virtual driver."""

    camera_info = (
        resolve_camera(source_identifier)
        if source_identifier.startswith("webcam:")
        else None
    )
    is_gopro = "gopro" in (camera_info.device_name or "").casefold() if camera_info else False
    attempts = 4 if is_gopro else (2 if source_identifier.startswith("webcam:") else 1)
    for attempt in range(attempts):
        camera_runtime.start(
            source_identifier,
            CameraConfig(width=640, height=480, target_fps=30),
        )
        # GoPro's virtual DirectShow driver may need several seconds after a
        # process restart before it replaces its placeholder with a real frame.
        # Hard driver errors still return early from wait_until_ready().
        if camera_runtime.wait_until_ready(timeout=8.0):
            return True
        camera_runtime.stop()
        if attempt + 1 < attempts:
            logger.info(
                "Retrying saved camera after driver release",
                extra={"source": source_identifier, "attempt": attempt + 2},
            )
            # Closing the failed handle briefly is important for virtual
            # DirectShow devices: reopening it in the same instant can return
            # the very same black placeholder indefinitely.
            time.sleep(2.0)
    return False


def _restore_active_runtime() -> None:
    """Restore the saved preview and reconnect an active race after restart."""

    with SessionLocal() as session:
        setting = session.get(CameraSetting, 1)
        line = (
            FinishLine(
                x1=setting.finish_line_x1,
                y1=setting.finish_line_y1,
                x2=setting.finish_line_x2,
                y2=setting.finish_line_y2,
            )
            if setting is not None
            else FinishLine()
        )
        vision_runtime.set_finish_line(line)
        race = session.scalar(
            select(Race)
            .where(Race.status.in_([RaceStatus.RUNNING, RaceStatus.PAUSED]))
            .order_by(Race.created_at.desc())
        )
        if race is None:
            # The preview has value before a race starts. Restore a persisted
            # physical camera; on a clean install (or transient driver error),
            # keep the lightweight demo alive instead of displaying a broken
            # image element. The demo is never persisted as a physical camera.
            source_identifier = (
                setting.camera_identifier
                if setting is not None
                and not setting.camera_identifier.startswith("video:")
                else "synthetic"
            )
            if not _start_restored_camera(source_identifier):
                logger.warning(
                    "Saved camera preview could not be restored; using demo",
                    extra={"source": source_identifier},
                )
                camera_runtime.stop()
                camera_runtime.start(
                    "synthetic",
                    CameraConfig(width=640, height=480, target_fps=30),
                )
                camera_runtime.wait_until_ready(timeout=2.0)
            logger.info(
                "Camera preview restored",
                extra={"source": camera_runtime.source_identifier},
            )
            return
        # Rebase persisted monotonic references on every process start. This is
        # exact for a gracefully paused race and safely reconstructs an active
        # race after an operating-system reboot.
        restored_elapsed_ns = LapTimingService(session).rebase_active_clock(
            race,
            time.perf_counter_ns(),
            utc_now(),
        )
        if setting is None:
            setting = CameraSetting(
                id=1,
                camera_identifier=(
                    "synthetic"
                    if race.camera_identifier.startswith("video:")
                    else race.camera_identifier
                ),
            )
            session.add(setting)
        elif not race.camera_identifier.startswith("video:"):
            setting.camera_identifier = race.camera_identifier
        session.commit()
        if race.camera_identifier.startswith("video:"):
            video_runtime.start(
                race.camera_identifier.removeprefix("video:"),
                race,
                timeline_origin_ns=race.monotonic_start_reference_ns or time.perf_counter_ns(),
                start_position_ns=(
                    (restored_elapsed_ns or 0)
                    if race.status is RaceStatus.PAUSED
                    else 0
                ),
            )
        else:
            if not _start_restored_camera(race.camera_identifier):
                logger.warning("Active race camera could not be restored")
                return
            vision_runtime.configure(race)
            if race.camera_identifier.startswith("webcam:"):
                race_recording.start(race.id, race.camera_identifier)
        if race.status is RaceStatus.PAUSED:
            video_runtime.pause()
            vision_runtime.pause()
        logger.info("Active race runtime restored", extra={"race_id": race.id})


def _pause_running_race_on_shutdown() -> None:
    """A stopped local application must not silently keep adding race time."""

    race_id = vision_runtime.status().get("race_id")
    with SessionLocal() as session:
        race = (
            session.get(Race, race_id)
            if isinstance(race_id, int)
            else session.scalar(
                select(Race)
                .where(Race.status == RaceStatus.RUNNING)
                .order_by(Race.created_at.desc())
            )
        )
        if race is not None and race.status is RaceStatus.RUNNING:
            now_ns = (
                video_runtime.logical_now_ns()
                if race.camera_identifier.startswith("video:")
                else time.perf_counter_ns()
            )
            LapTimingService(session).pause(race.id, now_ns)
            session.commit()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_directories()
    configure_logging(
        settings.log_level,
        settings.log_dir,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    # SQLite's backup API remains consistent even when a WAL file exists.
    backup_sqlite_database(
        DATABASE_URL,
        settings.backup_dir,
        keep=settings.database_backup_count,
    )
    upgrade_database()
    live_event_hub.bind_loop(asyncio.get_running_loop())
    await asyncio.to_thread(_restore_active_runtime)
    logger.info(
        "Moto Laps started",
        extra={"version": __version__, "bind_host": settings.host, "port": settings.port},
    )
    try:
        yield
    finally:
        await asyncio.to_thread(_pause_running_race_on_shutdown)
        video_runtime.stop(stop_vision=False)
        vision_runtime.stop()
        race_recording.stop()
        camera_runtime.stop()
        live_event_hub.unbind_loop()
        engine.dispose()
        logger.info("Moto Laps stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Moto Laps API",
        description="Minimal local motorcycle lap timing API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    application.include_router(router)
    application.include_router(system_router)
    application.add_exception_handler(ApiError, api_error_handler)  # type: ignore[arg-type]
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "Request validation failed",
            extra={"method": request.method, "path": request.url.path},
        )
        issues = []
        for issue in exc.errors():
            serialized = dict(issue)
            if "ctx" in serialized:
                serialized["ctx"] = {key: str(value) for key, value in serialized["ctx"].items()}
            issues.append(serialized)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": {"issues": issues},
                }
            },
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Keep implementation details out of responses while retaining diagnostics."""

        logger.exception(
            "Unhandled request failure",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "method": request.method,
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Внутренняя ошибка сервера. Подробности записаны в журнал.",
                    "details": {},
                }
            },
        )

    @application.middleware("http")
    async def local_security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        content_length = request.headers.get("content-length")
        response: Response
        request_limit = (
            settings.video_upload_max_bytes + 2_000_000
            if request.url.path == "/api/videos"
            else 1_000_000
        )
        if content_length and content_length.isdigit() and int(content_length) > request_limit:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": "Размер запроса превышает допустимый предел.",
                        "details": {},
                    }
                },
            )
        else:
            response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self)"
        response.headers["X-Request-ID"] = request_id
        # Successful live-preview pulls occur once per camera frame. Logging
        # each one would create tens of records per second and add avoidable
        # disk latency; failures remain fully logged.
        high_frequency_preview = (
            request.method == "GET"
            and request.url.path == "/api/camera/snapshot"
            and response.status_code < 400
        )
        if not high_frequency_preview:
            logger.info(
                "HTTP request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client": request.client.host if request.client else None,
                },
            )
        return response

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"app_name": settings.app_name, "app_version": __version__},
        )

    @application.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @application.websocket("/ws/live")
    async def websocket_live(websocket: WebSocket) -> None:
        await live_event_hub.connect(websocket)
        try:
            while True:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected")
        except Exception:
            logger.debug("WebSocket client disconnected with an error", exc_info=True)
        finally:
            await live_event_hub.disconnect(websocket)

    return application


app = create_app()
