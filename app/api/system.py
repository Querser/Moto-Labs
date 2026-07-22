"""Local process lifecycle endpoint for the desktop-style web interface."""

from __future__ import annotations

import logging
import os
import signal
import time

from fastapi import APIRouter, BackgroundTasks

from app.services.camera_runtime import camera_runtime
from app.services.live_events import live_event_hub
from app.services.process_control import request_server_shutdown

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["system"])


def _interrupt_process_after_response() -> None:
    """Let the response leave the socket, then ask Uvicorn to shut down."""

    time.sleep(0.35)
    close_future = live_event_hub.close_all_threadsafe()
    if close_future is not None:
        try:
            close_future.result(timeout=2.0)
        except Exception:
            logger.warning("Live browser connections did not close promptly", exc_info=True)
    # End MJPEG generators before Uvicorn waits for open HTTP streams.
    camera_runtime.stop()
    if not request_server_shutdown():
        # Fallback for third-party ASGI runners that did not register a callback.
        os.kill(os.getpid(), signal.SIGINT)


@router.post("/shutdown")
def shutdown(background: BackgroundTasks) -> dict[str, str]:
    """Gracefully stop this local Moto Laps process."""

    logger.warning("Application shutdown requested from the web interface")
    background.add_task(_interrupt_process_after_response)
    return {"status": "shutting_down"}
