"""Thread-safe bridge from the local API to the owning Uvicorn server."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock

_lock = Lock()
_shutdown_callback: Callable[[], None] | None = None


def register_shutdown_callback(callback: Callable[[], None] | None) -> None:
    """Register the callback owned by the current CLI server process."""

    global _shutdown_callback
    with _lock:
        _shutdown_callback = callback


def request_server_shutdown() -> bool:
    """Request graceful shutdown; return false outside the normal CLI runner."""

    with _lock:
        callback = _shutdown_callback
    if callback is None:
        return False
    callback()
    return True
