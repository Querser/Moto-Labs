"""In-process WebSocket event fan-out for a local single-server deployment."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from concurrent.futures import Future
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class LiveEventHub:
    """Manage live clients without coupling domain services to FastAPI routes."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the server loop for safe publication from CV threads."""

        self._event_loop = loop

    def unbind_loop(self) -> None:
        self._event_loop = None

    def publish_threadsafe(
        self, event_type: str, data: Mapping[str, Any] | None = None
    ) -> Future[None] | None:
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return None
        return asyncio.run_coroutine_threadsafe(self.publish(event_type, data), loop)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        await websocket.send_json({"type": "connected", "websocket_clients": self.client_count})

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    def close_all_threadsafe(self) -> Future[None] | None:
        """Close browser sockets so Uvicorn shutdown cannot wait on live tabs."""

        loop = self._event_loop
        if loop is None or loop.is_closed():
            return None
        return asyncio.run_coroutine_threadsafe(self.close_all(), loop)

    async def close_all(self) -> None:
        async with self._lock:
            clients = tuple(self._clients)
            self._clients.clear()
        if clients:
            await asyncio.gather(
                *(client.close(code=1001) for client in clients),
                return_exceptions=True,
            )

    async def publish(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        message = {"type": event_type, "data": dict(data or {})}
        async with self._lock:
            clients = tuple(self._clients)
        if not clients:
            return
        results = await asyncio.gather(
            *(client.send_json(message) for client in clients),
            return_exceptions=True,
        )
        stale = [
            client
            for client, result in zip(clients, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if stale:
            async with self._lock:
                for client in stale:
                    self._clients.discard(client)
            logger.debug("Removed disconnected WebSocket clients", extra={"count": len(stale)})


live_event_hub = LiveEventHub()
