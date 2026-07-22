"""Cross-platform command line entry point."""

from __future__ import annotations

import argparse
import ipaddress

import uvicorn

from app.config import get_settings
from app.services.process_control import register_shutdown_callback
from app.single_instance import AlreadyRunningError, SingleInstanceLock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Moto Laps local server")
    parser.add_argument("--host", help="Bind address; defaults to local-only 127.0.0.1")
    parser.add_argument("--port", type=int, help="HTTP port")
    parser.add_argument("--reload", action="store_true", help="Enable development reload")
    return parser


def main() -> None:
    settings = get_settings()
    args = build_parser().parse_args()
    host = args.host or settings.host
    port = args.port or settings.port
    try:
        local_bind = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local_bind = False
    if not local_bind:
        raise SystemExit(
            "Remote binding is disabled: Moto Laps has no network authentication. "
            "Use 127.0.0.1 or localhost."
        )
    try:
        with SingleInstanceLock(f"moto-laps-{host}-{port}"):
            if args.reload:
                uvicorn.run("app.main:app", host=host, port=port, reload=True)
                return
            config = uvicorn.Config("app.main:app", host=host, port=port)
            server = uvicorn.Server(config)
            register_shutdown_callback(lambda: setattr(server, "should_exit", True))
            try:
                server.run()
            finally:
                register_shutdown_callback(None)
    except AlreadyRunningError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
