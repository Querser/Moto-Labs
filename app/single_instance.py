"""Cross-process guard preventing two local servers from owning one camera."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType


class AlreadyRunningError(RuntimeError):
    pass


class SingleInstanceLock:
    """Hold a Windows named mutex or a POSIX file lock for the server lifetime."""

    def __init__(self, name: str) -> None:
        self.name = "".join(character if character.isalnum() else "-" for character in name)
        self._handle: int | None = None
        self._file: object | None = None

    def __enter__(self) -> SingleInstanceLock:
        if os.name == "nt":
            self._acquire_windows()
        else:
            self._acquire_posix()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.release()

    def _acquire_windows(self) -> None:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_mutex = kernel32.CreateMutexW
        create_mutex.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
        create_mutex.restype = ctypes.c_void_p
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_bool
        handle = create_mutex(None, False, f"Local\\{self.name}")
        if not handle:
            raise OSError(ctypes.get_last_error(), "Could not create application mutex")
        if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            close_handle(handle)
            raise AlreadyRunningError("Moto Laps is already running")
        self._handle = int(handle)

    def _acquire_posix(self) -> None:
        import fcntl

        path = Path("data") / f".{self.name}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        file = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(  # type: ignore[attr-defined]
                file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB  # type: ignore[attr-defined]
            )
        except BlockingIOError as exc:
            file.close()
            raise AlreadyRunningError("Moto Laps is already running") from exc
        self._file = file

    def release(self) -> None:
        if self._handle is not None:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._handle)
            self._handle = None
        if self._file is not None:
            self._file.close()  # type: ignore[attr-defined]
            self._file = None
