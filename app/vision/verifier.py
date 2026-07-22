"""Bounded local VLM verification for ambiguous offline OCR results."""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Protocol

logger = logging.getLogger(__name__)
_PROTOCOL_PREFIX = "MOTO_LAPS_JSON:"


@dataclass(frozen=True, slots=True)
class NumberVerification:
    racing_number: str
    confidence: float
    raw_text: str


class NumberVerifier(Protocol):
    def verify(
        self,
        image: Any,
        *,
        candidates: Sequence[str],
    ) -> NumberVerification | None: ...

    def close(self) -> None: ...


class Florence2NumberVerifier:
    """Use Florence-2 only to choose among OCR-supported candidates.

    On Windows the verifier uses a dedicated Python environment and process.
    PyTorch CUDA 13 and ONNX Runtime CUDA 12 therefore never load their cuDNN
    DLLs into the same address space. The verifier cannot invent a participant:
    its result must exactly match a conventional OCR candidate from this track.
    """

    MODEL_NAME = "microsoft/Florence-2-base-ft"
    MODEL_VERSION = "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
    TASK_PROMPT = "<OCR>"

    def __init__(
        self,
        model_dir: str | Path = "models/florence-2-base-ft",
        *,
        worker_python: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.worker_script = (
            Path(__file__).resolve().parents[2] / "scripts" / "florence_worker.py"
        )
        self.worker_python = self._resolve_worker_python(worker_python)
        self._lock = RLock()
        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader: Thread | None = None
        self._device = "unloaded"
        self._last_raw_text = ""

    @property
    def available(self) -> bool:
        return (
            (self.model_dir / "config.json").is_file()
            and self.worker_script.is_file()
            and self.worker_python.is_file()
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def last_raw_text(self) -> str:
        return self._last_raw_text

    @staticmethod
    def _resolve_worker_python(value: str | Path | None) -> Path:
        if value is not None:
            return Path(value).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[2]
        if os.name == "nt":
            isolated = project_root / ".venv-florence" / "Scripts" / "python.exe"
            if isolated.is_file():
                return isolated
        return Path(sys.executable).resolve()

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            if not line.startswith(_PROTOCOL_PREFIX):
                continue
            try:
                payload = json.loads(line.removeprefix(_PROTOCOL_PREFIX))
                if isinstance(payload, dict):
                    self._messages.put(payload)
            except json.JSONDecodeError:
                logger.debug("Ignored malformed Florence worker response")

    def _receive(self, timeout: float) -> dict[str, Any]:
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Florence-2 worker response timed out") from exc

    def _load(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            if not self.available:
                raise RuntimeError(
                    "Florence-2 weights, worker, or isolated runtime are missing"
                )
            process = subprocess.Popen(
                [
                    str(self.worker_python),
                    str(self.worker_script),
                    "--model-dir",
                    str(self.model_dir),
                ],
                cwd=str(self.worker_script.parents[1]),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            self._messages = queue.Queue()
            reader = Thread(
                target=self._read_stdout,
                args=(process,),
                name="florence-2-protocol-reader",
                daemon=True,
            )
            reader.start()
            try:
                response = self._receive(120.0)
            except Exception:
                process.terminate()
                process.wait(timeout=5.0)
                raise
            if response.get("status") != "ready":
                process.terminate()
                process.wait(timeout=5.0)
                raise RuntimeError(
                    f"Florence-2 worker failed: {response.get('error', 'unknown error')}"
                )
            self._process = process
            self._reader = reader
            self._device = str(response.get("device", "unknown"))

    def prepare(self) -> None:
        """Load weights before offline processing reaches its first conflict."""

        self._load()

    def verify(
        self,
        image: Any,
        *,
        candidates: Sequence[str],
    ) -> NumberVerification | None:
        allowed = tuple(
            dict.fromkeys(
                value
                for value in candidates
                if value and value.isascii() and value.isdigit() and len(value) <= 4
            )
        )
        if len(allowed) < 2 or image is None or getattr(image, "size", 0) == 0:
            return None
        try:
            import cv2

            self._load()
            assert self._process is not None
            assert self._process.stdin is not None
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
            if not ok:
                return None
            request = {
                "command": "verify",
                "image_jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
            with self._lock:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                response = self._receive(30.0)
            if response.get("status") != "result":
                raise RuntimeError(str(response.get("error", "unknown worker error")))
            raw = str(response.get("raw_text", ""))
            self._last_raw_text = raw
        except Exception:
            logger.warning("Florence-2 ambiguity verification failed", exc_info=True)
            return None
        tokens = re.findall(r"(?<!\d)\d{1,4}(?!\d)", raw)
        matches = [token for token in tokens if token in allowed]
        if len(set(matches)) != 1:
            return None
        return NumberVerification(matches[0], 0.70, raw)

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            self._device = "closed"
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "close"}) + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5.0)


__all__ = [
    "Florence2NumberVerifier",
    "NumberVerification",
    "NumberVerifier",
]
