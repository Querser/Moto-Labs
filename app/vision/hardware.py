"""Cross-platform hardware discovery and ONNX Runtime preparation."""

from __future__ import annotations

import logging
import os
import platform
import site
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ``os.add_dll_directory`` handles must stay alive while ONNX Runtime is in use.
_DLL_DIRECTORY_HANDLES: list[Any] = []


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Resolved local execution capabilities without exposing UI settings."""

    system: str
    machine: str
    cpu_count: int
    apple_silicon: bool
    macos_intel_supported: bool
    available_ort_providers: tuple[str, ...]
    preferred_ort_provider: str
    accelerator: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nvidia_bin_directories() -> tuple[Path, ...]:
    """Find CUDA/cuDNN DLLs installed by ONNX Runtime's official extras."""

    result: list[Path] = []
    for root in (Path(value) for value in site.getsitepackages()):
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        result.extend(path for path in nvidia_root.glob("*/bin") if path.is_dir())
    return tuple(dict.fromkeys(path.resolve() for path in result))


def _prepare_windows_cuda_dlls(ort: Any) -> None:
    """Make package-local CUDA sublibraries visible to dependent cuDNN DLLs."""

    if platform.system() != "Windows":
        return
    directories = _nvidia_bin_directories()
    if directories:
        existing_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in directories), existing_path]
        )
        add_directory = getattr(os, "add_dll_directory", None)
        if callable(add_directory):
            for directory in directories:
                try:
                    _DLL_DIRECTORY_HANDLES.append(add_directory(str(directory)))
                except OSError:
                    logger.debug("Could not register CUDA DLL directory %s", directory)
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            # Empty directory asks ORT to search the NVIDIA Python packages.
            preload(directory="")
        except Exception:
            logger.warning("CUDA DLL preload failed; CPU fallback remains available")


@lru_cache(maxsize=1)
def prepare_onnx_runtime() -> tuple[Any, HardwareProfile]:
    """Import ONNX Runtime, prepare native libraries, and resolve a profile."""

    import onnxruntime as ort  # type: ignore[import-untyped]

    _prepare_windows_cuda_dlls(ort)
    providers = tuple(ort.get_available_providers())
    system = platform.system()
    machine = platform.machine().lower()
    apple_silicon = system == "Darwin" and machine in {"arm64", "aarch64"}
    if system == "Windows" and "CUDAExecutionProvider" in providers:
        preferred = "CUDAExecutionProvider"
        accelerator = "nvidia_cuda"
    elif apple_silicon and "CoreMLExecutionProvider" in providers:
        preferred = "CoreMLExecutionProvider"
        accelerator = "apple_coreml"
    else:
        preferred = "CPUExecutionProvider"
        accelerator = "cpu"
    profile = HardwareProfile(
        system=system,
        machine=machine,
        cpu_count=max(1, os.cpu_count() or 1),
        apple_silicon=apple_silicon,
        # The requested macOS target is Apple Silicon only. Intel macOS keeps
        # a best-effort CPU fallback but is deliberately not a supported target.
        macos_intel_supported=system != "Darwin" or apple_silicon,
        available_ort_providers=providers,
        preferred_ort_provider=preferred,
        accelerator=accelerator,
    )
    return ort, profile


def preferred_ort_providers() -> tuple[Any, ...]:
    """Return ordered heterogeneous providers with a safe CPU fallback."""

    _ort, profile = prepare_onnx_runtime()
    if profile.preferred_ort_provider == "CUDAExecutionProvider":
        return (
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": 1,
                    "use_tf32": 1,
                },
            ),
            "CPUExecutionProvider",
        )
    if profile.preferred_ort_provider == "CoreMLExecutionProvider":
        return (
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "0",
                },
            ),
            "CPUExecutionProvider",
        )
    return ("CPUExecutionProvider",)


def hardware_profile() -> HardwareProfile:
    """Return the cached execution profile."""

    return prepare_onnx_runtime()[1]


__all__ = [
    "HardwareProfile",
    "hardware_profile",
    "preferred_ort_providers",
    "prepare_onnx_runtime",
]
