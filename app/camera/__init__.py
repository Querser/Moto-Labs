"""Public camera subsystem API."""

from .base import (
    CameraConfig,
    CameraError,
    CameraReadError,
    CameraSource,
    CameraUnavailableError,
    EndOfStream,
    Frame,
    Rotation,
)
from .buffer import FrameBufferMetrics, LatestFrameBuffer
from .discovery import CameraInfo, discover_cameras, resolve_camera
from .manager import CameraCaptureWorker, CameraManager, CameraMetrics, CameraState
from .sources import (
    MockCameraSource,
    MockSource,
    SyntheticCameraSource,
    SyntheticFrameSpec,
    VideoFileSource,
    VideoSource,
    WebcamCameraSource,
    WebcamSource,
)
from .transforms import transform_frame, transform_image

__all__ = [
    "CameraCaptureWorker",
    "CameraConfig",
    "CameraError",
    "CameraInfo",
    "CameraManager",
    "CameraMetrics",
    "CameraReadError",
    "CameraSource",
    "CameraState",
    "CameraUnavailableError",
    "EndOfStream",
    "Frame",
    "FrameBufferMetrics",
    "LatestFrameBuffer",
    "MockCameraSource",
    "MockSource",
    "Rotation",
    "SyntheticCameraSource",
    "SyntheticFrameSpec",
    "VideoFileSource",
    "VideoSource",
    "WebcamCameraSource",
    "WebcamSource",
    "discover_cameras",
    "resolve_camera",
    "transform_frame",
    "transform_image",
]
