"""Image transformations that preserve frame capture timestamps."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .base import CameraConfig, Frame, Rotation


def _opencv_transform(image: Any, config: CameraConfig) -> Any | None:
    if not hasattr(image, "shape"):
        return None
    try:
        import cv2
    except ImportError:
        return None

    result = image
    if config.rotation is Rotation.CLOCKWISE_90:
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    elif config.rotation is Rotation.UPSIDE_DOWN:
        result = cv2.rotate(result, cv2.ROTATE_180)
    elif config.rotation is Rotation.COUNTERCLOCKWISE_90:
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if config.mirror_horizontal and config.mirror_vertical:
        result = cv2.flip(result, -1)
    elif config.mirror_horizontal:
        result = cv2.flip(result, 1)
    elif config.mirror_vertical:
        result = cv2.flip(result, 0)
    return result


def _matrix_transform(image: Any, config: CameraConfig) -> Any:
    """Small dependency-free fallback used by mock sources and unit tests."""

    if isinstance(image, (bytes, bytearray, memoryview, str)):
        return image
    try:
        rows = [list(row) for row in image]
    except (TypeError, ValueError):
        return image
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Frame matrix rows must have equal length")

    rotation = config.rotation
    if rotation is Rotation.CLOCKWISE_90:
        rows = [list(row) for row in zip(*rows[::-1], strict=True)] if rows else []
    elif rotation is Rotation.UPSIDE_DOWN:
        rows = [row[::-1] for row in rows[::-1]]
    elif rotation is Rotation.COUNTERCLOCKWISE_90:
        rows = [list(row) for row in zip(*rows, strict=True)][::-1] if rows else []
    if config.mirror_horizontal:
        rows = [row[::-1] for row in rows]
    if config.mirror_vertical:
        rows = rows[::-1]
    return rows


def transform_image(image: Any, config: CameraConfig) -> Any:
    """Apply configured rotation and mirroring to an image."""

    transformed = _opencv_transform(image, config)
    return transformed if transformed is not None else _matrix_transform(image, config)


def transform_frame(frame: Frame, config: CameraConfig) -> Frame:
    """Transform pixels without changing either capture clock."""

    image = transform_image(frame.image, config)
    metadata = dict(frame.metadata)
    metadata["rotation"] = int(config.rotation)
    metadata["mirror_horizontal"] = config.mirror_horizontal
    metadata["mirror_vertical"] = config.mirror_vertical
    if hasattr(image, "shape") and len(image.shape) >= 2:
        metadata["frame_size"] = (int(image.shape[1]), int(image.shape[0]))
    elif isinstance(image, list) and image:
        metadata["frame_size"] = (len(image[0]), len(image))
    return replace(frame, image=image, metadata=metadata)
