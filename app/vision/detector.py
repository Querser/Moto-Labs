"""Synthetic adapters, motion candidates, and the semantic motorcycle detector."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.camera import Frame

from .hardware import preferred_ort_providers, prepare_onnx_runtime
from .interfaces import InferenceError, ModelLoadError, ObjectDetector
from .types import BoundingBox, Detection

logger = logging.getLogger(__name__)


def _parse_bbox(value: Any) -> BoundingBox:
    if isinstance(value, BoundingBox):
        return value
    if isinstance(value, Mapping):
        return BoundingBox(
            float(value["x1"]),
            float(value["y1"]),
            float(value["x2"]),
            float(value["y2"]),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return BoundingBox(*(float(item) for item in value))
    raise ValueError("Detection bbox must be BoundingBox, mapping, or four coordinates")


class MetadataObjectDetector(ObjectDetector):
    """Read explicit detections embedded by a synthetic/mock source.

    This adapter is intentionally demo-only. It never invents detections from a
    real image and therefore cannot mask a failed production model load.
    """

    def __init__(self, metadata_key: str = "detections") -> None:
        self.metadata_key = metadata_key

    def detect(self, frame: Frame) -> list[Detection]:
        raw_detections = frame.metadata.get(self.metadata_key, ())
        detections: list[Detection] = []
        for raw in raw_detections:
            if isinstance(raw, Detection):
                detections.append(raw)
                continue
            if not isinstance(raw, Mapping):
                raise InferenceError("Synthetic detection must be a mapping or Detection")
            reserved = {"bbox", "confidence", "label", "metadata"}
            metadata = dict(raw.get("metadata", {}))
            metadata.update({key: value for key, value in raw.items() if key not in reserved})
            try:
                detections.append(
                    Detection(
                        bbox=_parse_bbox(raw["bbox"]),
                        confidence=float(raw.get("confidence", 1.0)),
                        label=str(raw.get("label", "motorcycle_candidate")),
                        metadata=metadata,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise InferenceError(f"Invalid synthetic detection: {exc}") from exc
        return detections


class OpenCVMotionDetector(ObjectDetector):
    """Portable CPU motion baseline for a fixed finish-line camera.

    It detects moving foreground regions, not semantic motorcycle classes. It
    is useful as a local baseline and dataset collector; a trained detector can
    replace it through :class:`ObjectDetector` for production accuracy.
    """

    def __init__(
        self,
        *,
        minimum_area: float = 1_500,
        maximum_area_ratio: float = 0.9,
        history: int = 250,
        variance_threshold: float = 32,
        learning_rate: float = -1,
        confidence: float = 0.55,
    ) -> None:
        if minimum_area <= 0:
            raise ValueError("minimum_area must be positive")
        if not 0 < maximum_area_ratio <= 1:
            raise ValueError("maximum_area_ratio must be in (0, 1]")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        try:
            import cv2
        except ImportError as exc:
            raise ModelLoadError(
                "OpenCV motion detector is unavailable; install opencv-python-headless"
            ) from exc
        try:
            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=variance_threshold,
                detectShadows=False,
            )
        except Exception as exc:
            raise ModelLoadError(f"Could not initialize OpenCV motion detector: {exc}") from exc
        self._cv2 = cv2
        self._subtractor = subtractor
        self.minimum_area = minimum_area
        self.maximum_area_ratio = maximum_area_ratio
        self.learning_rate = learning_rate
        self.confidence = confidence

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.image
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            raise InferenceError("OpenCV motion detector requires an image array")
        try:
            mask = self._subtractor.apply(image, learningRate=self.learning_rate)
            kernel = self._cv2.getStructuringElement(self._cv2.MORPH_ELLIPSE, (5, 5))
            mask = self._cv2.morphologyEx(mask, self._cv2.MORPH_OPEN, kernel)
            mask = self._cv2.morphologyEx(mask, self._cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = self._cv2.findContours(
                mask, self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
            )
        except Exception as exc:
            raise InferenceError(f"Motion detection failed: {exc}") from exc
        height, width = int(image.shape[0]), int(image.shape[1])
        maximum_area = width * height * self.maximum_area_ratio
        detections: list[Detection] = []
        for contour in contours:
            area = float(self._cv2.contourArea(contour))
            if not self.minimum_area <= area <= maximum_area:
                continue
            x, y, box_width, box_height = self._cv2.boundingRect(contour)
            detections.append(
                Detection(
                    bbox=BoundingBox(x, y, x + box_width, y + box_height),
                    confidence=self.confidence,
                    label="moving_motorcycle_candidate",
                    metadata={"motion_area": area, "baseline": "opencv_mog2"},
                )
            )
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


BaselineObjectDetector = OpenCVMotionDetector


class YoloXMotorcycleDetector(ObjectDetector):
    """Run the official COCO YOLOX-Tiny ONNX model and keep class ``motorcycle``.

    The official YOLOX export has raw 416x416 predictions. Post-processing here
    follows the project's ONNX Runtime example and deliberately never treats a
    motion contour or another COCO vehicle class as a motorcycle.
    """

    MODEL_NAME = "YOLOX-Tiny"
    MODEL_VERSION = "0.1.1rc0"
    MOTORCYCLE_CLASS_ID = 3

    def __init__(
        self,
        model_path: str | Path = "models/yolox_tiny.onnx",
        *,
        confidence_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        intra_op_threads: int = 4,
    ) -> None:
        if not 0 < confidence_threshold < 1 or not 0 < nms_threshold < 1:
            raise ValueError("YOLOX confidence/NMS thresholds must be in (0, 1)")
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise ModelLoadError(f"YOLOX weights are missing: {path}")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ModelLoadError("YOLOX requires OpenCV, NumPy, and ONNX Runtime") from exc
        try:
            ort, _profile = prepare_onnx_runtime()
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, intra_op_threads)
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                str(path),
                sess_options=options,
                providers=list(preferred_ort_providers()),
            )
        except Exception as exc:
            raise ModelLoadError(f"Could not load YOLOX model: {exc}") from exc
        input_info = self._session.get_inputs()[0]
        if len(input_info.shape) != 4 or not all(
            isinstance(value, int) for value in input_info.shape[-2:]
        ):
            raise ModelLoadError("YOLOX model input shape is unsupported")
        self._cv2 = cv2
        self._np = np
        self._input_name = input_info.name
        self._input_size = (int(input_info.shape[-2]), int(input_info.shape[-1]))
        self._grids, self._expanded_strides = self._build_grids(self._input_size)
        self.model_path = path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.providers = tuple(self._session.get_providers())
        self._warm_up()

    def _build_grids(self, input_size: tuple[int, int]) -> tuple[Any, Any]:
        grids, strides = [], []
        for stride in (8, 16, 32):
            height, width = input_size[0] // stride, input_size[1] // stride
            grid_y, grid_x = self._np.meshgrid(
                self._np.arange(height), self._np.arange(width), indexing="ij"
            )
            grid = self._np.stack((grid_x, grid_y), axis=2).reshape(1, -1, 2)
            grids.append(grid)
            strides.append(self._np.full((*grid.shape[:2], 1), stride))
        return (
            self._np.concatenate(grids, axis=1).astype(self._np.float32),
            self._np.concatenate(strides, axis=1).astype(self._np.float32),
        )

    def _warm_up(self) -> None:
        try:
            empty = self._np.zeros((1, 3, *self._input_size), dtype=self._np.float32)
            self._session.run(None, {self._input_name: empty})
        except Exception as exc:
            current = tuple(self._session.get_providers())
            if current and current[0] != "CPUExecutionProvider":
                logger.warning(
                    "YOLOX accelerator warm-up failed; retrying on CPU",
                    exc_info=True,
                )
                try:
                    self._session.set_providers(["CPUExecutionProvider"])
                    self._session.run(None, {self._input_name: empty})
                    self.providers = tuple(self._session.get_providers())
                    return
                except Exception as cpu_exc:
                    raise ModelLoadError(
                        f"YOLOX warm-up failed on accelerator and CPU: {cpu_exc}"
                    ) from cpu_exc
            raise ModelLoadError(f"YOLOX warm-up failed: {exc}") from exc

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.image
        if image is None or not hasattr(image, "shape") or len(image.shape) != 3:
            raise InferenceError("YOLOX requires a BGR image array")
        frame_height, frame_width = int(image.shape[0]), int(image.shape[1])
        ratio = min(
            self._input_size[0] / frame_height,
            self._input_size[1] / frame_width,
        )
        resized_width, resized_height = int(frame_width * ratio), int(frame_height * ratio)
        padded = self._np.full((*self._input_size, 3), 114, dtype=self._np.uint8)
        resized = self._cv2.resize(
            image, (resized_width, resized_height), interpolation=self._cv2.INTER_LINEAR
        )
        padded[:resized_height, :resized_width] = resized
        tensor = self._np.ascontiguousarray(
            padded.transpose(2, 0, 1)[None], dtype=self._np.float32
        )
        try:
            raw = self._session.run(None, {self._input_name: tensor})[0]
        except Exception as exc:
            raise InferenceError(f"YOLOX inference failed: {exc}") from exc
        predictions = raw.copy()
        predictions[..., :2] = (
            predictions[..., :2] + self._grids
        ) * self._expanded_strides
        predictions[..., 2:4] = (
            self._np.exp(predictions[..., 2:4]) * self._expanded_strides
        )
        values = predictions[0]
        scores = values[:, 4] * values[:, 5 + self.MOTORCYCLE_CLASS_ID]
        selected = scores >= self.confidence_threshold
        if not self._np.any(selected):
            return []
        boxes = values[selected, :4]
        scores = scores[selected]
        xyxy = self._np.empty_like(boxes)
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) / ratio
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) / ratio
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) / ratio
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) / ratio
        xyxy[:, 0::2] = self._np.clip(xyxy[:, 0::2], 0, frame_width)
        xyxy[:, 1::2] = self._np.clip(xyxy[:, 1::2], 0, frame_height)
        nms_boxes = [
            [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
            for box in xyxy
        ]
        indexes = self._cv2.dnn.NMSBoxes(
            nms_boxes,
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )
        detections: list[Detection] = []
        for raw_index in indexes:
            index = int(raw_index)
            box = xyxy[index]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            detections.append(
                Detection(
                    bbox=BoundingBox(*(float(value) for value in box)),
                    confidence=float(scores[index]),
                    label="motorcycle",
                    metadata={
                        "class_id": self.MOTORCYCLE_CLASS_ID,
                        "model": self.MODEL_NAME,
                    },
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)


class MotionAwareMotorcycleDetector(ObjectDetector):
    """Use foreground motion only to decide when to run the semantic model."""

    def __init__(
        self,
        semantic_detector: ObjectDetector,
        *,
        motion_detector: OpenCVMotionDetector | None = None,
        full_frame_interval: int = 10,
    ) -> None:
        if full_frame_interval < 1:
            raise ValueError("full_frame_interval must be positive")
        self.semantic_detector = semantic_detector
        self.motion_detector = motion_detector or OpenCVMotionDetector(minimum_area=900)
        self.full_frame_interval = full_frame_interval
        self._frame_count = 0
        self.last_motion_candidates: tuple[Detection, ...] = ()

    def detect(self, frame: Frame) -> list[Detection]:
        self._frame_count += 1
        self.last_motion_candidates = tuple(self.motion_detector.detect(frame))
        should_scan = bool(self.last_motion_candidates) or self._frame_count == 1
        should_scan = should_scan or self._frame_count % self.full_frame_interval == 0
        return list(self.semantic_detector.detect(frame)) if should_scan else []

    def close(self) -> None:
        self.semantic_detector.close()
