"""Replaceable detector, tracker, number-region, and OCR contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.camera import Frame

from .types import CandidateRegion, Detection, OcrPrediction, Track


class VisionError(RuntimeError):
    pass


class ModelLoadError(VisionError):
    pass


class InferenceError(VisionError):
    pass


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> Sequence[Detection]: ...

    def close(self) -> None:
        return None


class ObjectTracker(ABC):
    @abstractmethod
    def update(
        self, detections: Sequence[Detection], *, captured_monotonic_ns: int
    ) -> Sequence[Track]: ...

    @abstractmethod
    def reset(self) -> None: ...


class NumberRegionExtractor(ABC):
    @abstractmethod
    def extract(self, frame: Frame, track: Track) -> Sequence[CandidateRegion]: ...


class OcrEngine(ABC):
    @abstractmethod
    def recognize(
        self, region: CandidateRegion, *, frame: Frame, track: Track
    ) -> Sequence[OcrPrediction]: ...

    def close(self) -> None:
        return None
