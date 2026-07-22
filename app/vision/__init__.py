"""Public local motorcycle recognition and lap-event API."""

from .detector import (
    MetadataObjectDetector,
    MotionAwareMotorcycleDetector,
    OpenCVMotionDetector,
    YoloXMotorcycleDetector,
)
from .interfaces import (
    InferenceError,
    ModelLoadError,
    NumberRegionExtractor,
    ObjectDetector,
    ObjectTracker,
    OcrEngine,
    VisionError,
)
from .line_crossing import CrossingEvent, FinishLine, LineCrossingDetector, leading_point
from .ocr import (
    HybridDigitOcrEngine,
    MetadataOcrEngine,
    OcrAggregationConfig,
    OcrAggregator,
    OpenCVDigitOcrEngine,
    PaddleOcrV6DigitEngine,
    RapidOcrDigitEngine,
    normalize_racing_number,
)
from .pipeline import (
    MotorcycleVisionPipeline,
    ParticipantPassageGuard,
    StablePassage,
    VisionStageRegion,
)
from .regions import BoundingBoxNumberRegionExtractor, FrontNumberBoardRegionExtractor
from .tracker import CentroidIoUTracker, SupervisionByteTrack
from .types import (
    BoundingBox,
    CandidateRegion,
    Detection,
    OcrPrediction,
    OcrResolution,
    OcrResolutionStatus,
    Track,
    TrajectoryPoint,
)
from .verifier import Florence2NumberVerifier, NumberVerification, NumberVerifier

__all__ = [
    "BoundingBox",
    "BoundingBoxNumberRegionExtractor",
    "CandidateRegion",
    "CentroidIoUTracker",
    "CrossingEvent",
    "Detection",
    "FinishLine",
    "Florence2NumberVerifier",
    "FrontNumberBoardRegionExtractor",
    "HybridDigitOcrEngine",
    "InferenceError",
    "LineCrossingDetector",
    "MetadataObjectDetector",
    "MetadataOcrEngine",
    "ModelLoadError",
    "MotionAwareMotorcycleDetector",
    "MotorcycleVisionPipeline",
    "NumberRegionExtractor",
    "NumberVerification",
    "NumberVerifier",
    "ObjectDetector",
    "ObjectTracker",
    "OcrAggregationConfig",
    "OcrAggregator",
    "OcrEngine",
    "OcrPrediction",
    "OcrResolution",
    "OcrResolutionStatus",
    "OpenCVDigitOcrEngine",
    "OpenCVMotionDetector",
    "PaddleOcrV6DigitEngine",
    "ParticipantPassageGuard",
    "RapidOcrDigitEngine",
    "StablePassage",
    "SupervisionByteTrack",
    "Track",
    "TrajectoryPoint",
    "VisionError",
    "VisionStageRegion",
    "YoloXMotorcycleDetector",
    "leading_point",
    "normalize_racing_number",
]
