from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.camera import Frame
from app.services.camera_runtime import _synthetic_frame
from app.vision import (
    BoundingBox,
    BoundingBoxNumberRegionExtractor,
    CentroidIoUTracker,
    Detection,
    FinishLine,
    FrontNumberBoardRegionExtractor,
    LineCrossingDetector,
    MetadataObjectDetector,
    MetadataOcrEngine,
    MotorcycleVisionPipeline,
    OcrAggregationConfig,
    OcrAggregator,
    OcrPrediction,
    OpenCVMotionDetector,
    ParticipantPassageGuard,
    RapidOcrDigitEngine,
    StablePassage,
    SupervisionByteTrack,
    Track,
    TrajectoryPoint,
    YoloXMotorcycleDetector,
    normalize_racing_number,
)

UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def camera_frame(
    sequence: int,
    detections: list[dict[str, object]],
    *,
    timestamp_ns: int | None = None,
    image: np.ndarray | None = None,
) -> Frame:
    return Frame(
        image=np.zeros((100, 100, 3), dtype=np.uint8) if image is None else image,
        sequence=sequence,
        source_id="test",
        captured_monotonic_ns=timestamp_ns or sequence * 100_000_000,
        captured_at_utc=UTC,
        metadata={"detections": detections},
    )


def metadata_detection(
    bbox: tuple[int, int, int, int], number: str, *, confidence: float = 0.99
) -> dict[str, object]:
    return {
        "bbox": bbox,
        "confidence": confidence,
        "label": "motorcycle",
        "racing_number": number,
        "ocr_confidence": 0.96,
    }


def synthetic_pipeline(line: FinishLine | None = None) -> MotorcycleVisionPipeline:
    return MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100, max_missed_frames=8),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=MetadataOcrEngine(),
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(
                instant_resolution_confidence=None,
                minimum_observations=2,
                minimum_consecutive=2,
            )
        ),
        finish_line=line or FinishLine(0.1, 0.5, 0.9, 0.5),
    )


def tracked(
    track_id: int,
    bbox: tuple[int, int, int, int],
    centers: list[tuple[float, float, int]],
) -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(*bbox),
        confidence=0.95,
        hits=len(centers),
        age=len(centers),
        missed_frames=0,
        observed=True,
        captured_monotonic_ns=centers[-1][2],
        trajectory=tuple(TrajectoryPoint(*value) for value in centers),
    )


def test_moving_object_candidate_generation() -> None:
    detector = OpenCVMotionDetector(minimum_area=80, maximum_area_ratio=0.95, history=20)
    first = camera_frame(1, [], image=np.zeros((120, 160, 3), dtype=np.uint8))
    detector.detect(first)
    moved = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(moved, (30, 30), (90, 95), (255, 255, 255), -1)
    candidates = detector.detect(camera_frame(2, [], image=moved))
    assert candidates
    assert all(item.label == "moving_motorcycle_candidate" for item in candidates)


def test_pipeline_exposes_separate_real_board_and_digit_regions() -> None:
    pipeline = synthetic_pipeline()
    detection = metadata_detection((20, 10, 80, 70), "007")
    pipeline.process(camera_frame(1, [detection]))
    result = pipeline.process(camera_frame(2, [detection]))
    assert len(result.board_regions) == 1
    assert result.board_regions[0].kind == "number_board"
    assert len(result.digit_regions) == 1
    assert result.digit_regions[0].text == "007"
    assert result.digit_regions[0].bbox.area < result.board_regions[0].bbox.area


def test_synthetic_object_never_emits_an_invalid_offscreen_bbox() -> None:
    for sequence in range(240):
        specification = _synthetic_frame(sequence)
        for detection in specification.metadata.get("detections", []):  # type: ignore[union-attr]
            x1, y1, x2, y2 = detection["bbox"]
            assert x2 > x1
            assert y2 > y1


def test_actual_yolox_filters_blank_and_detects_representative_motorcycle() -> None:
    detector = YoloXMotorcycleDetector(ROOT / "models" / "yolox_tiny.onnx")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(camera_frame(1, [], image=blank)) == []
    sample = cv2.imdecode(
        np.fromfile(ROOT / "tests" / "assets" / "motorcycle_cc_by_4.jpg", dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    detections = detector.detect(camera_frame(2, [], image=sample))
    assert detections
    assert all(item.label == "motorcycle" for item in detections)
    assert all(item.metadata["class_id"] == 3 for item in detections)


def test_bytetrack_continuity_through_short_detection_gap() -> None:
    tracker = SupervisionByteTrack(lost_track_buffer=12)
    detection = Detection(BoundingBox(10, 20, 40, 70), 0.95, "motorcycle")
    first = tracker.update([detection], captured_monotonic_ns=100)
    assert len(first) == 1
    track_id = first[0].track_id
    assert tracker.update([], captured_monotonic_ns=200) == []
    moved = Detection(BoundingBox(14, 22, 44, 72), 0.93, "motorcycle")
    returned = tracker.update([moved], captured_monotonic_ns=300)
    assert returned[0].track_id == track_id
    assert len(returned[0].trajectory) == 2


def test_pipeline_stitches_one_unambiguous_short_tracker_split() -> None:
    class SplitTracker:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, detections, *, captured_monotonic_ns):  # type: ignore[no-untyped-def]
            del detections
            self.calls += 1
            if self.calls == 1:
                return [
                    tracked(
                        1,
                        (50, 20, 90, 70),
                        [(70, 45, captured_monotonic_ns)],
                    )
                ]
            return [
                tracked(
                    5,
                    (34, 20, 74, 70),
                    [(54, 45, captured_monotonic_ns)],
                )
            ]

        def reset(self) -> None:
            self.calls = 0

    class EmptyOcr:
        def recognize(self, region, *, frame, track):  # type: ignore[no-untyped-def]
            del region, frame, track
            return []

        def close(self) -> None:
            return None

    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=SplitTracker(),  # type: ignore[arg-type]
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=EmptyOcr(),  # type: ignore[arg-type]
        ocr_aggregator=OcrAggregator(),
        finish_line=FinishLine(0.1, 0.9, 0.9, 0.9),
    )
    first = pipeline.process(camera_frame(1, []))
    second = pipeline.process(camera_frame(2, []))
    assert [item.track_id for item in first.tracks] == [1]
    assert [item.track_id for item in second.tracks] == [1]
    assert second.tracks[0].hits == 2


def test_ocr_consensus_prefers_three_supporting_frames_and_preserves_zeroes() -> None:
    aggregator = OcrAggregator(
        OcrAggregationConfig(
            resolution_confidence=0.60,
            instant_resolution_confidence=None,
            minimum_observations=2,
            minimum_consecutive=2,
        )
    )
    result = None
    for index, value in enumerate(("777", "777", "771", "777"), start=1):
        result = aggregator.observe(
            1,
            OcrPrediction(value, 0.95),
            captured_monotonic_ns=index * 100_000_000,
            frame_sequence=index,
        )
    assert result is not None and result.is_resolved
    assert result.racing_number == "777"
    assert normalize_racing_number("007") == "007"
    assert normalize_racing_number("7") == "7"


def test_ocr_accepts_one_exceptionally_clear_fast_pass_frame() -> None:
    aggregator = OcrAggregator(
        OcrAggregationConfig(
            instant_resolution_confidence=0.93,
            minimum_observations=2,
            minimum_consecutive=2,
        )
    )
    result = aggregator.observe(
        1,
        OcrPrediction("044", 0.94),
        captured_monotonic_ns=100_000_000,
        frame_sequence=3,
    )
    assert result.is_resolved
    assert result.racing_number == "044"


def test_single_digit_requires_stronger_temporal_evidence() -> None:
    aggregator = OcrAggregator(
        OcrAggregationConfig(
            instant_resolution_confidence=0.93,
            minimum_observations=2,
            minimum_consecutive=2,
        )
    )
    result = aggregator.observe(
        1,
        OcrPrediction("8", 0.98),
        captured_monotonic_ns=100_000_000,
        frame_sequence=1,
    )
    assert not result.is_resolved
    for index in (2, 3):
        result = aggregator.observe(
            1,
            OcrPrediction("8", 0.98),
            captured_monotonic_ns=index * 100_000_000,
            frame_sequence=index,
        )
    assert result.is_resolved
    assert result.racing_number == "8"


def test_front_board_crop_and_local_ocr_on_generated_digits() -> None:
    image = np.full((480, 640, 3), 35, dtype=np.uint8)
    cv2.rectangle(image, (170, 105), (470, 285), (245, 245, 245), -1)
    cv2.putText(
        image, "007", (205, 245), cv2.FONT_HERSHEY_SIMPLEX, 3.6, (5, 5, 5), 9, cv2.LINE_AA
    )
    frame = camera_frame(1, [], image=image)
    track = tracked(1, (80, 40, 560, 440), [(320, 240, 100)])
    regions = FrontNumberBoardRegionExtractor().extract(frame, track)
    assert regions
    engine = RapidOcrDigitEngine()
    values = [
        prediction.text
        for region in regions
        for prediction in engine.recognize(region, frame=frame, track=track)
    ]
    assert "007" in values


def test_front_board_extractor_keeps_directional_fallbacks() -> None:
    image = np.full((240, 320, 3), 30, dtype=np.uint8)
    frame = camera_frame(1, [], image=image)
    track = tracked(1, (40, 20, 280, 220), [(160, 120, 100)])
    regions = FrontNumberBoardRegionExtractor(maximum_candidates=1).extract(frame, track)
    kinds = {region.kind for region in regions}
    assert {
        "front_board_anchor_left",
        "front_board_anchor_center",
        "front_board_anchor_right",
        "front_board_fallback",
    } <= kinds


def test_front_board_extractor_uses_leading_side_during_side_pass() -> None:
    image = np.full((240, 400, 3), 30, dtype=np.uint8)
    frame = camera_frame(4, [], image=image)
    track = tracked(
        1,
        (40, 20, 360, 220),
        [(300, 120, 100), (270, 120, 200), (230, 120, 300)],
    )
    regions = FrontNumberBoardRegionExtractor(maximum_candidates=1).extract(frame, track)
    fallback = next(item for item in regions if item.kind == "front_board_fallback")
    assert fallback.bbox.x2 <= 40 + 320 * 0.75


def test_rapidocr_normalization_rejects_partial_mixed_tokens() -> None:
    assert RapidOcrDigitEngine._normalize_model_text("123") == "123"
    assert RapidOcrDigitEngine._normalize_model_text("O44") == "044"
    assert RapidOcrDigitEngine._normalize_model_text("A5") == "85"
    assert RapidOcrDigitEngine._normalize_model_text("X5") is None


def test_preprocessing_consensus_strengthens_matching_digit_readings() -> None:
    predictions = [
        OcrPrediction("85", 0.54, metadata={"preprocessing": "color"}),
        OcrPrediction("85", 0.55, metadata={"preprocessing": "clahe"}),
        OcrPrediction("35", 0.70, metadata={"preprocessing": "grayscale"}),
    ]
    merged = RapidOcrDigitEngine._merge_preprocessing_consensus(predictions)
    assert merged.text == "85"
    assert merged.confidence >= 0.94
    assert merged.metadata["preprocessing_consensus"] == 2


def test_line_crossing_horizontal_and_interpolates_capture_time() -> None:
    detector = LineCrossingDetector(FinishLine(0.1, 0.5, 0.9, 0.5))
    before = tracked(1, (40, 10, 60, 30), [(50, 15, 100), (50, 20, 200)])
    after = tracked(1, (40, 40, 60, 60), [(50, 20, 200), (50, 50, 300)])
    assert detector.update([before], (100, 100)) == ()
    events = detector.update([after], (100, 100))
    assert len(events) == 1
    assert events[0].captured_monotonic_ns == 267
    assert events[0].point == (50.0, 50.0)


def test_line_crossing_supports_vertical_orientation() -> None:
    detector = LineCrossingDetector(FinishLine(0.5, 0.1, 0.5, 0.9))
    before = tracked(2, (10, 40, 30, 60), [(15, 50, 100), (20, 50, 200)])
    after = tracked(2, (40, 40, 60, 60), [(20, 50, 200), (50, 50, 300)])
    detector.update([before], (100, 100))
    assert len(detector.update([after], (100, 100))) == 1


def test_line_jitter_for_fifty_frames_creates_no_crossing() -> None:
    detector = LineCrossingDetector(FinishLine(0.1, 0.5, 0.9, 0.5), hysteresis_px=4)
    for index in range(50):
        center_y = 47 if index % 2 else 53
        item = tracked(
            1,
            (20, center_y - 10, 40, center_y + 10),
            [(25 + index, center_y, index * 100), (30 + index, center_y, index * 100 + 1)],
        )
        assert detector.update([item], (100, 100)) == ()


def test_pipeline_one_record_for_many_frames_then_return_next_lap() -> None:
    pipeline = synthetic_pipeline()
    first_approach = camera_frame(1, [metadata_detection((20, 10, 40, 30), "007")])
    second_approach = camera_frame(2, [metadata_detection((20, 20, 40, 40), "007")])
    assert pipeline.process(first_approach).passages == ()
    assert pipeline.process(second_approach).passages == ()
    first = pipeline.process(camera_frame(3, [metadata_detection((20, 40, 40, 60), "007")]))
    assert [item.racing_number for item in first.passages] == ["007"]
    for sequence in range(4, 54):
        assert pipeline.process(
            camera_frame(sequence, [metadata_detection((20, 40, 40, 60), "007")])
        ).passages == ()
    for sequence in range(54, 68):
        assert pipeline.process(camera_frame(sequence, [])).passages == ()
    pipeline.process(camera_frame(68, [metadata_detection((20, 10, 40, 30), "007")]))
    pipeline.process(camera_frame(69, [metadata_detection((20, 20, 40, 40), "007")]))
    second = pipeline.process(camera_frame(70, [metadata_detection((20, 40, 40, 60), "007")]))
    assert [item.racing_number for item in second.passages] == ["007"]


def test_changed_tracker_id_does_not_duplicate_same_physical_passage() -> None:
    """A short tracker ID switch must not create a second lap for one rider."""

    guard = ParticipantPassageGuard(clear_frames=3, disappearance_frames=12)
    line_detector = LineCrossingDetector(FinishLine(0.1, 0.5, 0.9, 0.5))
    assert guard.accept("777")
    replacement_track = tracked(
        99,
        (40, 42, 60, 62),
        [(50, 45, 100), (50, 55, 200)],
    )
    for _ in range(5):
        guard.update(
            (replacement_track,),
            {99: ("777", 0.97)},
            line_detector,
            (100, 100),
        )
    assert not guard.accept("777")


def test_stable_front_number_is_not_replaced_by_later_rear_number() -> None:
    pipeline = synthetic_pipeline()
    pipeline.process(camera_frame(1, [metadata_detection((20, 5, 40, 25), "410")]))
    pipeline.process(camera_frame(2, [metadata_detection((20, 20, 40, 40), "410")]))
    result = pipeline.process(
        camera_frame(3, [metadata_detection((20, 40, 40, 60), "313")])
    )
    assert [item.racing_number for item in result.passages] == ["410"]


def test_crossing_consensus_upgrades_an_early_partial_instant_reading() -> None:
    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=MetadataOcrEngine(),
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(instant_resolution_confidence=0.93)
        ),
        finish_line=FinishLine(0.1, 0.5, 0.9, 0.5),
    )
    pipeline.process(camera_frame(1, [metadata_detection((20, 5, 50, 25), "15")]))
    pipeline.process(camera_frame(2, [metadata_detection((20, 15, 50, 35), "435")]))
    pipeline.process(camera_frame(3, [metadata_detection((20, 25, 50, 45), "435")]))
    result = pipeline.process(
        camera_frame(4, [metadata_detection((20, 45, 50, 65), "435")])
    )
    assert [item.racing_number for item in result.passages] == ["435"]


def test_crossing_recovers_number_from_short_best_frame_buffer() -> None:
    class SecondLookOcr:
        def __init__(self) -> None:
            self.calls: dict[int, int] = {}

        def recognize(self, region, *, frame, track):  # type: ignore[no-untyped-def]
            del region, track
            self.calls[frame.sequence] = self.calls.get(frame.sequence, 0) + 1
            if self.calls[frame.sequence] == 1:
                return []
            return [OcrPrediction("007", 0.99)]

        def close(self) -> None:
            return None

    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=SecondLookOcr(),  # type: ignore[arg-type]
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(instant_resolution_confidence=None)
        ),
        finish_line=FinishLine(0.1, 0.5, 0.9, 0.5),
    )
    plain = {"bbox": (20, 5, 60, 35), "confidence": 0.99, "label": "motorcycle"}
    pipeline.process(camera_frame(1, [plain]))
    crossed = {"bbox": (20, 45, 60, 75), "confidence": 0.99, "label": "motorcycle"}
    result = pipeline.process(camera_frame(2, [crossed]))
    assert [item.racing_number for item in result.passages] == ["007"]


def test_uploaded_track_without_crossing_does_not_run_exit_ocr() -> None:
    class CountingOcr:
        def __init__(self) -> None:
            self.calls = 0

        def recognize(self, region, *, frame, track):  # type: ignore[no-untyped-def]
            del region, frame, track
            self.calls += 1
            return [OcrPrediction("123", 0.99)]

        def close(self) -> None:
            return None

    ocr = CountingOcr()
    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100, max_missed_frames=2),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=ocr,  # type: ignore[arg-type]
        ocr_aggregator=OcrAggregator(OcrAggregationConfig()),
        finish_line=FinishLine(0.1, 0.8, 0.9, 0.8),
        recover_identities_on_exit=False,
    )
    plain = {"bbox": (20, 5, 60, 35), "confidence": 0.99, "label": "motorcycle"}
    for sequence in range(1, 5):
        frame = camera_frame(sequence, [plain])
        pipeline.process(
            Frame(
                image=frame.image,
                sequence=frame.sequence,
                source_id=frame.source_id,
                captured_monotonic_ns=frame.captured_monotonic_ns,
                captured_at_utc=frame.captured_at_utc,
                metadata={**frame.metadata, "deferred_ocr": True},
            )
        )
    for sequence in range(5, 15):
        frame = camera_frame(sequence, [])
        pipeline.process(
            Frame(
                image=frame.image,
                sequence=frame.sequence,
                source_id=frame.source_id,
                captured_monotonic_ns=frame.captured_monotonic_ns,
                captured_at_utc=frame.captured_at_utc,
                metadata={**frame.metadata, "deferred_ocr": True},
            )
        )
    assert ocr.calls == 0


def test_crossing_recovery_accepts_repeated_number_with_truncated_variant() -> None:
    class TruncatedBurstOcr:
        def __init__(self) -> None:
            self.calls: dict[int, int] = {}

        def recognize(self, region, *, frame, track):  # type: ignore[no-untyped-def]
            del region, track
            self.calls[frame.sequence] = self.calls.get(frame.sequence, 0) + 1
            if self.calls[frame.sequence] == 1:
                return []
            text = {1: "004", 2: "004", 3: "00", 4: "8"}[frame.sequence]
            confidence = 0.98 if text != "8" else 0.56
            return [OcrPrediction(text, confidence)]

        def close(self) -> None:
            return None

    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=TruncatedBurstOcr(),  # type: ignore[arg-type]
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(instant_resolution_confidence=None)
        ),
        finish_line=FinishLine(0.1, 0.5, 0.9, 0.5),
    )
    for sequence, top in ((1, 5), (2, 15)):
        result = pipeline.process(
            camera_frame(
                sequence,
                [{"bbox": (20, top, 60, top + 30), "confidence": 0.99, "label": "motorcycle"}],
            )
        )
        assert result.passages == ()
    result = pipeline.process(
        camera_frame(
            3,
            [{"bbox": (20, 25, 60, 55), "confidence": 0.99, "label": "motorcycle"}],
        )
    )
    assert [item.racing_number for item in result.passages] == ["004"]


def test_crossing_recovery_rejects_unconfirmed_long_reading() -> None:
    class OneSharpFrameOcr:
        def __init__(self) -> None:
            self.calls: dict[int, int] = {}

        def recognize(self, region, *, frame, track):  # type: ignore[no-untyped-def]
            del region, track
            self.calls[frame.sequence] = self.calls.get(frame.sequence, 0) + 1
            if self.calls[frame.sequence] == 1 or frame.sequence != 2:
                return []
            return [OcrPrediction("435", 0.97)]

        def close(self) -> None:
            return None

    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=OneSharpFrameOcr(),  # type: ignore[arg-type]
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(instant_resolution_confidence=None)
        ),
        finish_line=FinishLine(0.1, 0.5, 0.9, 0.5),
    )
    plain = {"bbox": (20, 5, 60, 35), "confidence": 0.99, "label": "motorcycle"}
    crossed = {"bbox": (20, 45, 60, 75), "confidence": 0.99, "label": "motorcycle"}
    sharp = np.full((100, 100, 3), 245, dtype=np.uint8)
    cv2.putText(sharp, "435", (21, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 5, 5), 2)
    pipeline.process(camera_frame(1, [plain], image=sharp))
    result = pipeline.process(camera_frame(2, [crossed], image=sharp))
    assert result.passages == ()


def test_crossing_recovery_prefers_complete_three_digits_over_truncated_prefix() -> None:
    class CompleteAndTruncatedOcr:
        def __init__(self) -> None:
            self.calls: dict[int, int] = {}

        def recognize(self, region, *, frame, track):  # type: ignore[no-untyped-def]
            del region, track
            self.calls[frame.sequence] = self.calls.get(frame.sequence, 0) + 1
            if self.calls[frame.sequence] == 1 or frame.sequence != 2:
                return []
            return [OcrPrediction("222", 0.99), OcrPrediction("22", 0.91)]

        def close(self) -> None:
            return None

    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=CompleteAndTruncatedOcr(),  # type: ignore[arg-type]
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(instant_resolution_confidence=None)
        ),
        finish_line=FinishLine(0.1, 0.5, 0.9, 0.5),
    )
    plain = {"bbox": (20, 5, 60, 35), "confidence": 0.99, "label": "motorcycle"}
    crossed = {"bbox": (20, 45, 60, 75), "confidence": 0.99, "label": "motorcycle"}
    sharp = np.full((100, 100, 3), 245, dtype=np.uint8)
    cv2.putText(sharp, "222", (21, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 5, 5), 2)
    pipeline.process(camera_frame(1, [plain], image=sharp))
    result = pipeline.process(camera_frame(2, [crossed], image=sharp))
    assert [item.racing_number for item in result.passages] == ["222"]


def test_recovery_stitches_ordered_partial_readings_without_roster() -> None:
    class PartialOcr:
        def recognize(self, region, *, frame, track):  # type: ignore[no-untyped-def]
            del region, track
            return [
                OcrPrediction(
                    "43" if frame.sequence == 1 else "35",
                    0.96,
                    metadata={"engine": "test_partial"},
                )
            ]

        def close(self) -> None:
            return None

    pipeline = MotorcycleVisionPipeline(
        detector=MetadataObjectDetector(),
        tracker=CentroidIoUTracker(maximum_centroid_distance=100),
        region_extractor=BoundingBoxNumberRegionExtractor(),
        ocr_engine=PartialOcr(),  # type: ignore[arg-type]
        ocr_aggregator=OcrAggregator(),
        finish_line=FinishLine(0.1, 0.9, 0.9, 0.9),
    )
    detection = {"bbox": (20, 10, 60, 50), "confidence": 0.99, "label": "motorcycle"}
    for sequence in (1, 2):
        source = camera_frame(sequence, [detection])
        pipeline.process(
            Frame(
                image=source.image,
                sequence=source.sequence,
                source_id=source.source_id,
                captured_monotonic_ns=source.captured_monotonic_ns,
                captured_at_utc=source.captured_at_utc,
                metadata={**source.metadata, "deferred_ocr": True},
            )
        )
    recovered = pipeline.recover_pending_identities()
    assert [item[1] for item in recovered] == ["435"]


def test_two_motorcycles_cross_close_together_independently() -> None:
    pipeline = synthetic_pipeline()
    for sequence, top in ((1, 5), (2, 20)):
        result = pipeline.process(
            camera_frame(
                sequence,
                [
                    metadata_detection((10, top, 30, top + 20), "777"),
                    metadata_detection((65, top, 85, top + 20), "123"),
                ],
            )
        )
        assert result.passages == ()
    result = pipeline.process(
        camera_frame(
            3,
            [
                metadata_detection((10, 40, 30, 60), "777"),
                metadata_detection((65, 40, 85, 60), "123"),
            ],
        )
    )
    assert {item.racing_number for item in result.passages} == {"777", "123"}


def test_two_motorcycles_keep_independent_crossing_times_fifty_ms_apart() -> None:
    pipeline = synthetic_pipeline()
    before_timestamp = 60_000_000_000
    after_timestamp = 60_100_000_000
    pipeline.process(
        camera_frame(
            1,
            [
                metadata_detection((10, 10, 30, 30), "123"),
                metadata_detection((65, 0, 85, 10), "007"),
            ],
            timestamp_ns=59_900_000_000,
        )
    )
    pipeline.process(
        camera_frame(
            2,
            [
                metadata_detection((10, 20, 30, 40), "123"),
                metadata_detection((65, 0, 85, 20), "007"),
            ],
            timestamp_ns=before_timestamp,
        )
    )
    result = pipeline.process(
        camera_frame(
            3,
            [
                metadata_detection((10, 60, 30, 80), "123"),
                metadata_detection((65, 40, 85, 60), "007"),
            ],
            timestamp_ns=after_timestamp,
        )
    )
    times = {item.racing_number: item.captured_monotonic_ns for item in result.passages}
    assert times == {"123": 60_025_000_000, "007": 60_075_000_000}
    assert times["007"] - times["123"] == 50_000_000


def test_conflicting_ocr_never_creates_false_passage() -> None:
    pipeline = synthetic_pipeline()
    values = ("10", "1", "10", "1")
    passages: list[StablePassage] = []
    for sequence, (top, number) in enumerate(
        zip((5, 20, 35, 45), values, strict=True), start=1
    ):
        result = pipeline.process(
            camera_frame(sequence, [metadata_detection((20, top, 40, top + 20), number)])
        )
        passages.extend(result.passages)
    assert passages == []
