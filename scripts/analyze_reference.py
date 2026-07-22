"""Reproducibly inspect and benchmark the user-provided race reference material."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2

from app.camera import Frame
from app.vision import (
    CandidateRegion,
    FinishLine,
    FrontNumberBoardRegionExtractor,
    OcrAggregationConfig,
    OcrAggregator,
    RapidOcrDigitEngine,
    SupervisionByteTrack,
    YoloXMotorcycleDetector,
)
from app.vision.pipeline import MotorcycleVisionPipeline
from app.vision.types import BoundingBox

EXTRACTION_POINTS = (
    (0.00, "empty_scene", "manual inspection"),
    (4.50, "first_appearance", "manual inspection"),
    (5.00, "first_reliable_detection_opportunity", "benchmark reference"),
    (5.43, "approach", "manual inspection"),
    (5.77, "clearest_number", "OCR validation"),
    (6.10, "clearest_board", "board/OCR validation"),
    (6.30, "finish_approach", "line configuration reference"),
    (6.43, "clearest_motorcycle", "detector validation"),
    (6.47, "close_pass", "detector validation"),
    (6.77, "leaving", "manual inspection"),
    (7.03, "rear_departure", "negative board example"),
    (7.47, "left_frame", "negative example"),
    (9.00, "empty_scene_after_pass", "negative example"),
    (16.50, "distant_second_appearance", "manual inspection"),
    (17.33, "distant_motorcycle", "detector validation"),
    (18.40, "distant_motorcycle", "detector validation"),
    (18.97, "video_end", "manual inspection"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=Path, default=Path("data/reference/original/reference_race.MOV")
    )
    parser.add_argument(
        "--still", type=Path, default=Path("data/reference/original/reference_still.png")
    )
    parser.add_argument(
        "--board", type=Path, default=Path("data/reference/original/reference_board.png")
    )
    parser.add_argument("--output", type=Path, default=Path("data/reference"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    (args.output / "annotated").mkdir(exist_ok=True)
    metadata = probe_video(args.video)
    extract_manifest(args.video, args.output, metadata)
    report = benchmark_video(args.video, args.output, metadata)
    report["standalone_images"] = {
        "still": inspect_image(args.still),
        "still_pipeline": evaluate_still_image(args.still, args.output),
        "board": inspect_image(args.board),
        "board_ocr": recognize_board_image(args.board),
    }
    (args.output / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip()
        return {
            "path": str(path),
            "format": path.suffix.upper().removeprefix("."),
            "width": width,
            "height": height,
            "orientation": "portrait" if height > width else "landscape",
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": frame_count / fps,
            "codec": codec,
            "backend": capture.getBackendName(),
        }
    finally:
        capture.release()


def extract_manifest(video_path: Path, output: Path, metadata: dict[str, Any]) -> None:
    capture = cv2.VideoCapture(str(video_path))
    rows: list[dict[str, Any]] = []
    try:
        for timestamp_s, visible, intended_use in EXTRACTION_POINTS:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            relative = Path("frames") / f"frame_{round(timestamp_s * 1000):06d}ms.jpg"
            cv2.imwrite(str(output / relative), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            rows.append(
                {
                    "source_filename": video_path.name,
                    "frame_timestamp_s": f"{timestamp_s:.3f}",
                    "extracted_file_path": str(relative).replace("\\", "/"),
                    "visible_object": visible,
                    "camera_angle": "front approach, then front-side pass and rear departure",
                    "number_visibility": (
                        "human-readable 306"
                        if 5.7 <= timestamp_s <= 6.5
                        else "limited or absent"
                    ),
                    "intended_use": intended_use,
                    "dataset_role": "manual inspection/validation only",
                }
            )
    finally:
        capture.release()
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "video_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def benchmark_video(path: Path, output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    detector = YoloXMotorcycleDetector(confidence_threshold=0.30)
    pipeline = MotorcycleVisionPipeline(
        detector=detector,
        tracker=SupervisionByteTrack(lost_track_buffer=12, frame_rate=round(metadata["fps"])),
        region_extractor=FrontNumberBoardRegionExtractor(maximum_candidates=3),
        ocr_engine=RapidOcrDigitEngine(),
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(
                maximum_length=4,
                resolution_confidence=0.62,
                minimum_observations=2,
                minimum_consecutive=2,
            )
        ),
        finish_line=FinishLine(0.52, 0.0, 0.52, 1.0),
    )
    capture = cv2.VideoCapture(str(path))
    first_detection_ms: float | None = None
    first_board_ms: float | None = None
    first_digit_ms: float | None = None
    first_stable_ms: float | None = None
    first_stable_number: str | None = None
    detector_latencies: list[float] = []
    processing_latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    crossing_count = 0
    started = time.perf_counter()
    origin_utc = datetime(2026, 7, 18, tzinfo=timezone.utc)
    try:
        sequence = 0
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            timestamp_ms = sequence * 1000.0 / metadata["fps"]
            frame = Frame(
                image=image,
                sequence=sequence,
                source_id="reference-video",
                captured_monotonic_ns=round(timestamp_ms * 1_000_000),
                captured_at_utc=origin_utc + timedelta(milliseconds=timestamp_ms),
            )
            inference_started = time.perf_counter_ns()
            result = pipeline.process(frame)
            latency_ms = (time.perf_counter_ns() - inference_started) / 1_000_000
            processing_latencies.append(latency_ms)
            # The pipeline detector is the direct model in this benchmark, so
            # this duration is an upper bound including tracking/board/OCR.
            if result.tracks:
                detector_latencies.append(latency_ms)
                if first_detection_ms is None:
                    first_detection_ms = timestamp_ms
                    annotate(image, result, output / "annotated" / "first_detection.jpg")
            if result.board_regions and first_board_ms is None:
                first_board_ms = timestamp_ms
                annotate(image, result, output / "annotated" / "first_board.jpg")
            if result.digit_regions and first_digit_ms is None:
                first_digit_ms = timestamp_ms
                annotate(image, result, output / "annotated" / "first_digit_region.jpg")
            if result.recognized_number and first_stable_ms is None:
                first_stable_ms = timestamp_ms
                first_stable_number = result.recognized_number
                annotate(image, result, output / "annotated" / "first_stable_number.jpg")
            crossing_count += len(result.passages)
            if result.tracks or result.board_regions or result.digit_regions:
                rows.append(
                    {
                        "frame": sequence,
                        "timestamp_ms": round(timestamp_ms, 3),
                        "tracks": len(result.tracks),
                        "boards": len(result.board_regions),
                        "digit_regions": len(result.digit_regions),
                        "stable_number": result.recognized_number or "",
                        "latency_ms": round(latency_ms, 3),
                    }
                )
            if sequence in {156, 172, 184, 193, 194}:
                annotate(image, result, output / "annotated" / f"pipeline_{sequence:04d}.jpg")
            sequence += 1
    finally:
        capture.release()
        pipeline.close()
    elapsed = time.perf_counter() - started
    with (output / "detections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["frame"])
        writer.writeheader()
        writer.writerows(rows)
    clearly_visible_ms = 5000.0
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "providers": list(detector.providers),
        },
        "video": metadata,
        "processed_frames": sequence,
        "wall_time_s": elapsed,
        "processed_fps": sequence / elapsed,
        "mean_pipeline_latency_ms": statistics.fmean(processing_latencies),
        "p95_pipeline_latency_ms": sorted(processing_latencies)[
            min(len(processing_latencies) - 1, round(len(processing_latencies) * 0.95))
        ],
        "first_motorcycle_detection_ms": first_detection_ms,
        "first_board_candidate_ms": first_board_ms,
        "first_digit_region_ms": first_digit_ms,
        "first_stable_number_ms": first_stable_ms,
        "first_stable_number": first_stable_number,
        "detection_after_manual_visible_ms": (
            first_detection_ms - clearly_visible_ms if first_detection_ms is not None else None
        ),
        "stable_after_first_detection_ms": (
            first_stable_ms - first_detection_ms
            if first_stable_ms is not None and first_detection_ms is not None
            else None
        ),
        "stable_after_first_digit_region_ms": (
            first_stable_ms - first_digit_ms
            if first_stable_ms is not None and first_digit_ms is not None
            else None
        ),
        "passages_from_reference_line": crossing_count,
        "training_performed": False,
    }


def annotate(image: Any, result: Any, target: Path) -> None:
    annotated = image.copy()
    colors = {"motorcycle": (60, 220, 120), "board": (255, 210, 30), "digits": (30, 150, 255)}
    for track in result.tracks:
        draw_box(annotated, track.bbox, colors["motorcycle"], f"Motorcycle #{track.track_id}")
    for board in result.board_regions:
        draw_box(annotated, board.bbox, colors["board"], "Number board")
    for digits in result.digit_regions:
        draw_box(annotated, digits.bbox, colors["digits"], f"Digits {digits.text or ''}")
    cv2.imwrite(str(target), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])


def draw_box(image: Any, bbox: BoundingBox, color: tuple[int, int, int], label: str) -> None:
    cv2.rectangle(
        image,
        (round(bbox.x1), round(bbox.y1)),
        (round(bbox.x2), round(bbox.y2)),
        color,
        3,
    )
    cv2.putText(
        image,
        label,
        (round(bbox.x1), max(24, round(bbox.y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def inspect_image(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path))
    if image is None:
        return {"path": str(path), "readable": False}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "path": str(path),
        "readable": True,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "orientation": "portrait" if image.shape[0] > image.shape[1] else "landscape",
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_luminance": float(gray.mean()),
    }


def evaluate_still_image(path: Path, output: Path) -> dict[str, Any]:
    """Run the real shared stages on a still, repeating it only for OCR consensus."""

    image = cv2.imread(str(path))
    if image is None:
        return {"path": str(path), "readable": False}
    detector = YoloXMotorcycleDetector(confidence_threshold=0.30)
    pipeline = MotorcycleVisionPipeline(
        detector=detector,
        tracker=SupervisionByteTrack(lost_track_buffer=12, frame_rate=30),
        region_extractor=FrontNumberBoardRegionExtractor(maximum_candidates=3),
        ocr_engine=RapidOcrDigitEngine(),
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(
                maximum_length=4,
                resolution_confidence=0.62,
                minimum_observations=2,
                minimum_consecutive=2,
            )
        ),
        finish_line=FinishLine(0.95, 0.0, 0.95, 1.0),
    )
    consensus_frames = 20
    results = []
    started = time.perf_counter_ns()
    try:
        for sequence in range(consensus_frames):
            frame = Frame(
                image=image.copy(),
                sequence=sequence,
                source_id="reference-still",
                captured_monotonic_ns=sequence * 33_333_333,
                captured_at_utc=datetime(2026, 7, 18, tzinfo=timezone.utc)
                + timedelta(microseconds=sequence * 33_333),
            )
            results.append(pipeline.process(frame))
        best = max(
            results,
            key=lambda item: (
                bool(item.recognized_number),
                len(item.digit_regions),
                len(item.board_regions),
                len(item.tracks),
            ),
        )
        annotate(image, best, output / "annotated" / "reference_still_pipeline.jpg")
        return {
            "path": str(path),
            "frames_repeated_for_consensus": consensus_frames,
            "motorcycle_tracks": max(len(item.tracks) for item in results),
            "board_regions": max(len(item.board_regions) for item in results),
            "digit_regions": max(len(item.digit_regions) for item in results),
            "stable_number": next(
                (item.recognized_number for item in results if item.recognized_number),
                None,
            ),
            "wall_latency_ms": (time.perf_counter_ns() - started) / 1_000_000,
            "annotated_path": str(
                output / "annotated" / "reference_still_pipeline.jpg"
            ),
        }
    finally:
        pipeline.close()


def recognize_board_image(path: Path) -> list[dict[str, Any]]:
    image = cv2.imread(str(path))
    if image is None:
        return []
    engine = RapidOcrDigitEngine()
    frame = Frame.captured(image, sequence=0, source_id="reference-board")
    region = CandidateRegion(
        image=image,
        bbox=BoundingBox(0, 0, image.shape[1], image.shape[0]),
        kind="reference_board",
    )
    # Track content is unused by RapidOCR but its contract requires one.
    from app.vision.types import Track

    track = Track(1, region.bbox, 1.0, 1, 1, 0, True, frame.captured_monotonic_ns)
    return [
        {"text": item.text, "confidence": item.confidence, "metadata": dict(item.metadata)}
        for item in engine.recognize(region, frame=frame, track=track)
    ]


if __name__ == "__main__":
    main()
