"""Evaluate production CV on user-supplied labels without influencing runtime.

The CSV is test-only ground truth. Expected numbers are read after inference and
are never passed to the detector, tracker, region extractor, OCR, or web app.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2

from app.camera import Frame
from app.vision import (
    FinishLine,
    FrontNumberBoardRegionExtractor,
    HybridDigitOcrEngine,
    MotorcycleVisionPipeline,
    OcrAggregationConfig,
    OcrAggregator,
    PaddleOcrV6DigitEngine,
    RapidOcrDigitEngine,
    SupervisionByteTrack,
    YoloXMotorcycleDetector,
)


@dataclass(frozen=True, slots=True)
class Label:
    event_id: str
    start_s: float
    end_s: float
    expected_number: str | None
    readable: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event", action="append", default=[])
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--window-gap",
        type=float,
        default=0.1,
        help="Merge labelled diagnostic windows separated by at most this many seconds",
    )
    parser.add_argument(
        "--line",
        default="0.25,0.05,0.25,0.95",
        help="Normalized finish line as x1,y1,x2,y2",
    )
    return parser.parse_args()


def load_labels(path: Path, selected: set[str]) -> list[Label]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Label(
            event_id=row["event_id"],
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            expected_number=row["expected_number"] or None,
            readable=row["readable"].strip().lower() == "true",
        )
        for row in rows
        if not selected or row["event_id"] in selected
    ]


def merge_windows(
    labels: list[Label], *, maximum_gap_s: float = 0.1
) -> list[tuple[float, float, list[Label]]]:
    windows: list[tuple[float, float, list[Label]]] = []
    for label in sorted(labels, key=lambda item: (item.start_s, item.end_s)):
        if windows and label.start_s <= windows[-1][1] + maximum_gap_s:
            start, end, members = windows[-1]
            windows[-1] = (start, max(end, label.end_s), [*members, label])
        else:
            windows.append((label.start_s, label.end_s, [label]))
    return windows


def parse_line(value: str) -> FinishLine:
    try:
        coordinates = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise SystemExit("--line must contain four decimal coordinates") from exc
    if len(coordinates) != 4:
        raise SystemExit("--line must contain x1,y1,x2,y2")
    try:
        return FinishLine(*coordinates)
    except ValueError as exc:
        raise SystemExit(f"Invalid --line: {exc}") from exc


def build_pipeline(finish_line: FinishLine) -> MotorcycleVisionPipeline:
    return MotorcycleVisionPipeline(
        detector=YoloXMotorcycleDetector(confidence_threshold=0.18),
        tracker=SupervisionByteTrack(
            track_activation_threshold=0.15,
            lost_track_buffer=30,
            minimum_matching_threshold=0.70,
            frame_rate=30,
        ),
        region_extractor=FrontNumberBoardRegionExtractor(maximum_candidates=3),
        ocr_engine=HybridDigitOcrEngine(
            RapidOcrDigitEngine(),
            PaddleOcrV6DigitEngine(),
        ),
        ocr_aggregator=OcrAggregator(
            OcrAggregationConfig(
                maximum_length=4,
                resolution_confidence=0.62,
                # Normal recognition requires temporal agreement. A single
                # exceptional frame is reconsidered only by the bounded
                # crossing-time recovery pass over the best retained crops.
                instant_resolution_confidence=None,
                minimum_observations=2,
                minimum_consecutive=2,
                maximum_observation_age_ns=3_000_000_000,
            )
        ),
        finish_line=finish_line,
        number_verifier=None,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    labels = load_labels(args.labels, set(args.event))
    if not labels:
        raise SystemExit("No labels selected")
    capture = cv2.VideoCapture(str(args.video.resolve()))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    finish_line = parse_line(args.line)
    pipeline = build_pipeline(finish_line)
    origin_utc = datetime.now(timezone.utc)
    report_windows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for start_s, end_s, members in merge_windows(
            labels, maximum_gap_s=max(0.0, args.window_gap)
        ):
            pipeline.reset()
            capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
            found: dict[str, float] = {}
            passages: dict[str, float] = {}
            processed_frames = 0
            full_cadence_until = 0
            source_index = max(0, round(start_s * fps))
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                position_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if position_s > end_s:
                    break
                timestamp_ns = round(position_s * 1_000_000_000)
                frame = Frame(
                    image=image,
                    sequence=source_index,
                    source_id=f"evaluation:{args.video.name}",
                    captured_monotonic_ns=timestamp_ns,
                    captured_at_utc=origin_utc + timedelta(seconds=position_s),
                    metadata={"evaluation_only": True, "deferred_ocr": True},
                )
                if pipeline.track_near_finish_line((image.shape[1], image.shape[0])):
                    full_cadence_until = max(
                        full_cadence_until,
                        source_index + round(fps * 0.65),
                    )
                if (
                    source_index % max(1, args.frame_step) == 0
                    or source_index <= full_cadence_until
                ):
                    result = pipeline.process(frame)
                    for _track_id, number, _confidence in result.track_numbers:
                        found.setdefault(number, position_s)
                    for passage in result.passages:
                        found.setdefault(passage.racing_number, position_s)
                        passages.setdefault(passage.racing_number, position_s)
                    processed_frames += 1
                else:
                    pipeline.collect_evidence(frame)
                source_index += 1
            # The timestamp supplied by the organizer identifies the visible
            # passage, not necessarily the later virtual-line crossing. Run
            # the same bounded best-frame recovery for OCR evaluation only;
            # this method cannot create a lap record.
            for _track_id, number, _confidence in pipeline.recover_pending_identities():
                found.setdefault(number, end_s)
            recovery_diagnostics = {
                str(track_id): list(items)
                for track_id, items in pipeline.recovery_diagnostics.items()
            }
            expected = sorted(
                item.expected_number
                for item in members
                if item.readable and item.expected_number is not None
            )
            report_windows.append(
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "event_ids": [item.event_id for item in members],
                    "expected": expected,
                    "recognized": sorted(found),
                    "matched": sorted(set(expected) & set(found)),
                    "missed": sorted(set(expected) - set(found)),
                    "unexpected": sorted(set(found) - set(expected)),
                    "first_seen_s": found,
                    "passages_s": passages,
                    "processed_frames": processed_frames,
                    "recovery_diagnostics": recovery_diagnostics,
                }
            )
    finally:
        pipeline.close()
        capture.release()
    expected_all = [value for row in report_windows for value in row["expected"]]
    matched_all = [value for row in report_windows for value in row["matched"]]
    unexpected_all = [value for row in report_windows for value in row["unexpected"]]
    precision = len(matched_all) / max(1, len(matched_all) + len(unexpected_all))
    recall = len(matched_all) / max(1, len(expected_all))
    return {
        "video": str(args.video.resolve()),
        "labels": str(args.labels.resolve()),
        "ground_truth_is_runtime_input": False,
        "finish_line": finish_line.as_dict(),
        "frame_step": max(1, args.frame_step),
        "elapsed_wall_s": round(time.perf_counter() - started, 3),
        "readable_expected_count": len(expected_all),
        "matched_count": len(matched_all),
        "unexpected_count": len(unexpected_all),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "windows": report_windows,
    }


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
