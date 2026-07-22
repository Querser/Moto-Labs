from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np

from app.camera import Frame
from app.vision.frame_selector import measure_region_quality, select_diverse_evidence
from app.vision.types import BoundingBox, CandidateRegion


@dataclass(frozen=True)
class Evidence:
    frame: Frame
    region: CandidateRegion
    quality: float


def evidence(sequence: int, image: np.ndarray, kind: str) -> Evidence:
    frame = Frame(
        image=image,
        sequence=sequence,
        source_id="selector-test",
        captured_monotonic_ns=sequence * 1_000_000,
        captured_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    region = CandidateRegion(
        image=image,
        bbox=BoundingBox(0, 0, image.shape[1], image.shape[0]),
        kind=kind,
        confidence=0.9,
    )
    return Evidence(frame, region, measure_region_quality(region).score)


def test_frame_selector_prefers_sharp_number_board() -> None:
    sharp = np.full((100, 180, 3), 245, dtype=np.uint8)
    cv2.putText(
        sharp,
        "123",
        (16, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (5, 5, 5),
        5,
        cv2.LINE_AA,
    )
    blurred = cv2.GaussianBlur(sharp, (25, 25), 8)
    assert measure_region_quality(
        evidence(1, sharp, "front_board_bright_0").region
    ).score > measure_region_quality(
        evidence(2, blurred, "front_board_bright_0").region
    ).score


def test_frame_selector_preserves_temporal_and_geometric_diversity() -> None:
    image = np.full((80, 140, 3), 180, dtype=np.uint8)
    values = [
        evidence(10, image, "front_board_bright_0"),
        evidence(11, image, "front_board_bright_0"),
        evidence(14, image, "front_board_rect_0"),
        evidence(18, image, "front_board_fallback"),
    ]
    selected = select_diverse_evidence(values, limit=3, minimum_frame_gap=2)
    assert {item.region.kind for item in selected} == {
        "front_board_bright_0",
        "front_board_rect_0",
        "front_board_fallback",
    }
