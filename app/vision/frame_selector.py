"""Quality and diversity selection for short per-track OCR bursts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar


class SelectableEvidence(Protocol):
    """Minimal evidence contract used without importing the main pipeline."""

    @property
    def frame(self) -> Any: ...

    @property
    def region(self) -> Any: ...

    @property
    def quality(self) -> float: ...


EvidenceT = TypeVar("EvidenceT", bound=SelectableEvidence)


@dataclass(frozen=True, slots=True)
class FrameQuality:
    score: float
    sharpness: float
    contrast: float
    exposure: float
    size: float


def measure_region_quality(region: Any) -> FrameQuality:
    """Measure useful OCR detail while penalizing blur and clipped exposure."""

    image = region.image
    if image is None or not hasattr(image, "shape") or getattr(image, "size", 0) == 0:
        return FrameQuality(0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = np.asarray(gray, dtype=np.uint8)
        laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = sobel_x * sobel_x + sobel_y * sobel_y
        tenengrad = float(np.sum(energy, dtype=np.float64)) / max(1, int(energy.size))
        _mean, standard_deviation = cv2.meanStdDev(gray)
        contrast_value = float(standard_deviation[0, 0])
        clipped_ratio = float(np.count_nonzero((gray <= 5) | (gray >= 250))) / max(
            1, int(gray.size)
        )
        mean_value = float(np.sum(gray, dtype=np.float64)) / max(1, int(gray.size))
    except Exception:
        laplacian = tenengrad = contrast_value = clipped_ratio = mean_value = 0.0
    height, width = int(image.shape[0]), int(image.shape[1])
    sharpness = min(1.0, 0.55 * laplacian / 420.0 + 0.45 * tenengrad / 6_000.0)
    contrast = min(1.0, contrast_value / 64.0)
    brightness_balance = max(0.0, 1.0 - abs(mean_value - 142.0) / 142.0)
    exposure = max(0.0, brightness_balance * (1.0 - min(1.0, clipped_ratio * 1.8)))
    size = min(1.0, math.sqrt(max(1, height * width)) / 190.0)
    localization = float(getattr(region, "confidence", 0.0))
    localized_bonus = (
        0.06
        if "bright" in str(getattr(region, "kind", ""))
        or "rect" in str(getattr(region, "kind", ""))
        else 0.0
    )
    score = min(
        1.0,
        0.30 * sharpness
        + 0.19 * contrast
        + 0.15 * exposure
        + 0.18 * size
        + 0.18 * localization
        + localized_bonus,
    )
    return FrameQuality(score, sharpness, contrast, exposure, size)


def select_diverse_evidence(
    evidence: list[EvidenceT],
    *,
    limit: int,
    minimum_frame_gap: int = 2,
) -> list[EvidenceT]:
    """Choose sharp candidates from distinct times and localization families."""

    if limit <= 0:
        return []
    ranked = sorted(evidence, key=lambda item: item.quality, reverse=True)
    selected: list[EvidenceT] = []
    selected_ids: set[int] = set()
    selected_sequences: list[int] = []
    selected_families: set[str] = set()

    def family(item: EvidenceT) -> str:
        kind = str(item.region.kind)
        if "bright" in kind:
            return "bright"
        if "rect" in kind:
            return "rect"
        if "anchor" in kind:
            return kind
        return "fallback"

    # Reserve a place for each useful geometric hypothesis before filling the
    # remaining slots by score. This helps when a white helmet outranks a board.
    for wanted in ("bright", "rect", "fallback"):
        candidate = next((item for item in ranked if family(item) == wanted), None)
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(id(candidate))
            selected_sequences.append(int(candidate.frame.sequence))
            selected_families.add(wanted)
            if len(selected) >= limit:
                return selected
    for item in ranked:
        if id(item) in selected_ids:
            continue
        sequence = int(item.frame.sequence)
        item_family = family(item)
        temporally_distinct = all(
            abs(sequence - existing) >= minimum_frame_gap for existing in selected_sequences
        )
        if not temporally_distinct and item_family in selected_families:
            continue
        selected.append(item)
        selected_ids.add(id(item))
        selected_sequences.append(sequence)
        selected_families.add(item_family)
        if len(selected) >= limit:
            break
    return selected


__all__ = ["FrameQuality", "measure_region_quality", "select_diverse_evidence"]
