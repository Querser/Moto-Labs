"""Candidate number-region extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.camera import Frame

from .interfaces import NumberRegionExtractor
from .types import BoundingBox, CandidateRegion, Track


def _image_size(image: Any, metadata: dict[str, Any] | None = None) -> tuple[int, int] | None:
    if hasattr(image, "shape") and len(image.shape) >= 2:
        return int(image.shape[1]), int(image.shape[0])
    if isinstance(image, Sequence) and not isinstance(image, (str, bytes)) and image:
        try:
            return len(image[0]), len(image)
        except TypeError:
            return None
    if metadata and "frame_size" in metadata:
        width, height = metadata["frame_size"]
        return int(width), int(height)
    return None


def _crop(image: Any, bbox: BoundingBox) -> Any:
    x1, y1 = int(bbox.x1), int(bbox.y1)
    x2, y2 = int(bbox.x2), int(bbox.y2)
    if hasattr(image, "shape"):
        return image[y1:y2, x1:x2]
    if isinstance(image, Sequence) and not isinstance(image, (str, bytes)):
        return [row[x1:x2] for row in image[y1:y2]]
    return image


class BoundingBoxNumberRegionExtractor(NumberRegionExtractor):
    """Crop configurable normalized subregions inside each tracked object."""

    def __init__(
        self,
        regions: Sequence[tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.regions = tuple(regions or ((0.1, 0.05, 0.9, 0.95),))
        if not self.regions:
            raise ValueError("At least one candidate region is required")
        for region in self.regions:
            left, top, right, bottom = region
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("Candidate regions must be normalized and non-empty")

    def extract(self, frame: Frame, track: Track) -> list[CandidateRegion]:
        size = _image_size(frame.image, dict(frame.metadata))
        object_box = track.bbox
        if size is not None:
            try:
                object_box = object_box.clipped(*size)
            except ValueError:
                return []
        candidates: list[CandidateRegion] = []
        for index, (left, top, right, bottom) in enumerate(self.regions):
            bbox = BoundingBox(
                object_box.x1 + object_box.width * left,
                object_box.y1 + object_box.height * top,
                object_box.x1 + object_box.width * right,
                object_box.y1 + object_box.height * bottom,
            )
            candidates.append(
                CandidateRegion(
                    image=_crop(frame.image, bbox),
                    bbox=bbox,
                    kind=f"number_candidate_{index}",
                )
            )
        return candidates


DefaultNumberRegionExtractor = BoundingBoxNumberRegionExtractor


class FrontNumberBoardRegionExtractor(NumberRegionExtractor):
    """Find high-contrast board candidates in the front portion of a motorcycle.

    A generic motorcycle detector does not localize racing numbers. This stage
    searches a conservative front/fairing subregion for rectangular contrast
    boundaries, perspective-rectifies the best candidates, and retains one
    normalized fallback crop for difficult boards.
    """

    def __init__(
        self,
        *,
        front_region: tuple[float, float, float, float] = (0.02, 0.03, 0.98, 0.82),
        maximum_candidates: int = 3,
    ) -> None:
        left, top, right, bottom = front_region
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("Front number-board region must be normalized")
        if maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        self.front_region = front_region
        self.maximum_candidates = maximum_candidates

    def extract(self, frame: Frame, track: Track) -> list[CandidateRegion]:
        import cv2
        import numpy as np

        size = _image_size(frame.image, dict(frame.metadata))
        if size is None or frame.image is None or not hasattr(frame.image, "shape"):
            return []
        try:
            object_box = track.bbox.clipped(*size)
        except ValueError:
            return []
        left, top, right, bottom = self.front_region
        # On a side pass the rear panel can become larger than the front board.
        # Restrict OCR to the leading part of the tracked motorcycle whenever
        # horizontal motion is unambiguous.  A frontal approach keeps the full
        # width because its horizontal velocity is intentionally close to zero.
        object_box = _leading_object_box(object_box, track)
        front_box = BoundingBox(
            object_box.x1 + object_box.width * left,
            object_box.y1 + object_box.height * top,
            object_box.x1 + object_box.width * right,
            object_box.y1 + object_box.height * bottom,
        )
        crop = _crop(frame.image, front_box)
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 18:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        candidates = _bright_board_candidates(
            crop,
            front_box=front_box,
            frame_size=size,
            maximum_candidates=self.maximum_candidates,
        )
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)
        edges = cv2.Canny(normalized, 45, 135)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
            iterations=2,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        crop_area = float(crop.shape[0] * crop.shape[1])
        ranked: list[tuple[float, Any, BoundingBox]] = []
        for contour in contours:
            contour_area = float(abs(cv2.contourArea(contour)))
            area_ratio = contour_area / max(1.0, crop_area)
            if not 0.012 <= area_ratio <= 0.68:
                continue
            rectangle = cv2.minAreaRect(contour)
            rect_width, rect_height = rectangle[1]
            if min(rect_width, rect_height) < 10:
                continue
            aspect = max(rect_width, rect_height) / max(1.0, min(rect_width, rect_height))
            if not 1.0 <= aspect <= 5.5:
                continue
            rectangularity = contour_area / max(1.0, rect_width * rect_height)
            if rectangularity < 0.32:
                continue
            points = cv2.boxPoints(rectangle).astype(np.float32)
            x, y, width, height = cv2.boundingRect(points.astype(np.int32))
            try:
                global_box = BoundingBox(
                    front_box.x1 + x,
                    front_box.y1 + y,
                    front_box.x1 + min(crop.shape[1], x + width),
                    front_box.y1 + min(crop.shape[0], y + height),
                ).clipped(*size)
            except ValueError:
                continue
            score = min(1.0, 0.35 + area_ratio * 1.8 + rectangularity * 0.35)
            ranked.append((score, points, global_box))
        for index, (score, points, global_box) in enumerate(
            sorted(ranked, key=lambda item: item[0], reverse=True)
        ):
            if len(candidates) >= self.maximum_candidates:
                break
            warped = _perspective_crop(crop, points)
            if warped is None or warped.size == 0:
                continue
            candidates.append(
                CandidateRegion(
                    image=warped,
                    bbox=global_box,
                    kind=f"front_board_rect_{index}",
                    confidence=score,
                    metadata={"perspective_corrected": True},
                )
            )
        candidates.extend(
            _anchored_front_board_candidates(
                frame.image,
                front_box=front_box,
                frame_size=size,
            )
        )
        # A normalized fairing crop keeps recognition available when blur or a
        # borderless number board prevents reliable contour localization.
        candidates.append(
            CandidateRegion(
                image=crop,
                bbox=front_box,
                kind="front_board_fallback",
                confidence=0.35,
                metadata={"perspective_corrected": False},
            )
        )
        return candidates


def _bright_board_candidates(
    crop: Any,
    *,
    front_box: BoundingBox,
    frame_size: tuple[int, int],
    maximum_candidates: int,
) -> list[CandidateRegion]:
    """Propose bright, low-saturation front boards used on pit/enduro bikes."""

    import cv2
    import numpy as np

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    broad_mask = cv2.inRange(
        hsv,
        np.asarray((0, 0, 160), dtype=np.uint8),
        np.asarray((179, 135, 255), dtype=np.uint8),
    )
    broad_mask = cv2.morphologyEx(
        broad_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    # A stricter low-saturation mask separates a white number card from a
    # connected white headlight surround/fairing.  This is particularly useful
    # in compressed side footage where the broad mask merges both components.
    strict_mask = cv2.inRange(
        hsv,
        np.asarray((0, 0, 180), dtype=np.uint8),
        np.asarray((179, 90, 255), dtype=np.uint8),
    )
    strict_mask = cv2.morphologyEx(
        strict_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    strict_mask = cv2.morphologyEx(
        strict_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    crop_area = float(crop.shape[0] * crop.shape[1])
    ranked: list[tuple[float, str, tuple[int, int, int, int]]] = []
    for mask_kind, mask, strict_bonus in (
        ("broad", broad_mask, 0.0),
        ("strict", strict_mask, 0.08),
    ):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area_ratio = float(cv2.contourArea(contour)) / max(1.0, crop_area)
            aspect = width / max(1, height)
            rectangularity = float(cv2.contourArea(contour)) / max(1.0, width * height)
            if not 0.006 <= area_ratio <= 0.45:
                continue
            if not 0.35 <= aspect <= 3.5 or min(width, height) < 10:
                continue
            if rectangularity < 0.30:
                continue
            score = min(
                1.0,
                0.45 + rectangularity * 0.30 + area_ratio * 1.5 + strict_bonus,
            )
            ranked.append((score, mask_kind, (x, y, width, height)))
    result: list[CandidateRegion] = []
    accepted_boxes: list[tuple[int, int, int, int]] = []
    for score, mask_kind, (x, y, width, height) in sorted(
        ranked,
        key=lambda item: item[0],
        reverse=True,
    ):
        if len(result) >= maximum_candidates:
            break
        if any(
            _box_iou((x, y, width, height), accepted) >= 0.68
            for accepted in accepted_boxes
        ):
            continue
        # Keep context around the white component. OCR text detectors are much
        # less reliable when the first/last digit touches the crop boundary.
        padding_x = max(3, round(width * 0.10))
        padding_y = max(3, round(height * 0.10))
        padded_x1 = max(0, x - padding_x)
        padded_y1 = max(0, y - padding_y)
        padded_x2 = min(crop.shape[1], x + width + padding_x)
        padded_y2 = min(crop.shape[0], y + height + padding_y)
        try:
            global_box = BoundingBox(
                front_box.x1 + padded_x1,
                front_box.y1 + padded_y1,
                front_box.x1 + padded_x2,
                front_box.y1 + padded_y2,
            ).clipped(*frame_size)
        except ValueError:
            continue
        # The white-mask contour can end at a dark edge digit. Warping that
        # contour then clips a leading ``1`` or trailing digit and can turn
        # ``123`` into ``23``. Keep the padded source pixels here; the separate
        # edge-rectangle candidates below still provide perspective warps.
        candidate_image = crop[padded_y1:padded_y2, padded_x1:padded_x2]
        result.append(
            CandidateRegion(
                image=candidate_image,
                bbox=global_box,
                kind=f"front_board_bright_{mask_kind}_{len(result)}",
                confidence=score,
                metadata={
                    "perspective_corrected": False,
                    "bright_board": True,
                },
            )
        )
        accepted_boxes.append((x, y, width, height))
    return result


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Return IoU for ``x, y, width, height`` component boxes."""

    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    intersection_width = max(
        0,
        min(first_x + first_width, second_x + second_width) - max(first_x, second_x),
    )
    intersection_height = max(
        0,
        min(first_y + first_height, second_y + second_height) - max(first_y, second_y),
    )
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def _leading_object_box(object_box: BoundingBox, track: Track) -> BoundingBox:
    """Return the leading 74% of a side-moving motorcycle bounding box."""

    trajectory = track.trajectory
    if len(trajectory) < 3:
        return object_box
    recent = trajectory[-min(5, len(trajectory)) :]
    dx = recent[-1].x - recent[0].x
    # Detector jitter must not decide which end is the motorcycle front.
    if abs(dx) < max(4.0, object_box.width * 0.035):
        return object_box
    retained = object_box.width * 0.74
    if dx < 0:
        return BoundingBox(object_box.x1, object_box.y1, object_box.x1 + retained, object_box.y2)
    return BoundingBox(object_box.x2 - retained, object_box.y1, object_box.x2, object_box.y2)


def _anchored_front_board_candidates(
    image: Any,
    *,
    front_box: BoundingBox,
    frame_size: tuple[int, int],
) -> list[CandidateRegion]:
    """Cover predictable plate positions when contour localization is weak.

    Phone video compression, dust and motion blur often merge a white number
    board with the fairing.  Three overlapping upper-front crops cover frontal
    and front-side approaches without pretending that the generic motorcycle
    detector knows the exact board coordinates.
    """

    anchors = (
        ("left", 0.00, 0.00, 0.62, 0.48),
        ("compact", 0.18, 0.02, 0.82, 0.38),
        ("center", 0.16, 0.00, 0.86, 0.50),
        ("right", 0.38, 0.00, 1.00, 0.48),
    )
    result: list[CandidateRegion] = []
    for name, left, top, right, bottom in anchors:
        try:
            bbox = BoundingBox(
                front_box.x1 + front_box.width * left,
                front_box.y1 + front_box.height * top,
                front_box.x1 + front_box.width * right,
                front_box.y1 + front_box.height * bottom,
            ).clipped(*frame_size)
        except ValueError:
            continue
        crop = _crop(image, bbox)
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        result.append(
            CandidateRegion(
                image=crop,
                bbox=bbox,
                kind=f"front_board_anchor_{name}",
                confidence=0.50,
                metadata={"perspective_corrected": False, "anchored": True},
            )
        )
    return result


def _perspective_crop(image: Any, points: Any) -> Any | None:
    import cv2
    import numpy as np

    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sums = points.sum(axis=1)
    point_differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sums)]
    ordered[2] = points[np.argmax(point_sums)]
    ordered[1] = points[np.argmin(point_differences)]
    ordered[3] = points[np.argmax(point_differences)]
    top_left, top_right, bottom_right, bottom_left = ordered
    width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if width < 8 or height < 8:
        return None
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    result = cv2.warpPerspective(
        image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )
    if result.shape[0] > result.shape[1]:
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    return result


__all__ = ["BoundingBoxNumberRegionExtractor", "FrontNumberBoardRegionExtractor"]
