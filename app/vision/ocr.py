"""OCR engines, normalization, and confidence-weighted temporal consensus."""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.camera import Frame

from .hardware import hardware_profile
from .interfaces import InferenceError, ModelLoadError, OcrEngine
from .types import (
    CandidateRegion,
    OcrPrediction,
    OcrResolution,
    OcrResolutionStatus,
    Track,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OcrAggregationConfig:
    minimum_length: int = 1
    maximum_length: int = 4
    minimum_observation_confidence: float = 0.15
    resolution_confidence: float = 0.62
    instant_resolution_confidence: float | None = None
    minimum_observations: int = 2
    minimum_consecutive: int = 2
    conflict_margin: float = 0.12
    window_size: int = 16
    maximum_observation_age_ns: int = 900_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_length <= self.maximum_length:
            raise ValueError("OCR number length range is invalid")
        if not 0 <= self.minimum_observation_confidence <= 1:
            raise ValueError("minimum_observation_confidence must be in [0, 1]")
        if not 0 <= self.resolution_confidence <= 1:
            raise ValueError("resolution_confidence must be in [0, 1]")
        if (
            self.instant_resolution_confidence is not None
            and not 0 <= self.instant_resolution_confidence <= 1
        ):
            raise ValueError("instant_resolution_confidence must be in [0, 1]")
        if self.minimum_observations < 1 or self.minimum_consecutive < 1:
            raise ValueError("OCR observation thresholds must be positive")
        if not 0 <= self.conflict_margin <= 1:
            raise ValueError("conflict_margin must be in [0, 1]")
        if self.window_size < self.minimum_observations:
            raise ValueError("window_size is smaller than minimum_observations")
        if self.maximum_observation_age_ns <= 0:
            raise ValueError("maximum_observation_age_ns must be positive")


@dataclass(frozen=True, slots=True)
class OcrObservation:
    raw_text: str
    normalized_text: str | None
    confidence: float
    captured_monotonic_ns: int
    frame_sequence: int


def normalize_racing_number(
    raw_text: str,
    *,
    minimum_length: int = 1,
    maximum_length: int = 4,
) -> str | None:
    """Keep ASCII digits only while preserving leading zeroes."""

    digits = "".join(character for character in str(raw_text) if character in "0123456789")
    if not minimum_length <= len(digits) <= maximum_length:
        return None
    return digits


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_character != second_character),
                )
            )
        previous = current
    return previous[-1]


class OcrAggregator:
    """Track-scoped weighted voting with temporal stability and whitelist checks."""

    def __init__(
        self,
        config: OcrAggregationConfig | None = None,
        *,
        participant_whitelist: Iterable[str] | None = None,
    ) -> None:
        self.config = config or OcrAggregationConfig()
        self._whitelist: set[str] = set()
        self.set_whitelist(participant_whitelist)
        self._observations: dict[int, deque[OcrObservation]] = {}

    @property
    def participant_whitelist(self) -> frozenset[str]:
        return frozenset(self._whitelist)

    def set_whitelist(self, values: Iterable[str] | None) -> None:
        normalized: set[str] = set()
        for value in values or ():
            candidate = normalize_racing_number(
                str(value),
                minimum_length=self.config.minimum_length,
                maximum_length=self.config.maximum_length,
            )
            if candidate != str(value):
                raise ValueError(f"Invalid participant number in whitelist: {value!r}")
            normalized.add(candidate)
        self._whitelist = normalized

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._observations.clear()
        else:
            self._observations.pop(track_id, None)

    def observe(
        self,
        track_id: int,
        prediction: OcrPrediction,
        *,
        captured_monotonic_ns: int,
        frame_sequence: int,
    ) -> OcrResolution:
        observations = self._observations.setdefault(
            track_id, deque(maxlen=self.config.window_size)
        )
        normalized = normalize_racing_number(
            prediction.text,
            minimum_length=self.config.minimum_length,
            maximum_length=self.config.maximum_length,
        )
        observations.append(
            OcrObservation(
                raw_text=prediction.text,
                normalized_text=normalized,
                confidence=prediction.confidence,
                captured_monotonic_ns=captured_monotonic_ns,
                frame_sequence=frame_sequence,
            )
        )
        cutoff = captured_monotonic_ns - self.config.maximum_observation_age_ns
        while observations and observations[0].captured_monotonic_ns < cutoff:
            observations.popleft()
        return self.resolve(track_id)

    def observe_predictions(
        self,
        track_id: int,
        predictions: Sequence[OcrPrediction],
        *,
        captured_monotonic_ns: int,
        frame_sequence: int,
    ) -> OcrResolution:
        # Multiple candidate crops can report the same value in one frame. Only
        # the strongest same-value prediction is counted to avoid overweighting.
        strongest: dict[str, OcrPrediction] = {}
        for prediction in predictions:
            normalized = normalize_racing_number(
                prediction.text,
                minimum_length=self.config.minimum_length,
                maximum_length=self.config.maximum_length,
            )
            key = normalized if normalized is not None else f"!{prediction.text}"
            existing = strongest.get(key)
            if existing is None or prediction.confidence > existing.confidence:
                strongest[key] = prediction
        resolution = self.resolve(track_id)
        for prediction in strongest.values():
            resolution = self.observe(
                track_id,
                prediction,
                captured_monotonic_ns=captured_monotonic_ns,
                frame_sequence=frame_sequence,
            )
        return resolution

    def resolve(self, track_id: int) -> OcrResolution:
        observations = list(self._observations.get(track_id, ()))
        if not observations:
            return OcrResolution(OcrResolutionStatus.UNKNOWN, None, 0, 0)
        valid = [
            observation
            for observation in observations
            if observation.normalized_text is not None
            and observation.confidence >= self.config.minimum_observation_confidence
        ]
        if not valid:
            return OcrResolution(
                OcrResolutionStatus.INVALID,
                None,
                0,
                len(observations),
            )

        weights: defaultdict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for observation in valid:
            assert observation.normalized_text is not None
            weights[observation.normalized_text] += observation.confidence
            counts[observation.normalized_text] += 1
        total_weight = sum(weights.values())
        scored = sorted(
            (
                (
                    text,
                    (weight / max(1, counts[text])) * (weight / max(0.001, total_weight)),
                )
                for text, weight in weights.items()
            ),
            key=lambda item: (item[1], counts[item[0]], item[0]),
            reverse=True,
        )
        top_text, top_score = scored[0]
        alternatives = tuple(scored[:5])
        latest = valid[-1]
        latest_text = latest.normalized_text
        instant_threshold = self.config.instant_resolution_confidence
        effective_instant_threshold = (
            max(0.995, instant_threshold)
            if instant_threshold is not None
            and latest_text is not None
            and len(latest_text) == 1
            else instant_threshold
        )
        if (
            latest_text is not None
            and effective_instant_threshold is not None
            and latest.confidence >= effective_instant_threshold
        ):
            if self._whitelist and latest_text not in self._whitelist:
                instant_nearby = tuple(
                    sorted(
                        self._whitelist,
                        key=lambda value: (_edit_distance(latest_text, value), value),
                    )[:5]
                )
                return OcrResolution(
                    OcrResolutionStatus.NOT_IN_WHITELIST,
                    latest_text,
                    latest.confidence,
                    len(valid),
                    alternatives,
                    instant_nearby,
                )
            return OcrResolution(
                OcrResolutionStatus.RESOLVED,
                latest_text,
                latest.confidence,
                len(valid),
                alternatives,
            )
        consecutive = 0
        for observation in reversed(valid):
            if observation.normalized_text != top_text:
                break
            consecutive += 1

        nearby: tuple[str, ...] = ()
        if self._whitelist and top_text not in self._whitelist:
            nearby = tuple(
                sorted(
                    self._whitelist,
                    key=lambda value: (_edit_distance(top_text, value), value),
                )[:5]
            )
            return OcrResolution(
                OcrResolutionStatus.NOT_IN_WHITELIST,
                top_text,
                top_score,
                len(valid),
                alternatives,
                nearby,
            )

        runner_score = scored[1][1] if len(scored) > 1 else 0.0
        if len(scored) > 1 and top_score - runner_score < self.config.conflict_margin:
            status = OcrResolutionStatus.CONFLICTING
        elif (
            counts[top_text]
            < max(self.config.minimum_observations, 3 if len(top_text) == 1 else 1)
            or (
                consecutive < self.config.minimum_consecutive
                and not (
                    counts[top_text] >= 3
                    and weights[top_text] / max(0.001, total_weight) >= 0.72
                )
            )
            or top_score < self.config.resolution_confidence
        ):
            status = OcrResolutionStatus.LOW_CONFIDENCE
        else:
            status = OcrResolutionStatus.RESOLVED
        return OcrResolution(
            status,
            top_text,
            top_score,
            len(valid),
            alternatives,
            nearby,
        )


class MetadataOcrEngine(OcrEngine):
    """Read explicit OCR observations from synthetic detection metadata."""

    def __init__(self, metadata_key: str = "ocr") -> None:
        self.metadata_key = metadata_key

    def recognize(
        self,
        region: CandidateRegion,
        *,
        frame: Frame,
        track: Track,
    ) -> list[OcrPrediction]:
        raw = track.metadata.get(self.metadata_key)
        if raw is None and "racing_number" in track.metadata:
            raw = (
                str(track.metadata["racing_number"]),
                float(track.metadata.get("ocr_confidence", 1.0)),
            )
        if raw is None:
            return []
        items: list[Any]
        if isinstance(raw, (str, Mapping, OcrPrediction)) or (
            isinstance(raw, Sequence)
            and len(raw) == 2
            and isinstance(raw[0], str)
            and isinstance(raw[1], (int, float))
        ):
            items = [raw]
        else:
            items = list(raw)
        predictions: list[OcrPrediction] = []
        for item in items:
            if isinstance(item, OcrPrediction):
                predictions.append(item)
            elif isinstance(item, str):
                predictions.append(
                    OcrPrediction(
                        item,
                        1.0,
                        metadata={"synthetic": True, "digit_bbox": (0.2, 0.2, 0.8, 0.8)},
                    )
                )
            elif isinstance(item, Mapping):
                predictions.append(
                    OcrPrediction(
                        text=str(item.get("text", "")),
                        confidence=float(item.get("confidence", 0)),
                        alternatives=tuple(item.get("alternatives", ())),
                        metadata={
                            "synthetic": True,
                            "digit_bbox": tuple(item.get("digit_bbox", (0.2, 0.2, 0.8, 0.8))),
                        },
                    )
                )
            elif isinstance(item, Sequence) and len(item) == 2:
                predictions.append(
                    OcrPrediction(
                        str(item[0]),
                        float(item[1]),
                        metadata={"synthetic": True, "digit_bbox": (0.2, 0.2, 0.8, 0.8)},
                    )
                )
            else:
                raise InferenceError(f"Invalid synthetic OCR observation: {item!r}")
        return predictions


class OpenCVDigitOcrEngine(OcrEngine):
    """Template-matching CPU baseline for high-contrast printed digit cards."""

    def __init__(
        self,
        *,
        minimum_digit_height_ratio: float = 0.25,
        minimum_character_confidence: float = 0.35,
    ) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ModelLoadError(
                "OpenCV digit OCR requires opencv-python-headless and numpy"
            ) from exc
        if not 0 < minimum_digit_height_ratio <= 1:
            raise ValueError("minimum_digit_height_ratio must be in (0, 1]")
        if not 0 <= minimum_character_confidence <= 1:
            raise ValueError("minimum_character_confidence must be in [0, 1]")
        self._cv2 = cv2
        self._np = np
        self.minimum_digit_height_ratio = minimum_digit_height_ratio
        self.minimum_character_confidence = minimum_character_confidence
        self._template_size = (32, 48)
        self._templates = self._build_templates()

    def _build_templates(self) -> dict[str, Any]:
        templates: dict[str, Any] = {}
        width, height = self._template_size
        for digit in "0123456789":
            canvas = self._np.zeros((height, width), dtype=self._np.uint8)
            font = self._cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 1.5, 3
            (text_width, text_height), _ = self._cv2.getTextSize(digit, font, scale, thickness)
            origin = ((width - text_width) // 2, (height + text_height) // 2)
            self._cv2.putText(
                canvas,
                digit,
                origin,
                font,
                scale,
                (255,),
                thickness,
                self._cv2.LINE_AA,
            )
            templates[digit] = canvas
        return templates

    def recognize(
        self,
        region: CandidateRegion,
        *,
        frame: Frame,
        track: Track,
    ) -> list[OcrPrediction]:
        image = region.image
        if image is None or not hasattr(image, "shape"):
            return []
        try:
            gray = (
                self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3
                else image
            )
            gray = self._cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = self._cv2.threshold(
                gray, 0, 255, self._cv2.THRESH_BINARY_INV | self._cv2.THRESH_OTSU
            )
            contours, _ = self._cv2.findContours(
                binary, self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
            )
        except Exception as exc:
            raise InferenceError(f"Digit OCR preprocessing failed: {exc}") from exc
        minimum_height = binary.shape[0] * self.minimum_digit_height_ratio
        boxes = [
            self._cv2.boundingRect(contour)
            for contour in contours
            if self._cv2.boundingRect(contour)[3] >= minimum_height
        ]
        boxes.sort(key=lambda box: box[0])
        text = ""
        confidences: list[float] = []
        alternatives: list[tuple[str, float]] = []
        for x, y, width, height in boxes:
            glyph = binary[y : y + height, x : x + width]
            resized = self._cv2.resize(glyph, self._template_size)
            scores = []
            for digit, template in self._templates.items():
                correlation = float(
                    self._cv2.matchTemplate(resized, template, self._cv2.TM_CCOEFF_NORMED)[0][0]
                )
                scores.append((digit, max(0.0, min(1.0, (correlation + 1) / 2))))
            scores.sort(key=lambda item: item[1], reverse=True)
            digit, confidence = scores[0]
            if confidence < self.minimum_character_confidence:
                continue
            text += digit
            confidences.append(confidence)
            alternatives.extend(scores[1:3])
        if not text:
            return []
        return [
            OcrPrediction(
                text=text,
                confidence=sum(confidences) / len(confidences),
                alternatives=tuple(alternatives[:5]),
                metadata={"baseline": "opencv_template"},
            )
        ]


BaselineOcrEngine = OpenCVDigitOcrEngine


class RapidOcrDigitEngine(OcrEngine):
    """Offline ONNX text detector/recognizer filtered to 1-4 ASCII digits."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.38,
        edge_margin_ratio: float = 0.02,
        fast_minimum_confidence: float = 0.68,
        trigger_center_tolerance: float = 0.14,
        full_detection_interval: int = 2,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0 <= edge_margin_ratio < 0.25:
            raise ValueError("edge_margin_ratio must be in [0, 0.25)")
        if not 0 <= fast_minimum_confidence <= 1:
            raise ValueError("fast_minimum_confidence must be in [0, 1]")
        if not 0 < trigger_center_tolerance <= 0.5:
            raise ValueError("trigger_center_tolerance must be in (0, 0.5]")
        if full_detection_interval < 1:
            raise ValueError("full_detection_interval must be positive")
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ModelLoadError(
                "RapidOCR is unavailable; install rapidocr-onnxruntime"
            ) from exc
        try:
            # Live inference stays lightweight. A second session with a larger
            # detector input is reserved for the bounded best-frame recovery at
            # a crossing, so camera capture never waits for exhaustive OCR.
            # RapidOCR defaults to letting every ONNX session use all logical
            # CPU cores.  The detector, recognizer and YOLO session then fight
            # over the same cores and one small board can take more than a
            # second.  Two OCR threads measured substantially faster on the
            # target Windows laptop while leaving capacity for video decode.
            session_options = {
                "intra_op_num_threads": 2,
                "inter_op_num_threads": 1,
            }
            self._engine = RapidOCR(det_limit_side_len=256, **session_options)
            self._recovery_engine = RapidOCR(
                det_limit_side_len=480,
                **session_options,
            )
            # PP-OCRv5 English/number recognition is used only for a bounded
            # second look at the best whole-fairing crops.  The older compact
            # engine remains the low-latency path on every analyzed frame.
            from rapidocr import (  # type: ignore[import-untyped]
                LangRec,
                ModelType,
                OCRVersion,
            )
            from rapidocr import (
                RapidOCR as RapidOCRV5,
            )

            self._v5_engine = RapidOCRV5(
                params={
                    "Global.log_level": "critical",
                    "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                    "Det.limit_type": "min",
                    "Det.limit_side_len": 480,
                    "Rec.lang_type": LangRec.EN,
                    "Rec.model_type": ModelType.MOBILE,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                }
            )
        except Exception as exc:
            raise ModelLoadError(f"Could not initialize offline digit OCR: {exc}") from exc
        self.minimum_confidence = minimum_confidence
        self.edge_margin_ratio = edge_margin_ratio
        self.fast_minimum_confidence = fast_minimum_confidence
        self.trigger_center_tolerance = trigger_center_tolerance
        self.full_detection_interval = full_detection_interval
        self._warm_up()

    def _warm_up(self) -> None:
        """Pay ONNX first-inference cost before a motorcycle reaches the line."""

        try:
            import cv2
            import numpy as np

            image = np.full((96, 192, 3), 255, dtype=np.uint8)
            cv2.putText(
                image,
                "88",
                (28, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            self._engine(image, use_det=True, use_cls=False, use_rec=True)
            self._recovery_engine(image, use_det=True, use_cls=False, use_rec=True)
            self._v5_engine(image, use_det=True, use_cls=False, use_rec=True)
        except Exception:
            # A warm-up failure is not fatal; the normal inference path still
            # reports a precise error if the model is genuinely unavailable.
            logger.warning("RapidOCR warm-up was not completed", exc_info=True)

    def recognize(
        self,
        region: CandidateRegion,
        *,
        frame: Frame,
        track: Track,
    ) -> list[OcrPrediction]:
        image = region.image
        if image is None or not hasattr(image, "shape") or image.size == 0:
            return []
        image = self._upscale_small_board(image)
        v5_prediction: OcrPrediction | None = None
        if frame.metadata.get("v5_recovery_ocr", False):
            v5_prediction = self._recognize_v5_recovery(image, track=track)
        direct_prediction = self._recognize_direct_localized_board(image, region)
        fast_prediction, _digit_group_outside_center = self._recognize_fast_digit_group(
            image
        )
        if direct_prediction is not None and (
            fast_prediction is None
            or len(direct_prediction.text) > len(fast_prediction.text)
            or direct_prediction.confidence >= fast_prediction.confidence + 0.08
        ):
            return [direct_prediction, *([v5_prediction] if v5_prediction else [])]
        if fast_prediction is not None:
            return [fast_prediction, *([v5_prediction] if v5_prediction else [])]
        if frame.metadata.get("fast_ocr_only", False):
            return [v5_prediction] if v5_prediction is not None else []
        # Number identity is associated with the motorcycle track, while the
        # tracked motorcycle geometry decides whether the finish line was
        # crossed.  An older capture-zone implementation rejected text outside
        # the centre of an arbitrary crop here.  That discarded valid plates
        # near the left/right edge during a side pass and is unrelated to the
        # configured finish line, so centring is now diagnostic metadata only.
        # The recognition-only contour path runs on every candidate. The much
        # heavier text detector is a periodic fallback for angled or blurred
        # boards, preventing empty motorcycle crops from stalling fresh frames.
        if (
            frame.sequence % self.full_detection_interval
            and not frame.metadata.get("force_full_ocr", False)
        ):
            return []
        full_engine = (
            self._recovery_engine
            if frame.metadata.get("high_resolution_ocr", False)
            else self._engine
        )
        try:
            results, _timing = full_engine(
                image,
                use_det=True,
                use_cls=False,
                use_rec=True,
            )
        except Exception as exc:
            raise InferenceError(f"Offline digit OCR failed: {exc}") from exc
        base_prediction = self._prediction_from_results(
            results,
            image=image,
            preprocessing="color",
        )
        should_check_variants = (
            base_prediction is not None and base_prediction.confidence < 0.90
        ) or (
            base_prediction is None
            and region.kind != "front_board_fallback"
            and region.confidence >= 0.48
        )
        predictions = [base_prediction] if base_prediction is not None else []
        variant_limit = int(frame.metadata.get("preprocessing_variant_limit", 3))
        if should_check_variants and variant_limit > 0:
            predictions.extend(
                self._preprocessing_variants(
                    image,
                    engine=full_engine,
                    limit=variant_limit,
                )
            )
        if not predictions:
            return [v5_prediction] if v5_prediction is not None else []
        merged = self._merge_preprocessing_consensus(predictions)
        return [merged, *([v5_prediction] if v5_prediction else [])]

    def _recognize_v5_recovery(
        self,
        image: Any,
        *,
        track: Track,
    ) -> OcrPrediction | None:
        """Run PP-OCRv5 on one retained whole-board/fairing crop.

        Its text detector recovers small lines that the contour shortcut cannot
        isolate, while strict token filtering prevents timestamps, punctuation,
        and ordinary words from becoming racing numbers.
        """

        try:
            output = self._v5_engine(
                image,
                use_det=True,
                use_cls=False,
                use_rec=True,
            )
        except Exception:
            logger.debug("RapidOCR PP-OCRv5 recovery failed", exc_info=True)
            return None
        texts = tuple(getattr(output, "txts", ()) or ())
        scores = tuple(getattr(output, "scores", ()) or ())
        boxes = getattr(output, "boxes", None)
        candidates: list[tuple[float, str, float, Any]] = []
        for index, (raw_text, raw_score) in enumerate(zip(texts, scores, strict=False)):
            compact = str(raw_text).strip().upper()
            # Do not strip punctuation here: ``48.3s`` must never become a
            # four-digit racing number merely because the OCR saw a timestamp.
            if not compact or not compact.isalnum():
                continue
            digits = self._normalize_model_text(compact)
            confidence = float(raw_score)
            if digits is None or confidence < 0.45:
                continue
            box = boxes[index] if boxes is not None and index < len(boxes) else None
            center_x = 0.0
            if box is not None:
                try:
                    center_x = sum(float(point[0]) for point in box) / len(box)
                except (TypeError, ValueError, IndexError):
                    center_x = 0.0
            normalized_center_x = center_x / max(1, int(image.shape[1]))
            if not self._is_on_leading_side(normalized_center_x, track):
                continue
            candidates.append((center_x, digits, confidence, box))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        combined = "".join(item[1] for item in candidates)
        if 1 <= len(combined) <= 4:
            digits = combined
            confidence = sum(item[2] for item in candidates) / len(candidates)
            selected = candidates
        else:
            strongest = max(candidates, key=lambda item: (len(item[1]), item[2]))
            digits, confidence, selected = strongest[1], strongest[2], [strongest]
        points = [point for item in selected if item[3] is not None for point in item[3]]
        if points:
            width = max(1, int(image.shape[1]))
            height = max(1, int(image.shape[0]))
            digit_bbox = (
                min(float(point[0]) for point in points) / width,
                min(float(point[1]) for point in points) / height,
                max(float(point[0]) for point in points) / width,
                max(float(point[1]) for point in points) / height,
            )
        else:
            digit_bbox = (0.0, 0.0, 1.0, 1.0)
        return OcrPrediction(
            text=digits,
            confidence=confidence,
            metadata={
                "engine": "rapidocr_ppocrv5_english_mobile",
                "offline": True,
                "recovery_path": True,
                "trigger_ready": True,
                "digit_bbox": digit_bbox,
            },
        )

    @staticmethod
    def _is_on_leading_side(normalized_center_x: float, track: Track) -> bool:
        """Reject side-visible rear numbers while preserving frontal views."""

        if len(track.trajectory) < 3:
            return True
        recent = track.trajectory[-min(5, len(track.trajectory)) :]
        dx = recent[-1].x - recent[0].x
        if abs(dx) < max(4.0, track.bbox.width * 0.035):
            return True
        if dx < 0:
            return normalized_center_x <= 0.68
        return normalized_center_x >= 0.32

    def _preprocessing_variants(
        self,
        image: Any,
        *,
        engine: Any | None = None,
        limit: int = 3,
    ) -> list[OcrPrediction]:
        """Confirm a difficult single frame without changing its timestamp."""

        import cv2
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        sharpened = cv2.filter2D(
            gray,
            -1,
            np.asarray(
                ((0, -1, 0), (-1, 5, -1), (0, -1, 0)), dtype="float32"
            ),
        )
        variants = (
            ("grayscale", gray),
            ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)),
            ("sharpened", sharpened),
        )
        predictions: list[OcrPrediction] = []
        selected_engine = engine or self._engine
        for name, variant in variants[: max(0, limit)]:
            try:
                results, _timing = selected_engine(
                    variant,
                    use_det=True,
                    use_cls=False,
                    use_rec=True,
                )
            except Exception:
                logger.debug("RapidOCR preprocessing variant failed", exc_info=True)
                continue
            prediction = self._prediction_from_results(
                results,
                image=variant,
                preprocessing=name,
            )
            if prediction is not None:
                predictions.append(prediction)
        return predictions

    def _recognize_direct_localized_board(
        self,
        image: Any,
        region: CandidateRegion,
    ) -> OcrPrediction | None:
        """Read an already-localized board without the expensive text detector.

        Bright/rectangular candidates are tight board proposals produced by the
        preceding localization stage.  The recognition model can consume them
        directly in roughly tens of milliseconds.  A high threshold and the
        normal temporal consensus keep this shortcut from turning arbitrary
        fairing crops into participant numbers.
        """

        if not (
            region.kind.startswith("front_board_bright_")
            or region.kind.startswith("front_board_rect_")
        ):
            return None
        try:
            results, _timing = self._engine(
                image,
                use_det=False,
                use_cls=False,
                use_rec=True,
            )
        except Exception:
            logger.debug("RapidOCR direct board recognition failed", exc_info=True)
            return None
        candidates: list[tuple[str, float]] = []
        for result in results or ():
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                continue
            text = self._normalize_model_text(str(result[-2]))
            confidence = float(result[-1])
            if text is not None and confidence >= 0.88:
                candidates.append((text, confidence))
        if not candidates:
            return None
        text, confidence = max(candidates, key=lambda item: (len(item[0]), item[1]))
        return OcrPrediction(
            text=text,
            confidence=confidence,
            metadata={
                "engine": "rapidocr_onnxruntime",
                "offline": True,
                "direct_board_path": True,
                "trigger_ready": True,
                "center_x": 0.5,
                "center_y": 0.5,
                "digit_bbox": (0.0, 0.0, 1.0, 1.0),
            },
        )

    @staticmethod
    def _merge_preprocessing_consensus(
        predictions: Sequence[OcrPrediction],
    ) -> OcrPrediction:
        groups: defaultdict[str, list[OcrPrediction]] = defaultdict(list)
        for prediction in predictions:
            groups[prediction.text].append(prediction)
        matching = max(
            groups.values(),
            key=lambda items: (len(items), max(item.confidence for item in items)),
        )
        strongest = max(matching, key=lambda item: item.confidence)
        if len(matching) < 2:
            return max(predictions, key=lambda item: item.confidence)
        mean_confidence = sum(item.confidence for item in matching) / len(matching)
        # Agreement after materially different preprocessing is stronger than
        # either weak reading alone, but remains capped below certainty.
        confidence = min(0.98, max(0.94, mean_confidence + 0.35))
        metadata = dict(strongest.metadata)
        metadata.update(
            {
                "preprocessing_consensus": len(matching),
                "preprocessing_variants": tuple(
                    str(item.metadata.get("preprocessing", "unknown"))
                    for item in matching
                ),
            }
        )
        return OcrPrediction(
            text=strongest.text,
            confidence=confidence,
            alternatives=strongest.alternatives,
            metadata=metadata,
        )

    def _prediction_from_results(
        self,
        results: Any,
        *,
        image: Any,
        preprocessing: str,
    ) -> OcrPrediction | None:
        chunks: list[tuple[float, str, float, list[tuple[float, float]]]] = []
        for result in results or ():
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                continue
            # Reject incomplete text clipped by the candidate crop boundary.
            if not _polygon_inside_margin(
                result[0],
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                margin_ratio=self.edge_margin_ratio,
            ):
                continue
            raw_text, raw_confidence = str(result[-2]), float(result[-1])
            digits = self._normalize_model_text(raw_text)
            if digits is not None and raw_confidence >= self.minimum_confidence:
                polygon = [(float(point[0]), float(point[1])) for point in result[0]]
                center_x = sum(float(point[0]) for point in polygon) / len(polygon)
                chunks.append((center_x, digits, raw_confidence, polygon))
        if not chunks:
            return None
        chunks.sort(key=lambda item: item[0])
        combined = "".join(text for _x, text, _confidence, _polygon in chunks)
        if not 1 <= len(combined) <= 4:
            _x, combined, confidence, combined_polygon = max(
                chunks, key=lambda item: item[2]
            )
        else:
            confidence = sum(value for _x, _text, value, _polygon in chunks) / len(chunks)
            combined_polygon = [
                point for _x, _text, _value, polygon in chunks for point in polygon
            ]
        trigger_ready = _polygon_centered_in_trigger_core(
            combined_polygon,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            tolerance=self.trigger_center_tolerance,
        )
        center_x = (
            min(point[0] for point in combined_polygon)
            + max(point[0] for point in combined_polygon)
        ) / (2 * int(image.shape[1]))
        center_y = (
            min(point[1] for point in combined_polygon)
            + max(point[1] for point in combined_polygon)
        ) / (2 * int(image.shape[0]))
        digit_bbox = (
            min(point[0] for point in combined_polygon) / int(image.shape[1]),
            min(point[1] for point in combined_polygon) / int(image.shape[0]),
            max(point[0] for point in combined_polygon) / int(image.shape[1]),
            max(point[1] for point in combined_polygon) / int(image.shape[0]),
        )
        return OcrPrediction(
            text=combined,
            confidence=confidence,
            metadata={
                "engine": "rapidocr_onnxruntime",
                "offline": True,
                "preprocessing": preprocessing,
                "trigger_ready": trigger_ready,
                "center_x": center_x,
                "center_y": center_y,
                "digit_bbox": digit_bbox,
            },
        )

    def _recognize_fast_digit_group(self, image: Any) -> tuple[OcrPrediction | None, bool]:
        """Recognize one contour-group crop without running the slower text detector."""

        crop, trigger_ready, digit_group_seen, center, digit_bbox = (
            self._fast_digit_group_crop(image)
        )
        if crop is None:
            return None, digit_group_seen
        try:
            results, _timing = self._engine(
                crop,
                use_det=False,
                use_cls=False,
                use_rec=True,
            )
        except Exception:
            logger.debug("RapidOCR fast recognition path failed", exc_info=True)
            return None, False
        predictions: list[tuple[str, float]] = []
        for result in results or ():
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                continue
            digits = self._normalize_model_text(str(result[-2]))
            confidence = float(result[-1])
            if digits is not None and confidence >= self.fast_minimum_confidence:
                predictions.append((digits, confidence))
        if not predictions:
            return None, False
        digits, confidence = max(predictions, key=lambda item: (len(item[0]), item[1]))
        return (
            OcrPrediction(
                text=digits,
                confidence=confidence,
                metadata={
                    "engine": "rapidocr_onnxruntime",
                    "offline": True,
                    "fast_path": True,
                    "trigger_ready": trigger_ready,
                    "center_x": center[0] if center is not None else None,
                    "center_y": center[1] if center is not None else None,
                    "digit_bbox": digit_bbox,
                },
            ),
            False,
        )

    def _fast_digit_group_crop(
        self, image: Any
    ) -> tuple[
        Any | None,
        bool,
        bool,
        tuple[float, float] | None,
        tuple[float, float, float, float] | None,
    ]:
        """Find an aligned 1-4 digit group using cheap OpenCV contours."""

        import cv2

        if len(image.shape) < 2:
            return None, False, False, None, None
        height, width = int(image.shape[0]), int(image.shape[1])
        if height < 24 or width < 24:
            return None, False, False, None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binaries = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
        ]
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        digit_group_seen = False
        for binary in binaries:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes: list[tuple[int, int, int, int]] = []
            for contour in contours:
                x, y, box_width, box_height = cv2.boundingRect(contour)
                aspect = box_width / max(1, box_height)
                contour_area = float(cv2.contourArea(contour))
                if not max(12, height * 0.10) <= box_height <= height * 0.82:
                    continue
                if not 0.035 <= aspect <= 1.25:
                    continue
                if contour_area < max(6.0, box_width * box_height * 0.025):
                    continue
                boxes.append((x, y, box_width, box_height))
            boxes.sort(key=lambda item: item[0])
            start = 0
            while start < len(boxes):
                group = [boxes[start]]
                self._append_aligned_digit_boxes(group, boxes[start + 1 :])
                start += len(group)
                x1 = min(item[0] for item in group)
                y1 = min(item[1] for item in group)
                x2 = max(item[0] + item[2] for item in group)
                y2 = max(item[1] + item[3] for item in group)
                digit_group_seen = True
                margin_x = max(4.0, width * self.edge_margin_ratio)
                margin_y = max(4.0, height * self.edge_margin_ratio)
                if x1 < margin_x or x2 > width - margin_x:
                    continue
                if y1 < margin_y or y2 > height - margin_y:
                    continue
                average_height = sum(item[3] for item in group) / len(group)
                vertical_spread = max(
                    abs((item[1] + item[3] / 2) - (y1 + y2) / 2) for item in group
                )
                score = len(group) * 10 + average_height / height - vertical_spread / height
                candidates.append((score, (x1, y1, x2, y2)))
        if not candidates:
            return None, False, digit_group_seen, None, None
        _score, (x1, y1, x2, y2) = max(candidates, key=lambda item: item[0])
        padding = max(3, int((y2 - y1) * 0.60))
        trigger_ready = _box_centered_in_trigger_core(
            x1,
            y1,
            x2,
            y2,
            width=width,
            height=height,
            tolerance=self.trigger_center_tolerance,
        )
        return (
            image[
                max(0, y1 - padding) : min(height, y2 + padding),
                max(0, x1 - padding) : min(width, x2 + padding),
            ],
            trigger_ready,
            False,
            ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height)),
            (x1 / width, y1 / height, x2 / width, y2 / height),
        )

    @staticmethod
    def _upscale_small_board(image: Any) -> Any:
        """Give tiny racing-board digits enough pixels for the fixed OCR recognizer."""

        import cv2

        height, width = int(image.shape[0]), int(image.shape[1])
        shortest = min(height, width)
        if shortest >= 112:
            return image
        scale = min(6.0, 112.0 / max(1, shortest))
        return cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _append_aligned_digit_boxes(
        group: list[tuple[int, int, int, int]],
        remaining: Sequence[tuple[int, int, int, int]],
    ) -> None:
        for candidate in remaining:
            if len(group) >= 4:
                return
            previous = group[-1]
            previous_height = previous[3]
            candidate_height = candidate[3]
            height_ratio = candidate_height / max(1, previous_height)
            vertical_distance = abs(
                (candidate[1] + candidate_height / 2)
                - (previous[1] + previous_height / 2)
            )
            gap = candidate[0] - (previous[0] + previous[2])
            maximum_height = max(previous_height, candidate_height)
            if gap > maximum_height * 1.4:
                return
            if (
                gap >= -maximum_height * 0.15
                and 0.45 <= height_ratio <= 2.2
                and vertical_distance <= maximum_height * 0.42
            ):
                group.append(candidate)

    @staticmethod
    def _normalize_model_text(raw_text: str) -> str | None:
        compact = "".join(
            character for character in raw_text.upper() if character.isalnum()
        )
        substitutions = str.maketrans(
            {
                "O": "0",
                "Q": "0",
                "I": "1",
                "L": "1",
                "Z": "2",
                "S": "5",
                "A": "8",
                "B": "8",
            }
        )
        if compact and all(character in "0123456789OQILZSAB" for character in compact):
            return normalize_racing_number(
                compact.translate(substitutions),
                minimum_length=1,
                maximum_length=4,
            )
        # Do not silently turn an unrelated mixed token such as ``A5`` into
        # racing number ``5``.  Only a pure digit token or the conservative OCR
        # look-alike substitutions above is accepted.
        return None


class PaddleOcrV6DigitEngine(OcrEngine):
    """Accurate local PP-OCRv6 recognition for pre-localized number boards."""

    MODEL_NAME = "PP-OCRv6_medium_rec"
    DETECTOR_MODEL_NAME = "PP-OCRv5_mobile_det"
    MODEL_VERSION = "PaddleOCR 3.7.0"

    def __init__(
        self,
        *,
        cache_dir: str | Path = "models/paddlex",
        minimum_confidence: float = 0.35,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")
        cache_path = Path(cache_dir).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_path))
        model_dir = cache_path / "official_models" / f"{self.MODEL_NAME}_onnx"
        detector_model_dir = (
            cache_path / "official_models" / f"{self.DETECTOR_MODEL_NAME}_onnx"
        )
        if model_dir.is_dir() and detector_model_dir.is_dir():
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            from paddleocr import (  # type: ignore[import-untyped]
                TextDetection,
                TextRecognition,
            )

            profile = hardware_profile()
            device = (
                "gpu:0"
                if profile.preferred_ort_provider == "CUDAExecutionProvider"
                else "cpu"
            )
            kwargs: dict[str, Any] = {
                "model_name": self.MODEL_NAME,
                "engine": "onnxruntime",
                "device": device,
            }
            if model_dir.is_dir():
                kwargs["model_dir"] = str(model_dir)
            self._engine = TextRecognition(**kwargs)
            detector_kwargs: dict[str, Any] = {
                "model_name": self.DETECTOR_MODEL_NAME,
                "engine": "onnxruntime",
                "device": device,
            }
            if detector_model_dir.is_dir():
                detector_kwargs["model_dir"] = str(detector_model_dir)
            self._detector = TextDetection(**detector_kwargs)
        except Exception as exc:
            raise ModelLoadError(f"Could not initialize PP-OCRv6 digit OCR: {exc}") from exc
        self.minimum_confidence = minimum_confidence
        self.cache_dir = cache_path
        self.model_dir = model_dir
        self.detector_model_dir = detector_model_dir
        self.device = device
        self._warm_up()

    def _warm_up(self) -> None:
        try:
            import cv2
            import numpy as np

            image = np.full((96, 192, 3), 255, dtype=np.uint8)
            cv2.putText(
                image,
                "007",
                (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            list(self._engine.predict(image, batch_size=1))
            list(self._detector.predict(image, batch_size=1))
        except Exception:
            logger.warning("PP-OCRv6 warm-up was not completed", exc_info=True)

    def recognize(
        self,
        region: CandidateRegion,
        *,
        frame: Frame,
        track: Track,
    ) -> list[OcrPrediction]:
        del frame, track
        image = region.image
        if image is None or not hasattr(image, "shape") or image.size == 0:
            return []
        prepared = self._prepare_image(image)
        try:
            recognition_inputs = [(prepared, 1.0, False)]
            detection_results = list(self._detector.predict(image, batch_size=1))
            if detection_results:
                payload = self._result_payload(detection_results[0])
                polygons = payload.get("dt_polys", ())
                scores = payload.get("dt_scores", ())
                for polygon, score_value in zip(polygons, scores, strict=False):
                    detection_score = float(score_value)
                    if detection_score < 0.45:
                        continue
                    crop = self._perspective_crop(image, polygon)
                    if crop is None or getattr(crop, "size", 0) == 0:
                        continue
                    recognition_inputs.append(
                        (self._prepare_image(crop), detection_score, True)
                    )
        except Exception as exc:
            raise InferenceError(f"PP-OCRv6 digit recognition failed: {exc}") from exc
        predictions: list[OcrPrediction] = []
        for recognition_input, detection_score, localized in recognition_inputs:
            try:
                results = list(self._engine.predict(recognition_input, batch_size=1))
            except Exception as exc:
                raise InferenceError(
                    f"PP-OCRv6 digit recognition failed: {exc}"
                ) from exc
            for result in results:
                values = self._result_payload(result)
                raw_text = str(values.get("rec_text", "")).strip().replace(" ", "")
                # Recognition is deliberately digits-only. Do not turn logos or
                # letter tokens into numbers by deleting their non-digit symbols.
                if not raw_text or not raw_text.isascii() or not raw_text.isdigit():
                    continue
                text = normalize_racing_number(
                    raw_text,
                    minimum_length=1,
                    maximum_length=4,
                )
                model_confidence = float(values.get("rec_score", 0.0))
                confidence = model_confidence * (0.92 + 0.08 * detection_score)
                if text is None or confidence < self.minimum_confidence:
                    continue
                predictions.append(
                    OcrPrediction(
                        text=text,
                        confidence=min(1.0, max(0.0, confidence)),
                        metadata={
                            "engine": "paddleocr_ppocrv6_medium_rec",
                            "text_detector": self.DETECTOR_MODEL_NAME,
                            "text_region_localized": localized,
                            "offline": True,
                            "digits_only": True,
                            "trigger_ready": True,
                            "digit_bbox": (0.0, 0.0, 1.0, 1.0),
                        },
                    )
                )
        strongest: dict[str, OcrPrediction] = {}
        for prediction in predictions:
            previous = strongest.get(prediction.text)
            if previous is None or prediction.confidence > previous.confidence:
                strongest[prediction.text] = prediction
        return sorted(
            strongest.values(),
            key=lambda item: (item.confidence, len(item.text)),
            reverse=True,
        )

    @staticmethod
    def _result_payload(result: Any) -> Mapping[str, Any]:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if not isinstance(payload, Mapping):
            return {}
        values = payload.get("res", payload)
        return values if isinstance(values, Mapping) else {}

    @staticmethod
    def _perspective_crop(image: Any, polygon: Any) -> Any | None:
        """Rectify one Paddle text polygon before recognition."""

        try:
            import cv2
            import numpy as np

            points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
            width = max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
            height = max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
            target_width = max(1, round(float(width)))
            target_height = max(1, round(float(height)))
            destination = np.asarray(
                [
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1],
                ],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(points, destination)
            return cv2.warpPerspective(
                image,
                transform,
                (target_width, target_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception:
            logger.debug("Could not rectify PaddleOCR text polygon", exc_info=True)
            return None

    @staticmethod
    def _prepare_image(image: Any) -> Any:
        """Upscale tiny moving-board crops without inventing extra detail."""

        import cv2

        height, width = int(image.shape[0]), int(image.shape[1])
        scale = max(1.0, 96.0 / max(1, height), 180.0 / max(1, width))
        scale = min(8.0, scale)
        if scale <= 1.05:
            return image
        return cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )


class HybridDigitOcrEngine(OcrEngine):
    """Use lightweight OCR online and PP-OCRv6 on selected buffered frames."""

    def __init__(self, fast_engine: OcrEngine, accurate_engine: OcrEngine) -> None:
        self.fast_engine = fast_engine
        self.accurate_engine = accurate_engine

    def recognize(
        self,
        region: CandidateRegion,
        *,
        frame: Frame,
        track: Track,
    ) -> list[OcrPrediction]:
        # During deferred uploaded-video recovery, only the diverse PP-OCRv6
        # subset is worth the comparatively expensive RapidOCR full pass.
        # Other retained geometric hypotheses remain useful for selection but
        # must not multiply OCR latency after the motorcycle has left.
        if frame.metadata.get("force_full_ocr", False) and not frame.metadata.get(
            "v6_recovery_ocr", False
        ):
            return []
        predictions = list(
            self.fast_engine.recognize(region, frame=frame, track=track)
        )
        if frame.metadata.get("v6_recovery_ocr", False):
            predictions.extend(
                self.accurate_engine.recognize(region, frame=frame, track=track)
            )
        strongest: dict[tuple[str, str], OcrPrediction] = {}
        for prediction in predictions:
            engine = str(prediction.metadata.get("engine", "unknown"))
            key = (prediction.text, engine)
            previous = strongest.get(key)
            if previous is None or prediction.confidence > previous.confidence:
                strongest[key] = prediction
        return sorted(
            strongest.values(),
            key=lambda item: (item.confidence, len(item.text)),
            reverse=True,
        )

    def close(self) -> None:
        self.fast_engine.close()
        self.accurate_engine.close()


def _polygon_inside_margin(
    polygon: Any,
    *,
    width: int,
    height: int,
    margin_ratio: float,
) -> bool:
    """Return true only when a detected text box is wholly inside the OCR crop."""

    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return False
    if len(points) < 4 or width <= 0 or height <= 0:
        return False
    margin_x = max(3.0, width * margin_ratio)
    margin_y = max(3.0, height * margin_ratio)
    return (
        min(point[0] for point in points) >= margin_x
        and max(point[0] for point in points) <= width - margin_x
        and min(point[1] for point in points) >= margin_y
        and max(point[1] for point in points) <= height - margin_y
    )


def _polygon_centered_in_trigger_core(
    polygon: Any,
    *,
    width: int,
    height: int,
    tolerance: float,
) -> bool:
    """Require detected text to cross the repeatable center of the operator zone."""

    try:
        points = [(float(point[0]), float(point[1])) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return False
    if not points:
        return False
    return _box_centered_in_trigger_core(
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
        width=width,
        height=height,
        tolerance=tolerance,
    )


def _box_centered_in_trigger_core(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    width: int,
    height: int,
    tolerance: float,
) -> bool:
    if width <= 0 or height <= 0:
        return False
    center_x = (x1 + x2) / (2 * width)
    center_y = (y1 + y2) / (2 * height)
    return (
        0.5 - tolerance <= center_x <= 0.5 + tolerance
        and 0.5 - tolerance <= center_y <= 0.5 + tolerance
    )
