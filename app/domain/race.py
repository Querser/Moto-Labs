"""Race lifecycle, monotonic lap timing, corrections, and completion rules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import LapRecord, Race, RaceStatus, utc_now
from app.schemas import validate_racing_number


class RaceError(RuntimeError):
    pass


class RaceNotFound(RaceError):
    pass


class InvalidRaceState(RaceError):
    pass


class InvalidCorrection(RaceError):
    pass


@dataclass(slots=True)
class RaceClock:
    start_ns: int
    total_paused_ns: int = 0
    paused_at_ns: int | None = None

    def elapsed(self, captured_ns: int) -> int:
        if captured_ns < self.start_ns:
            raise RaceError("capture timestamp precedes race start")
        end = self.paused_at_ns if self.paused_at_ns is not None else captured_ns
        return max(0, end - self.start_ns - self.total_paused_ns)


@dataclass(frozen=True, slots=True)
class PassageCandidate:
    racing_number: str
    captured_monotonic_ns: int
    detected_at_utc: datetime
    recognition_confidence: float
    track_id: str | None = None
    raw_recognition: str | None = None
    idempotency_key: str | None = None
    race_elapsed_ns: int | None = None

    def __post_init__(self) -> None:
        validate_racing_number(self.racing_number)
        if self.captured_monotonic_ns < 0:
            raise ValueError("captured_monotonic_ns must be non-negative")
        if self.detected_at_utc.tzinfo is None or self.detected_at_utc.utcoffset() is None:
            raise ValueError("detected_at_utc must include timezone")
        if not 0 <= self.recognition_confidence <= 1:
            raise ValueError("recognition confidence must be in [0, 1]")


class PassageStatus(str, Enum):
    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    FINISHED = "finished"
    IGNORED = "ignored"


@dataclass(frozen=True, slots=True)
class PassageDecision:
    status: PassageStatus
    lap: LapRecord | None
    message: str


class LapTimingService:
    """Transactional service; callers own commit and rollback."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_race(self, race_id: int) -> Race:
        race = self.session.get(Race, race_id)
        if race is None:
            raise RaceNotFound(f"race {race_id} was not found")
        return race

    def create_race(
        self,
        *,
        name: str,
        required_laps: int,
        camera_identifier: str,
        description: str | None = None,
    ) -> Race:
        if not name.strip() or required_laps < 1 or not camera_identifier.strip():
            raise RaceError("race name, lap count, and camera are required")
        race = Race(
            name=name.strip(),
            description=description.strip() if description else None,
            required_laps=required_laps,
            camera_identifier=camera_identifier.strip(),
        )
        self.session.add(race)
        self.session.flush()
        return race

    def update_draft(self, race_id: int, **values: object) -> Race:
        race = self.get_race(race_id)
        if race.status is not RaceStatus.DRAFT:
            raise InvalidRaceState("race setup cannot change after start")
        for key in ("name", "description", "required_laps", "camera_identifier"):
            if key in values:
                setattr(race, key, values[key])
        self.session.flush()
        return race

    def start(self, race_id: int, now_ns: int, started_at_utc: datetime | None = None) -> Race:
        race = self.get_race(race_id)
        if race.status is not RaceStatus.DRAFT:
            raise InvalidRaceState("only a draft race can be started")
        race.status = RaceStatus.RUNNING
        race.monotonic_start_reference_ns = now_ns
        race.started_at_utc = _utc(started_at_utc or utc_now())
        race.paused_at_monotonic_ns = None
        race.total_paused_ns = 0
        race.final_elapsed_ns = None
        self.session.flush()
        return race

    def finish_other_active_races(self, race_id: int, now_ns: int) -> list[Race]:
        """Finish older active races before a new race starts.

        The application intentionally permits many persisted races, but only one
        of them may own the live camera/vision runtime.  Keeping this invariant in
        the domain service prevents a browser refresh or a second client from
        leaving two races marked as active in the database.
        """
        active_races = list(
            self.session.scalars(
                select(Race)
                .where(
                    Race.id != race_id,
                    Race.status.in_((RaceStatus.RUNNING, RaceStatus.PAUSED)),
                )
                .order_by(Race.id)
            )
        )
        for active_race in active_races:
            self.finish(active_race.id, now_ns)
        return active_races

    def pause(self, race_id: int, now_ns: int) -> Race:
        race = self.get_race(race_id)
        if race.status is not RaceStatus.RUNNING:
            raise InvalidRaceState("only a running race can be paused")
        race.status = RaceStatus.PAUSED
        race.paused_at_monotonic_ns = now_ns
        self.session.flush()
        return race

    def resume(self, race_id: int, now_ns: int) -> Race:
        race = self.get_race(race_id)
        if race.status is not RaceStatus.PAUSED or race.paused_at_monotonic_ns is None:
            raise InvalidRaceState("only a paused race can be resumed")
        if now_ns < race.paused_at_monotonic_ns:
            raise RaceError("resume timestamp precedes pause")
        race.total_paused_ns += now_ns - race.paused_at_monotonic_ns
        race.paused_at_monotonic_ns = None
        race.status = RaceStatus.RUNNING
        self.session.flush()
        return race

    def finish(self, race_id: int, now_ns: int, finished_at_utc: datetime | None = None) -> Race:
        race = self.get_race(race_id)
        if race.status not in {RaceStatus.RUNNING, RaceStatus.PAUSED}:
            raise InvalidRaceState("only a running or paused race can be finished")
        race.final_elapsed_ns = self.elapsed_ns(race, now_ns)
        race.status = RaceStatus.FINISHED
        race.finished_at_utc = _utc(finished_at_utc or utc_now())
        race.paused_at_monotonic_ns = None
        self.session.flush()
        return race

    def elapsed_ns(self, race: Race, now_ns: int) -> int | None:
        if race.final_elapsed_ns is not None:
            return race.final_elapsed_ns
        if race.monotonic_start_reference_ns is None:
            return None
        return RaceClock(
            race.monotonic_start_reference_ns,
            race.total_paused_ns,
            race.paused_at_monotonic_ns,
        ).elapsed(now_ns)

    def rebase_active_clock(
        self,
        race: Race,
        now_ns: int,
        now_utc: datetime | None = None,
    ) -> int | None:
        """Move an active clock to the current process monotonic epoch.

        Monotonic values are ideal while a race is running but they cannot be
        compared across an operating-system reboot.  A graceful application
        stop leaves the race paused, so its old paired start/pause references
        retain the exact elapsed duration.  An unexpectedly interrupted running
        race is reconstructed from its UTC start and accumulated pauses.
        """

        if race.status not in {RaceStatus.RUNNING, RaceStatus.PAUSED}:
            return self.elapsed_ns(race, now_ns)
        start_ns = race.monotonic_start_reference_ns
        if start_ns is None:
            return None
        if race.status is RaceStatus.PAUSED and race.paused_at_monotonic_ns is not None:
            elapsed_ns = max(
                0,
                race.paused_at_monotonic_ns - start_ns - race.total_paused_ns,
            )
        elif race.started_at_utc is not None:
            current_utc = _utc(now_utc or utc_now())
            started_utc = _stored_utc(race.started_at_utc)
            wall_elapsed_ns = max(
                0,
                int((current_utc - started_utc).total_seconds() * 1_000_000_000),
            )
            last_lap_elapsed = self.session.scalar(
                select(LapRecord.race_elapsed_ns)
                .where(LapRecord.race_id == race.id)
                .order_by(LapRecord.race_elapsed_ns.desc())
                .limit(1)
            )
            elapsed_ns = max(
                int(last_lap_elapsed or 0),
                wall_elapsed_ns - race.total_paused_ns,
            )
        else:
            elapsed_ns = max(0, now_ns - start_ns - race.total_paused_ns)

        # A negative virtual origin is valid: it represents elapsed race time
        # that predates the current boot, while future arithmetic still uses
        # only the new high-resolution monotonic clock.
        race.monotonic_start_reference_ns = (
            now_ns - elapsed_ns - race.total_paused_ns
        )
        race.paused_at_monotonic_ns = (
            now_ns if race.status is RaceStatus.PAUSED else None
        )
        self.session.flush()
        return elapsed_ns

    def record_passage(self, race_id: int, candidate: PassageCandidate) -> PassageDecision:
        race = self.get_race(race_id)
        if race.status is not RaceStatus.RUNNING:
            return PassageDecision(PassageStatus.IGNORED, None, "race is not running")
        number = validate_racing_number(candidate.racing_number)
        elapsed = (
            candidate.race_elapsed_ns
            if candidate.race_elapsed_ns is not None
            else self.elapsed_ns(race, candidate.captured_monotonic_ns)
        )
        if elapsed is None:
            raise RaceError("race clock has not started")
        key = candidate.idempotency_key or hashlib.sha256(
            f"{race_id}|{number}|{candidate.track_id}|{candidate.captured_monotonic_ns}".encode()
        ).hexdigest()[:40]
        existing = self.session.scalar(
            select(LapRecord).where(
                LapRecord.race_id == race_id,
                LapRecord.idempotency_key == key,
            )
        )
        if existing is not None:
            return PassageDecision(PassageStatus.DUPLICATE, existing, "passage already recorded")
        records = self._number_records(race_id, number)
        if len(records) >= race.required_laps:
            return PassageDecision(PassageStatus.FINISHED, None, "number already finished")
        previous_elapsed = records[-1].race_elapsed_ns if records else 0
        if elapsed <= previous_elapsed:
            return PassageDecision(PassageStatus.DUPLICATE, None, "non-increasing passage time")
        lap = LapRecord(
            race_id=race_id,
            racing_number=number,
            lap_number=len(records) + 1,
            lap_time_ns=elapsed - previous_elapsed,
            race_elapsed_ns=elapsed,
            detected_at_utc=_utc(candidate.detected_at_utc),
            recognition_confidence=candidate.recognition_confidence,
            track_id=candidate.track_id,
            raw_recognition=candidate.raw_recognition,
            idempotency_key=key,
        )
        self.session.add(lap)
        self.session.flush()
        return PassageDecision(PassageStatus.RECORDED, lap, f"lap {lap.lap_number} recorded")

    def delete_lap(self, race_id: int, lap_id: int) -> None:
        lap = self._get_lap(race_id, lap_id)
        number = lap.racing_number
        self.session.delete(lap)
        self.session.flush()
        self._renumber(race_id, number)

    def correct_lap(
        self,
        race_id: int,
        lap_id: int,
        *,
        racing_number: str | None = None,
        lap_number: int | None = None,
    ) -> LapRecord:
        lap = self._get_lap(race_id, lap_id)
        old_number = lap.racing_number
        if racing_number is not None:
            corrected_number = validate_racing_number(racing_number)
            if corrected_number != old_number:
                lap.lap_number = 1_000_000 + lap.id
                self.session.flush()
                lap.racing_number = corrected_number
        self.session.flush()
        self._renumber(race_id, old_number)
        if lap.racing_number != old_number:
            self._renumber(race_id, lap.racing_number)
        self.session.refresh(lap)
        if lap_number is not None and lap_number != lap.lap_number:
            records = self._number_records(race_id, lap.racing_number)
            if not 1 <= lap_number <= len(records):
                raise InvalidCorrection("corrected lap must be within existing lap rows")
            reordered = sorted(records, key=lambda item: (item.lap_number, item.id))
            reordered = [item for item in reordered if item.id != lap.id]
            reordered.insert(lap_number - 1, lap)
            self._assign_order(reordered, recalculate_times=False)
        self.session.flush()
        return lap

    def _get_lap(self, race_id: int, lap_id: int) -> LapRecord:
        lap = self.session.get(LapRecord, lap_id)
        if lap is None or lap.race_id != race_id:
            raise RaceNotFound(f"lap {lap_id} was not found")
        return lap

    def _number_records(self, race_id: int, number: str) -> list[LapRecord]:
        return list(
            self.session.scalars(
                select(LapRecord)
                .where(LapRecord.race_id == race_id, LapRecord.racing_number == number)
                .order_by(LapRecord.race_elapsed_ns, LapRecord.id)
            )
        )

    def _renumber(self, race_id: int, number: str) -> None:
        self._assign_order(self._number_records(race_id, number))

    def _assign_order(
        self, records: list[LapRecord], *, recalculate_times: bool = True
    ) -> None:
        for record in records:
            record.lap_number = 1_000_000 + record.id
        self.session.flush()
        for index, record in enumerate(records, start=1):
            record.lap_number = index
        if recalculate_times:
            previous_elapsed = 0
            for record in records:
                record.lap_time_ns = max(0, record.race_elapsed_ns - previous_elapsed)
                previous_elapsed = record.race_elapsed_ns
        self.session.flush()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RaceError("wall-clock timestamp must include timezone")
    return value.astimezone(timezone.utc)


def _stored_utc(value: datetime) -> datetime:
    """Interpret SQLite's timezone-naive round trip as the stored UTC value."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "InvalidCorrection",
    "InvalidRaceState",
    "LapTimingService",
    "PassageCandidate",
    "PassageDecision",
    "PassageStatus",
    "RaceClock",
    "RaceError",
    "RaceNotFound",
]
