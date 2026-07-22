from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain import (
    InvalidCorrection,
    LapTimingService,
    PassageCandidate,
    PassageStatus,
)
from app.models import LapRecord, RaceStatus

UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)


def passage(number: str, at_ns: int, key: str) -> PassageCandidate:
    return PassageCandidate(number, at_ns, UTC, 0.98, "track", number, key)


def running_service(session: Session, required_laps: int = 3) -> tuple[LapTimingService, int]:
    service = LapTimingService(session)
    race = service.create_race(
        name="Final", required_laps=required_laps, camera_identifier="synthetic"
    )
    service.start(race.id, 1_000)
    return service, race.id


def test_separate_rows_independent_numbers_and_leading_zeroes(session: Session) -> None:
    service, race_id = running_service(session)

    first = service.record_passage(race_id, passage("007", 2_000, "a"))
    other = service.record_passage(race_id, passage("7", 2_500, "b"))
    second = service.record_passage(race_id, passage("007", 4_000, "c"))

    assert first.status is PassageStatus.RECORDED
    assert other.status is PassageStatus.RECORDED
    assert second.status is PassageStatus.RECORDED
    assert (first.lap.lap_number, first.lap.lap_time_ns) == (1, 1_000)  # type: ignore[union-attr]
    assert (other.lap.lap_number, other.lap.lap_time_ns) == (1, 1_500)  # type: ignore[union-attr]
    assert (second.lap.lap_number, second.lap.lap_time_ns) == (2, 2_000)  # type: ignore[union-attr]
    rows = session.scalars(select(LapRecord).order_by(LapRecord.id)).all()
    assert [row.racing_number for row in rows] == ["007", "7", "007"]


def test_777_123_and_007_are_counted_as_independent_participants(session: Session) -> None:
    service, race_id = running_service(session, required_laps=2)

    for index, number in enumerate(("777", "123", "007"), start=1):
        first = service.record_passage(
            race_id, passage(number, 2_000 + index * 100, f"{number}-1")
        )
        second = service.record_passage(
            race_id, passage(number, 4_000 + index * 100, f"{number}-2")
        )
        assert first.lap is not None and first.lap.lap_number == 1
        assert second.lap is not None and second.lap.lap_number == 2

    rows = session.scalars(select(LapRecord).order_by(LapRecord.id)).all()
    assert {row.racing_number for row in rows} == {"777", "123", "007"}
    assert all(
        [row.lap_number for row in rows if row.racing_number == number] == [1, 2]
        for number in ("777", "123", "007")
    )


def test_pause_is_excluded_and_duplicate_and_finish_are_safe(session: Session) -> None:
    service, race_id = running_service(session, required_laps=2)
    first = service.record_passage(race_id, passage("123", 2_000, "one"))
    assert first.lap is not None and first.lap.lap_time_ns == 1_000
    assert service.pause(race_id, 3_000).status is RaceStatus.PAUSED
    paused = service.record_passage(race_id, passage("123", 4_000, "paused"))
    assert paused.status is PassageStatus.IGNORED
    service.resume(race_id, 8_000)
    decision = service.record_passage(race_id, passage("123", 10_000, "two"))
    assert decision.lap.lap_time_ns == 3_000  # type: ignore[union-attr]
    duplicate = service.record_passage(race_id, passage("123", 11_000, "two"))
    finished = service.record_passage(race_id, passage("123", 12_000, "three"))
    assert duplicate.status is PassageStatus.DUPLICATE
    assert finished.status is PassageStatus.FINISHED


def test_total_time_is_exact_sum_of_all_lap_times(session: Session) -> None:
    service, race_id = running_service(session, required_laps=3)
    decisions = [
        service.record_passage(race_id, passage("103", timestamp, key))
        for timestamp, key in ((2_000, "one"), (5_000, "two"), (9_000, "three"))
    ]
    laps = [decision.lap for decision in decisions]
    assert all(lap is not None for lap in laps)
    assert [lap.lap_time_ns for lap in laps if lap is not None] == [1_000, 3_000, 4_000]
    assert sum(lap.lap_time_ns for lap in laps if lap is not None) == 8_000
    assert laps[-1] is not None and laps[-1].race_elapsed_ns == 8_000


def test_paused_clock_is_rebased_exactly_after_operating_system_restart(
    session: Session,
) -> None:
    service, race_id = running_service(session)
    race = service.get_race(race_id)
    race.monotonic_start_reference_ns = 10_000
    race.total_paused_ns = 1_000
    race.paused_at_monotonic_ns = 16_000
    race.status = RaceStatus.PAUSED

    assert service.rebase_active_clock(race, 500, UTC + timedelta(days=1)) == 5_000
    assert service.elapsed_ns(race, 500) == 5_000
    service.resume(race.id, 1_000)
    assert service.elapsed_ns(race, 2_000) == 6_000


def test_running_clock_rebases_from_utc_without_regressing_laps(session: Session) -> None:
    service, race_id = running_service(session)
    race = service.get_race(race_id)
    race.started_at_utc = UTC
    race.monotonic_start_reference_ns = 999_000_000_000
    race.total_paused_ns = 2_000_000_000

    elapsed = service.rebase_active_clock(
        race,
        500_000_000,
        UTC + timedelta(seconds=10),
    )
    assert elapsed == 8_000_000_000
    assert service.elapsed_ns(race, 500_000_000) == 8_000_000_000


def test_correction_and_delete_recalculate_laps(session: Session) -> None:
    service, race_id = running_service(session, required_laps=5)
    one = service.record_passage(race_id, passage("10", 2_000, "one")).lap
    two = service.record_passage(race_id, passage("10", 5_000, "two")).lap
    three = service.record_passage(race_id, passage("10", 9_000, "three")).lap
    assert one and two and three

    moved = service.correct_lap(race_id, two.id, racing_number="010")
    assert (moved.racing_number, moved.lap_number, moved.lap_time_ns) == ("010", 1, 4_000)
    ten = session.scalars(
        select(LapRecord).where(LapRecord.racing_number == "10").order_by(LapRecord.lap_number)
    ).all()
    assert [(lap.lap_number, lap.lap_time_ns) for lap in ten] == [(1, 1_000), (2, 7_000)]

    service.delete_lap(race_id, one.id)
    session.expire_all()
    remaining = session.scalars(
        select(LapRecord).where(LapRecord.racing_number == "10")
    ).all()
    assert [(lap.lap_number, lap.lap_time_ns) for lap in remaining] == [(1, 8_000)]

    with pytest.raises(InvalidCorrection):
        service.correct_lap(race_id, three.id, lap_number=3)

    moved_lap = service.correct_lap(race_id, three.id, lap_number=1)
    assert moved_lap.lap_number == 1

    four = service.record_passage(race_id, passage("20", 10_000, "four")).lap
    five = service.record_passage(race_id, passage("20", 12_000, "five")).lap
    assert four and five
    service.correct_lap(race_id, five.id, lap_number=1)
    session.refresh(four)
    session.refresh(five)
    assert (five.lap_number, four.lap_number) == (1, 2)
