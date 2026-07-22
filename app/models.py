"""Minimal SQLAlchemy schema for races, separate lap rows, and camera choice."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RaceStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


def race_status_column() -> SqlEnum:
    return SqlEnum(
        RaceStatus,
        name="race_status",
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda values: [item.value for item in values],
    )


class Base(DeclarativeBase):
    pass


class Race(Base):
    __tablename__ = "races"
    __table_args__ = (
        CheckConstraint("required_laps > 0", name="ck_race_required_laps_positive"),
        CheckConstraint("total_paused_ns >= 0", name="ck_race_total_paused_nonnegative"),
        Index("ix_races_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    required_laps: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[RaceStatus] = mapped_column(
        race_status_column(), default=RaceStatus.DRAFT, nullable=False
    )
    camera_identifier: Mapped[str] = mapped_column(String(100), nullable=False)
    started_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    monotonic_start_reference_ns: Mapped[int | None] = mapped_column(Integer)
    paused_at_monotonic_ns: Mapped[int | None] = mapped_column(Integer)
    total_paused_ns: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_elapsed_ns: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    lap_records: Mapped[list[LapRecord]] = relationship(
        back_populates="race", cascade="all, delete-orphan"
    )


class LapRecord(Base):
    __tablename__ = "lap_records"
    __table_args__ = (
        UniqueConstraint("race_id", "racing_number", "lap_number", name="uq_lap_number"),
        UniqueConstraint("race_id", "idempotency_key", name="uq_lap_idempotency"),
        CheckConstraint("length(racing_number) > 0", name="ck_lap_number_text_nonempty"),
        CheckConstraint("lap_number > 0", name="ck_lap_positive"),
        CheckConstraint("lap_time_ns >= 0", name="ck_lap_time_nonnegative"),
        CheckConstraint("race_elapsed_ns >= 0", name="ck_lap_elapsed_nonnegative"),
        CheckConstraint(
            "recognition_confidence BETWEEN 0.0 AND 1.0",
            name="ck_lap_recognition_confidence",
        ),
        Index("ix_laps_race_recorded", "race_id", "detected_at_utc"),
        Index("ix_laps_race_number_elapsed", "race_id", "racing_number", "race_elapsed_ns"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    race_id: Mapped[int] = mapped_column(
        ForeignKey("races.id", ondelete="CASCADE"), nullable=False
    )
    racing_number: Mapped[str] = mapped_column(String(16), nullable=False)
    lap_number: Mapped[int] = mapped_column(Integer, nullable=False)
    lap_time_ns: Mapped[int] = mapped_column(Integer, nullable=False)
    race_elapsed_ns: Mapped[int] = mapped_column(Integer, nullable=False)
    detected_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    recognition_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    track_id: Mapped[str | None] = mapped_column(String(100))
    raw_recognition: Mapped[str | None] = mapped_column(String(100))
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    race: Mapped[Race] = relationship(back_populates="lap_records")


class CameraSetting(Base):
    __tablename__ = "camera_settings"
    __table_args__ = (CheckConstraint("id = 1", name="ck_single_camera_setting"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    camera_identifier: Mapped[str] = mapped_column(String(100), nullable=False)
    finish_line_x1: Mapped[float] = mapped_column(Float, default=0.10, nullable=False)
    finish_line_y1: Mapped[float] = mapped_column(Float, default=0.68, nullable=False)
    finish_line_x2: Mapped[float] = mapped_column(Float, default=0.90, nullable=False)
    finish_line_y2: Mapped[float] = mapped_column(Float, default=0.68, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


__all__ = ["Base", "CameraSetting", "LapRecord", "Race", "RaceStatus", "utc_now"]
