"""Create the minimal race and separate lap-record schema."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "races",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("required_laps", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(8), nullable=False, server_default="draft"),
        sa.Column("camera_identifier", sa.String(100), nullable=False),
        sa.Column("started_at_utc", sa.DateTime(timezone=True)),
        sa.Column("finished_at_utc", sa.DateTime(timezone=True)),
        sa.Column("monotonic_start_reference_ns", sa.Integer()),
        sa.Column("paused_at_monotonic_ns", sa.Integer()),
        sa.Column("total_paused_ns", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("final_elapsed_ns", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("required_laps > 0", name="ck_race_required_laps_positive"),
        sa.CheckConstraint(
            "status IN ('draft','running','paused','finished')", name="race_status"
        ),
        sa.CheckConstraint("total_paused_ns >= 0", name="ck_race_total_paused_nonnegative"),
    )
    op.create_index("ix_races_status_created", "races", ["status", "created_at"])
    op.create_table(
        "lap_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "race_id",
            sa.Integer(),
            sa.ForeignKey("races.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("racing_number", sa.String(16), nullable=False),
        sa.Column("lap_number", sa.Integer(), nullable=False),
        sa.Column("lap_time_ns", sa.Integer(), nullable=False),
        sa.Column("race_elapsed_ns", sa.Integer(), nullable=False),
        sa.Column("detected_at_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recognition_confidence", sa.Float(), nullable=False),
        sa.Column("track_id", sa.String(100)),
        sa.Column("raw_recognition", sa.String(100)),
        sa.Column("idempotency_key", sa.String(80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("race_id", "racing_number", "lap_number", name="uq_lap_number"),
        sa.UniqueConstraint("race_id", "idempotency_key", name="uq_lap_idempotency"),
        sa.CheckConstraint("length(racing_number) > 0", name="ck_lap_number_text_nonempty"),
        sa.CheckConstraint("lap_number > 0", name="ck_lap_positive"),
        sa.CheckConstraint("lap_time_ns >= 0", name="ck_lap_time_nonnegative"),
        sa.CheckConstraint("race_elapsed_ns >= 0", name="ck_lap_elapsed_nonnegative"),
        sa.CheckConstraint(
            "recognition_confidence BETWEEN 0.0 AND 1.0",
            name="ck_lap_recognition_confidence",
        ),
    )
    op.create_index("ix_laps_race_recorded", "lap_records", ["race_id", "detected_at_utc"])
    op.create_index(
        "ix_laps_race_number_elapsed",
        "lap_records",
        ["race_id", "racing_number", "race_elapsed_ns"],
    )
    op.create_table(
        "camera_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("camera_identifier", sa.String(100), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_single_camera_setting"),
    )


def downgrade() -> None:
    op.drop_table("camera_settings")
    op.drop_index("ix_laps_race_number_elapsed", table_name="lap_records")
    op.drop_index("ix_laps_race_recorded", table_name="lap_records")
    op.drop_table("lap_records")
    op.drop_index("ix_races_status_created", table_name="races")
    op.drop_table("races")
