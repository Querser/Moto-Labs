"""Persist the single adjustable normalized capture zone."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_capture_zone"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "camera_settings",
        sa.Column("capture_zone_x", sa.Float(), nullable=False, server_default="0.30"),
    )
    op.add_column(
        "camera_settings",
        sa.Column("capture_zone_y", sa.Float(), nullable=False, server_default="0.20"),
    )
    op.add_column(
        "camera_settings",
        sa.Column("capture_zone_width", sa.Float(), nullable=False, server_default="0.40"),
    )
    op.add_column(
        "camera_settings",
        sa.Column("capture_zone_height", sa.Float(), nullable=False, server_default="0.60"),
    )


def downgrade() -> None:
    op.drop_column("camera_settings", "capture_zone_height")
    op.drop_column("camera_settings", "capture_zone_width")
    op.drop_column("camera_settings", "capture_zone_y")
    op.drop_column("camera_settings", "capture_zone_x")
