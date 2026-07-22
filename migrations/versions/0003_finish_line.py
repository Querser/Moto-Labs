"""Replace the obsolete capture rectangle with one normalized finish line."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_finish_line"
down_revision: str | None = "0002_capture_zone"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("camera_settings") as batch:
        batch.add_column(
            sa.Column("finish_line_x1", sa.Float(), nullable=False, server_default="0.10")
        )
        batch.add_column(
            sa.Column("finish_line_y1", sa.Float(), nullable=False, server_default="0.68")
        )
        batch.add_column(
            sa.Column("finish_line_x2", sa.Float(), nullable=False, server_default="0.90")
        )
        batch.add_column(
            sa.Column("finish_line_y2", sa.Float(), nullable=False, server_default="0.68")
        )
        batch.drop_column("capture_zone_x")
        batch.drop_column("capture_zone_y")
        batch.drop_column("capture_zone_width")
        batch.drop_column("capture_zone_height")


def downgrade() -> None:
    with op.batch_alter_table("camera_settings") as batch:
        batch.add_column(
            sa.Column("capture_zone_x", sa.Float(), nullable=False, server_default="0.30")
        )
        batch.add_column(
            sa.Column("capture_zone_y", sa.Float(), nullable=False, server_default="0.20")
        )
        batch.add_column(
            sa.Column("capture_zone_width", sa.Float(), nullable=False, server_default="0.40")
        )
        batch.add_column(
            sa.Column("capture_zone_height", sa.Float(), nullable=False, server_default="0.60")
        )
        batch.drop_column("finish_line_x1")
        batch.drop_column("finish_line_y1")
        batch.drop_column("finish_line_x2")
        batch.drop_column("finish_line_y2")
