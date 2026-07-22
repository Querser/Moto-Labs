from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from app.database import create_db_engine, upgrade_database


def test_migration_creates_only_minimal_domain_tables(tmp_path: Path) -> None:
    database = tmp_path / "moto_laps.db"
    url = f"sqlite:///{database.as_posix()}"
    upgrade_database(url)
    engine = create_db_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert tables == {"alembic_version", "races", "lap_records", "camera_settings"}
        camera_columns = {
            column["name"] for column in inspect(engine).get_columns("camera_settings")
        }
        assert {
            "finish_line_x1",
            "finish_line_y1",
            "finish_line_x2",
            "finish_line_y2",
        } <= camera_columns
        assert "capture_zone_x" not in camera_columns
    finally:
        engine.dispose()
