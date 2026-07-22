"""Local-only application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MOTO_", extra="ignore", case_sensitive=False
    )

    app_name: str = "Moto Laps"
    app_version: str = "0.8.0"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/moto_laps.db"
    export_dir: Path = Path("data/exports")
    log_dir: Path = Path("data/logs")
    backup_dir: Path = Path("data/backups")
    video_upload_dir: Path = Path("data/uploads")
    recording_dir: Path = Path("data/recordings")
    video_upload_max_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=1_000_000,
        le=4 * 1024 * 1024 * 1024,
    )
    log_max_bytes: int = Field(default=10_000_000, ge=100_000, le=100_000_000)
    log_backup_count: int = Field(default=10, ge=1, le=50)
    database_backup_count: int = Field(default=10, ge=1, le=100)
    frame_queue_size: int = Field(default=2, ge=1, le=16)

    @field_validator("host")
    @classmethod
    def nonempty_host(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("host cannot be blank")
        return value

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported log level")
        return value

    def ensure_directories(self) -> None:
        self.data_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        self.export_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        self.log_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        self.backup_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        self.video_upload_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        self.recording_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
