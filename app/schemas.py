"""Small validated contracts for the simplified local API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models import RaceStatus


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


def validate_racing_number(value: str) -> str:
    value = value.strip()
    if not value or not value.isascii() or not value.isdigit():
        raise ValueError("racing number must contain ASCII digits only")
    return value


class CameraSelection(ApiModel):
    camera_identifier: str = Field(min_length=1, max_length=100)

    @field_validator("camera_identifier")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("camera identifier cannot be blank")
        return value


class FinishLineUpdate(ApiModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def useful_length(self) -> FinishLineUpdate:
        if (self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2 < 0.05**2:
            raise ValueError("finish line is too short")
        return self


class RaceCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    required_laps: int = Field(ge=1, le=10_000)
    camera_identifier: str = Field(min_length=1, max_length=100)

    @field_validator("name", "camera_identifier")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be blank")
        return value


class RaceUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    required_laps: int | None = Field(default=None, ge=1, le=10_000)
    camera_identifier: str | None = Field(default=None, min_length=1, max_length=100)


class RaceRead(ApiModel):
    id: int
    name: str
    description: str | None
    required_laps: int
    status: RaceStatus
    camera_identifier: str
    started_at_utc: datetime | None
    finished_at_utc: datetime | None
    created_at: datetime
    elapsed_ns: int | None = None


class LapRead(ApiModel):
    id: int
    race_id: int
    racing_number: str
    lap_number: int
    lap_time_ns: int
    race_elapsed_ns: int
    detected_at_utc: datetime
    finished: bool = False


class LapCorrection(ApiModel):
    racing_number: str | None = Field(default=None, min_length=1, max_length=16)
    lap_number: int | None = Field(default=None, ge=1)

    @field_validator("racing_number")
    @classmethod
    def validate_optional_number(cls, value: str | None) -> str | None:
        return None if value is None else validate_racing_number(value)

    @model_validator(mode="after")
    def at_least_one_change(self) -> LapCorrection:
        if self.racing_number is None and self.lap_number is None:
            raise ValueError("at least one correction field is required")
        return self


class LapSort(ApiModel):
    sort_by: Literal["number", "lap", "recorded"] = "recorded"
    direction: Literal["asc", "desc"] = "asc"


__all__ = [
    "CameraSelection",
    "FinishLineUpdate",
    "LapCorrection",
    "LapRead",
    "LapSort",
    "RaceCreate",
    "RaceRead",
    "RaceUpdate",
    "validate_racing_number",
]
