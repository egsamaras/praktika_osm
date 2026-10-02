"""Diarisation seam (control C-02).

A ``Turn`` carries a time span and an anonymous label and nothing else: there is no field for a
voiceprint or any other speaker vector anywhere in Praktika, and ``extra="forbid"`` means one
cannot be smuggled in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Turn(BaseModel):
    """One speaker turn: ``label`` is an anonymous tag such as ``SPEAKER_00``."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> Turn:
        if self.end < self.start:
            raise ValueError("turn end precedes start")
        return self


class Diarizer(Protocol):
    """Returns labelled turns only; implementations must never persist speaker vectors."""

    def diarize(
        self, wav: Path, *, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> list[Turn]: ...
