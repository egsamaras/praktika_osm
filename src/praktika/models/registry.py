"""Model register entries persisted in ``models.yaml`` (control C-13/C-16)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ModelRecord(BaseModel):
    """One mirrored model: HF repo and revision, licence, per-file SHA-256 and local path."""

    model_config = ConfigDict(extra="forbid")

    role: str
    repo: str
    revision: str
    licence: str
    files_sha256: dict[str, str]
    local_path: Path
    conversion: str | None = None
    pulled_at: datetime


class ModelRegister(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelRecord]

    def by_role(self) -> dict[str, ModelRecord]:
        """Records keyed by role; the last record for a role wins."""
        return {m.role: m for m in self.models}
