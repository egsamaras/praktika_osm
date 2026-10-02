"""Hash-chained audit event (control C-08)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class AuditEvent(BaseModel):
    """One audit line. ``hash`` = SHA-256 over canonical JSON of every other field, so a chain of
    events each carrying the previous ``hash`` in ``prev_hash`` is tamper-evident."""

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    actor: str
    actor_source: Literal["session", "local", "oidc", "system"]
    event: str
    meeting_id: str | None
    classification: str | None
    object: str | None
    detail: dict[str, Any] = {}
    model: str | None = None
    prompt_sha: str | None = None
    prev_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        """sha256 over canonical JSON of all fields except hash."""
        payload = json.dumps(
            self.model_dump(mode="json", exclude={"hash"}),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def sealed(self) -> AuditEvent:
        """Return a copy with ``hash`` set to ``compute_hash()``."""
        return self.model_copy(update={"hash": self.compute_hash()})

    def verify_hash(self) -> bool:
        """True when the stored ``hash`` matches the recomputed one."""
        return bool(self.hash) and self.hash == self.compute_hash()
