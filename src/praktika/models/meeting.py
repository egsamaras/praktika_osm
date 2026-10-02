"""Meeting, attendee and consent models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Classification(StrEnum):
    """Information classification; drives retention, indexing and export stamping (C-09)."""

    internal = "internal"
    confidential = "confidential"
    restricted = "restricted"


class MeetingType(StrEnum):
    """Minutes template family. Only these three exist; further families would each need a
    prompt, a minutes model and a render template."""

    general = "general"
    mancom = "mancom"
    one_to_one = "one_to_one"


class LanguageMode(StrEnum):
    """Per-meeting language routing mode set at the gate."""

    en = "en"
    ar_mixed = "ar-mixed"
    auto = "auto"


class Platform(StrEnum):
    teams = "teams"
    in_room = "in_room"
    hybrid = "hybrid"


class MeetingState(StrEnum):
    """Lifecycle state machine; one audit event per transition."""

    created = "created"
    capturing = "capturing"
    transcribing = "transcribing"
    drafting = "drafting"
    draft_ready = "draft_ready"
    in_review = "in_review"
    approved = "approved"
    discarded = "discarded"
    purged = "purged"


class Attendee(BaseModel):
    """A roster entry. ``aliases`` hold Arabic spellings, nicknames and Teams display variants."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    role: str | None = None
    organisation: str = ""
    status: Literal["present", "apologies", "partial", "guest", "secretary", "room"] = "present"
    aliases: list[str] = []
    upn: str | None = None


class ConsentRecord(BaseModel):
    """The organiser's logged attestation from the consent gate (C-03).

    A record can only exist when notice was given, no objections were received and every scope
    check passed; the validator enforces this, so an invalid record cannot be constructed.
    """

    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    notified: bool
    objections: bool
    method: Literal["spoken", "chat", "teams_transcription", "placard"]
    teams_transcription_started: bool
    script_version: str
    purpose: str = Field(min_length=10, max_length=500)
    scope_checks: dict[str, bool]
    recorded_by: str
    recorded_by_source: Literal["session", "local", "oidc"]
    recorded_at: datetime

    @model_validator(mode="after")
    def _valid(self) -> ConsentRecord:
        """notified must be True, objections False, all scope_checks True; else ValueError."""
        if not self.notified:
            raise ValueError("consent record requires notified=True")
        if self.objections:
            raise ValueError("consent record cannot be created when objections were raised")
        failed = sorted(k for k, ok in self.scope_checks.items() if not ok)
        if failed:
            raise ValueError(f"scope checks failed: {', '.join(failed)}")
        return self


class Meeting(BaseModel):
    """A meeting record. ``id`` has the form ``M-YYYYMMDD-xxxx``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^M-\d{8}-[0-9a-f]{4}$")
    title: str = Field(min_length=1, max_length=300)
    meeting_type: MeetingType
    classification: Classification
    language_mode: LanguageMode
    platform: Platform
    started_at: datetime
    ended_at: datetime | None = None
    chair: str | None = None
    organiser: str
    roster: list[Attendee]
    room_identities: list[str] = []
    private: bool = False
    legal_hold: bool = False
    state: MeetingState = MeetingState.created
    tags: set[str] = set()
