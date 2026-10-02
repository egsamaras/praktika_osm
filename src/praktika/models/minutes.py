"""Minutes models.

``Ref`` carries resolved times and a verbatim quote; ``Flag`` carries reviewer work including the
full text of any item the verifier removed (so a real decision the model failed to cite can be
restored). ``TemplateSpec`` describes a minutes template; the registry itself lives in
``llm/prompts.py``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from praktika.models.findings import DecisionKind, Figure, OwnerConfidence, Severity
from praktika.models.meeting import Attendee, Classification, MeetingType

FlagKind = Literal[
    "uncited_item_removed",
    "name_to_verify",
    "number_to_verify",
    "low_confidence_audio",
    "unresolved_speaker",
    "possible_mnpi",
    "contradiction",
    "personal_remark",
    "identifier_detected",
]


class Ref(BaseModel):
    """A citation resolved against the transcript."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=r"^S\d{4,5}$")
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    speaker: str
    quote: str = Field(max_length=240)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    kind: DecisionKind
    decided_by: str
    dissent_or_conditions: str | None = None
    refs: list[Ref] = Field(min_length=1)


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    owner: str | None
    owner_confidence: OwnerConfidence
    due_date: date | None
    due_text: str | None
    source_language: Literal["en", "ar", "mixed"]
    refs: list[Ref] = Field(min_length=1)


class OpenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    raised_by: str | None
    owner: str | None
    refs: list[Ref]


class Risk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    severity: Severity
    owner: str | None
    mitigation: str | None
    refs: list[Ref]


class TopicSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    key_points: list[str]
    refs: list[Ref]


class Flag(BaseModel):
    """A reviewer flag. ``priority == 1`` blocks approval until cleared."""

    model_config = ConfigDict(extra="forbid")

    kind: FlagKind
    detail: str
    item_json: str | None = None
    refs: list[Ref] = []
    priority: int = Field(default=2, ge=1, le=3)
    cleared_by: str | None = None
    cleared_at: datetime | None = None

    @property
    def cleared(self) -> bool:
        return self.cleared_by is not None


class Provenance(BaseModel):
    """Everything needed to reproduce a minutes version (C-13)."""

    model_config = ConfigDict(extra="forbid")

    generator_model: str
    model_digest: str
    prompt_version: str
    prompt_sha256: str
    glossary_sha256: str
    template: str
    git_sha: str
    transcript_sha256: str
    stt_engines: dict[str, str]
    model_hashes: dict[str, str]
    degraded: bool = False
    generated_at: datetime


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    action: Literal["accept", "modify", "reject", "restore"]
    reason_code: str
    before: str | None
    after: str | None
    by: str
    at: datetime


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["draft", "in_review", "approved", "discarded"] = "draft"
    reviewer: str | None = None
    reviewer_source: str | None = None
    reviewed_at: datetime | None = None
    items: list[ReviewItem] = []


class Minutes(BaseModel):
    """A versioned minutes record for one meeting."""

    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    version: int = Field(default=1, ge=1)
    title: str
    meeting_type: MeetingType
    date: date
    attendees: list[Attendee]
    language_profile: dict[str, float]
    summary: str = Field(max_length=900)
    topics: list[TopicSummary]
    decisions: list[Decision]
    actions: list[ActionItem]
    open_questions: list[OpenQuestion]
    risks: list[Risk]
    follow_ups: list[str]
    flags: list[Flag]
    classification: Classification
    provenance: Provenance
    review: Review = Review()

    def blocking_flags(self) -> list[Flag]:
        """Flags with ``priority == 1`` that have not been cleared; non-empty blocks approval."""
        return [f for f in self.flags if f.priority == 1 and not f.cleared]


class AgendaItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_no: int = Field(ge=1)
    title: str
    paper_ref: str | None = None
    presenter: str | None = None


class MatterArising(BaseModel):
    model_config = ConfigDict(extra="forbid")

    previous_action_id: str
    status: Literal["closed", "open", "overdue"]
    note: str
    refs: list[Ref]


class MancomMinutes(Minutes):
    agenda: list[AgendaItem] = []
    matters_arising: list[MatterArising] = []
    figures_mentioned: list[Figure] = []
    escalations_to_board: list[str] = []


class OneToOneMinutes(Minutes):
    private: bool = True
    my_commitments: list[ActionItem] = []
    their_commitments: list[ActionItem] = []
    next_one_to_one: date | None = None


class TemplateSpec(BaseModel):
    """Describes one minutes template; registry entries live in ``llm/prompts.py::TEMPLATES``."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model: type[Minutes]
    prompt_file: str
    render_template: str
    indexable: bool
    default_private: bool
