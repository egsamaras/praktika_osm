"""LLM output models.

Flat, enum-only, ``max_length`` on strings, no recursion: these are the JSON schemas handed to
Ollama's ``format`` and vLLM's ``json_schema`` constrained decoding. Every citation is a segment
id string; times are resolved later by the pipeline from the transcript.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from praktika.models.meeting import Classification

DecisionKind = Literal["approved", "noted", "deferred", "rejected", "agreed_in_principle"]
OwnerConfidence = Literal["explicit", "inferred", "unknown"]
Severity = Literal["high", "medium", "low"]
IdentifierKind = Literal[
    "iban", "card", "cpr", "iqama", "phone", "email", "account", "amount_with_name"
]


def _none_to_empty(value: object) -> object:
    """Optional draft strings are declared as plain ``str`` (``""`` when absent) rather than
    ``str | None``: under grammar-constrained decoding small local models pick ``null`` for
    ``anyOf`` fields almost every time (observed with qwen2.5:14b on Ollama: owner and due_text
    came back null although the transcript stated both), while a required string field is
    filled. Playback fixtures and older callers may still pass ``None``; it is accepted and
    normalised here, and ``assemble`` turns ``""`` back into ``None`` for the minutes models."""
    return "" if value is None else value


class DecisionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(max_length=400)
    kind: DecisionKind
    decided_by: str = Field(max_length=120)
    dissent_or_conditions: str = Field(default="", max_length=400)
    refs: list[str] = Field(min_length=1, max_length=6)
    quote: str = Field(max_length=240)

    @field_validator("dissent_or_conditions", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        return _none_to_empty(value)


class ActionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(max_length=400)
    #: Declared before ``owner`` on purpose: constrained decoding emits fields in schema order,
    #: so the model writes down the line that names the person (a chair's summary, "thanks
    #: Omar") before it has to commit to a name. Not copied into the minutes.
    owner_evidence: str = Field(default="", max_length=240)
    owner: str = Field(default="", max_length=120)
    owner_confidence: OwnerConfidence
    due_text: str = Field(default="", max_length=120)
    refs: list[str] = Field(min_length=1, max_length=6)
    quote: str = Field(max_length=240)

    @field_validator("owner_evidence", "owner", "due_text", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        return _none_to_empty(value)


class QuestionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(max_length=400)
    raised_by: str = Field(default="", max_length=120)
    owner: str = Field(default="", max_length=120)
    refs: list[str] = Field(max_length=6)

    @field_validator("raised_by", "owner", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        return _none_to_empty(value)


class RiskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(max_length=400)
    severity: Severity
    owner: str = Field(default="", max_length=120)
    mitigation: str = Field(default="", max_length=400)
    refs: list[str] = Field(max_length=6)

    @field_validator("owner", "mitigation", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> object:
        return _none_to_empty(value)


class KeyPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(max_length=120)
    point: str = Field(max_length=300)
    refs: list[str] = Field(max_length=6)


class Figure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(max_length=60)
    context: str = Field(max_length=200)
    refs: list[str] = Field(max_length=6)


class ChunkFindings(BaseModel):
    """Map-stage output for one transcript chunk."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[DecisionDraft] = []
    actions: list[ActionDraft] = []
    questions: list[QuestionDraft] = []
    risks: list[RiskDraft] = []
    key_points: list[KeyPoint] = []
    figures: list[Figure] = []


class MergedFindings(ChunkFindings):
    """Reduce output: deduplicated, ordered by first ref, owner/date conflicts resolved."""

    retracted_decisions: list[DecisionDraft] = []


class TopicDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=120)
    summary: str = Field(max_length=1200)
    key_points: list[str] = Field(max_length=8)
    refs: list[str] = Field(max_length=12)


class Narrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(max_length=900)
    topics: list[TopicDraft] = Field(max_length=12)


class RetractionVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retracted: bool
    refs: list[str] = Field(default=[], max_length=6)
    note: str = Field(max_length=300)


class ClassificationSuggestion(BaseModel):
    """Advisory only; the organiser sets the classification at the gate."""

    model_config = ConfigDict(extra="forbid")

    suggested: Classification
    reasons: list[str] = Field(max_length=5)
    identifiers_seen: list[IdentifierKind]
    mnpi_keywords: list[str] = Field(max_length=20)
