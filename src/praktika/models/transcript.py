"""Transcript models.

A ``Segment`` id (``S0001``) is the only thing the LLM may cite; ``Transcript.render_for_llm``
produces the exact line format the prompts describe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from praktika.models.meeting import Attendee

Language = Literal["en", "ar", "mixed", "unknown"]


class SpeechChunk(BaseModel):
    """A VAD-derived span of speech on one track, in seconds from the start of the file."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    track: Literal["mic", "system", "file"]
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    prob: float | None = None


class Segment(BaseModel):
    """One transcript utterance with a citable id."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^S\d{4,5}$")
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str
    speaker_kind: Literal["identity", "label", "self", "room", "unknown"]
    language: Language
    text: str = Field(max_length=4000)
    confidence: float | None = Field(default=None, ge=0, le=1)
    track: Literal["mic", "system", "file", "vtt", "docx"] = "file"
    engine: str = ""
    words: list[Word] = []


class RawSegment(BaseModel):
    """STT output before ids and speakers are assigned."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    language: Language
    confidence: float | None = None
    words: list[Word] = []
    engine: str


def _hms(seconds: float) -> str:
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _roster_lookup(roster: list[Attendee]) -> dict[str, str]:
    """Map lower-cased names and aliases to the roster's canonical spelling."""
    lookup: dict[str, str] = {}
    for a in roster:
        for key in (a.name, *a.aliases):
            lookup.setdefault(key.strip().lower(), a.name)
    return lookup


class Transcript(BaseModel):
    """Ordered segments for one meeting plus the engines that produced them."""

    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    source: Literal["file", "vtt", "docx", "capture", "graph"]
    engines: dict[str, str]
    segments: list[Segment]
    redacted: bool = False

    def sha256(self) -> str:
        """SHA-256 over the canonical JSON of the whole transcript (sorted keys, UTF-8)."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def by_id(self) -> dict[str, Segment]:
        """Segments keyed by id. Later duplicates (if any) win; ids are expected to be unique."""
        return {s.id: s for s in self.segments}

    def language_profile(self) -> dict[str, float]:
        """Share of speech time per language tag, summing to 1.0; empty when there is no speech."""
        totals: dict[str, float] = {}
        for s in self.segments:
            dur = max(0.0, s.end - s.start)
            totals[s.language] = totals.get(s.language, 0.0) + dur
        grand = sum(totals.values())
        if grand <= 0:
            return {}
        return {lang: round(t / grand, 4) for lang, t in sorted(totals.items())}

    def render_for_llm(self, roster: list[Attendee]) -> str:
        """One line per segment: ``[S0142 00:23:15-00:23:31 F. Khalid|ar] text``. Deterministic.

        Speakers matching a roster name or alias (case-insensitive) are rendered with the roster
        spelling; other speakers (``SPEAKER_01``, ``ME``, ``Room``) are rendered as stored. Newlines
        inside a segment are collapsed to spaces so every segment occupies exactly one line.
        """
        lookup = _roster_lookup(roster)
        lines: list[str] = []
        for s in self.segments:
            speaker = lookup.get(s.speaker.strip().lower(), s.speaker)
            text = " ".join(s.text.split())
            lines.append(f"[{s.id} {_hms(s.start)}-{_hms(s.end)} {speaker}|{s.language}] {text}")
        return "\n".join(lines)
