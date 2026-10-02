"""The in-package ``FakeLLM`` (``llm_provider="fake"``) drives the real pipeline and verifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import make_transcript

from praktika.errors import LLMError
from praktika.llm import pipeline
from praktika.llm.fake_client import FakeLLM, findings_from_lines, transcript_lines
from praktika.llm.prompts import PromptSet, load
from praktika.models import (
    Attendee,
    Classification,
    LanguageMode,
    Meeting,
    MeetingType,
    Platform,
)

REPO = Path(__file__).resolve().parent.parent


class _Audit:
    def __init__(self) -> None:
        self.events: list[str] = []

    def append(self, event: str, meeting_id: str | None, **detail: Any) -> None:
        self.events.append(event)


@pytest.fixture(scope="module")
def prompts() -> PromptSet:
    return load(REPO / "prompts", "v1", MeetingType.general)


def _meeting(roster: list[Attendee]) -> Meeting:
    return Meeting(
        id="M-20260916-a1b2",
        title="Data team weekly",
        meeting_type=MeetingType.general,
        classification=Classification.internal,
        language_mode=LanguageMode.ar_mixed,
        platform=Platform.teams,
        started_at="2026-09-16T09:00:00+00:00",
        organiser="f.khalid@acme.test",
        roster=roster,
    )


def _generate(llm: FakeLLM, prompts: PromptSet, roster: list[Attendee], **opts: Any) -> Any:
    transcript = make_transcript("mixed", n=12)
    meeting = _meeting(roster)
    options = pipeline.GenerateOptions(template=MeetingType.general, **opts)
    audit = _Audit()
    minutes = pipeline.generate(transcript, meeting, llm, prompts, "b" * 64, None, options, audit)
    return transcript, minutes, audit


def test_lines_and_findings_follow_the_rendered_transcript() -> None:
    user = (
        "Transcript:\n[S0001 00:00:00-00:00:09 F. Khalid|en] We agreed to start the pilot.\n"
        "[S0002 00:00:10-00:00:19 SPEAKER_01|ar] سنبدأ الأسبوع القادم\n"
    )
    lines = transcript_lines(user)
    assert [(sid, sp) for sid, sp, _, _ in lines] == [
        ("S0001", "F. Khalid"),
        ("S0002", "SPEAKER_01"),
    ]
    found = findings_from_lines(lines)
    assert found["decisions"][0]["refs"] == ["S0001"]
    assert found["decisions"][0]["quote"] == "We agreed to start the pilot."
    assert found["actions"][0]["owner"] is None, "an unmapped speaker label is not an owner"
    assert found["actions"][0]["owner_confidence"] == "unknown"
    assert findings_from_lines([]) == {
        "decisions": [],
        "actions": [],
        "questions": [],
        "risks": [],
        "key_points": [],
        "figures": [],
    }


def test_single_pass_draft_survives_the_verifier(
    prompts: PromptSet, roster: list[Attendee]
) -> None:
    llm = FakeLLM()
    transcript, minutes, audit = _generate(llm, prompts, roster)
    assert llm.calls == 3, "extract, narrative, retraction for one chunk"
    assert len(minutes.decisions) == 1 and len(minutes.actions) == 1
    ref = minutes.decisions[0].refs[0]
    assert ref.segment_id == "S0001" and ref.quote == transcript.segments[0].text
    assert ref.start_s == transcript.segments[0].start
    assert not [f for f in minutes.flags if f.kind == "uncited_item_removed"]
    assert minutes.blocking_flags() == []
    assert "placeholder" in minutes.summary.lower()
    assert minutes.provenance.generator_model.startswith("fake")
    assert audit.events.count("llm.call") == 3 and audit.events[-1] == "minutes.drafted"


def test_map_reduce_path_concatenates_chunks(prompts: PromptSet, roster: list[Attendee]) -> None:
    llm = FakeLLM()
    _, minutes, _ = _generate(llm, prompts, roster, full_context_max_tokens=40, map_chunk_tokens=60)
    assert llm.calls > 3, "several extract calls plus one reduce"
    assert len(minutes.decisions) >= 2, "one decision per chunk survives the merge"
    assert all(d.refs for d in minutes.decisions)
    assert minutes.blocking_flags() == []


def test_unknown_schema_is_an_llm_error() -> None:
    with pytest.raises(LLMError):
        FakeLLM().complete_json("s", "u", {"title": "Nothing"})
    assert FakeLLM().with_model("other").name == "fake"
