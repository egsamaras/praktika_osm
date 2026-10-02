"""Domain model contracts: round-trips, validators and Transcript helpers."""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import make_transcript
from helpers_foundation import (
    LINE_RE,
    consent_kwargs,
    meeting,
    sample_instances,
)
from pydantic import BaseModel, ValidationError

import praktika.models as models_pkg
from praktika.models import (
    Attendee,
    Classification,
    ConsentRecord,
    Decision,
    DecisionDraft,
    Flag,
    LanguageMode,
    MancomMinutes,
    Meeting,
    MeetingState,
    Minutes,
    Narrative,
    Platform,
    Segment,
    TemplateSpec,
    Transcript,
)

# --------------------------------------------------------------------------- round-trips


@pytest.mark.parametrize("instance", sample_instances(), ids=lambda m: type(m).__name__)
def test_round_trip_dict_and_json(instance: BaseModel) -> None:
    cls = type(instance)
    assert cls.model_validate(instance.model_dump(mode="json")) == instance
    assert cls.model_validate_json(instance.model_dump_json()) == instance


@pytest.mark.parametrize("instance", sample_instances(), ids=lambda m: type(m).__name__)
def test_extra_fields_forbidden(instance: BaseModel) -> None:
    payload = instance.model_dump(mode="json")
    payload["talk_time"] = 1
    with pytest.raises(ValidationError):
        type(instance).model_validate(payload)


def test_all_exports_importable() -> None:
    assert len(models_pkg.__all__) >= 40
    for name in models_pkg.__all__:
        assert isinstance(getattr(models_pkg, name), type), name


def test_enums_are_str_enums() -> None:
    assert Classification.internal == "internal"
    assert LanguageMode.ar_mixed == "ar-mixed"
    assert json.loads(json.dumps({"s": MeetingState.draft_ready})) == {"s": "draft_ready"}
    assert Platform("hybrid") is Platform.hybrid


# --------------------------------------------------------------------------- validators


def test_consent_record_valid() -> None:
    rec = ConsentRecord(**consent_kwargs())
    assert rec.notified and not rec.objections and all(rec.scope_checks.values())


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"notified": False}, "notified"),
        ({"objections": True}, "objections"),
        ({"scope_checks": {"not_board": True, "not_hr": False}}, "not_hr"),
        ({"scope_checks": {"not_board": False, "not_hr": False}}, "not_board, not_hr"),
        ({"purpose": "too short"}, "purpose"),
        ({"method": "email"}, "method"),
        ({"recorded_by_source": "header"}, "recorded_by_source"),
        ({"skip_gate": True}, "skip_gate"),
    ],
)
def test_consent_record_rejects(override: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError) as exc:
        ConsentRecord(**consent_kwargs(**override))
    assert fragment in str(exc.value)


@pytest.mark.parametrize("bad_id", ["S1", "S123", "S123456", "s0001", "X0001", "S00A1"])
def test_segment_id_pattern(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Segment(
            id=bad_id, start=0, end=1, speaker="ME", speaker_kind="self", language="en", text="x"
        )


@pytest.mark.parametrize(
    "bad_id", ["M-2026091-a1b2", "M-20260916-A1B2", "M-20260916-a1b", "X-20260916-a1b2"]
)
def test_meeting_id_pattern(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Meeting(**{**meeting().model_dump(), "id": bad_id})


def test_decision_requires_at_least_one_ref() -> None:
    with pytest.raises(ValidationError):
        Decision(id="D1", statement="s", kind="approved", decided_by="Chair", refs=[])
    with pytest.raises(ValidationError):
        DecisionDraft(statement="s", kind="approved", decided_by="Chair", refs=[], quote="q")
    with pytest.raises(ValidationError):
        DecisionDraft(
            statement="s", kind="approved", decided_by="Chair", refs=["S0001"] * 7, quote="q"
        )


def test_string_limits_enforced() -> None:
    with pytest.raises(ValidationError):
        DecisionDraft(
            statement="s", kind="approved", decided_by="Chair", refs=["S0001"], quote="q" * 241
        )
    with pytest.raises(ValidationError):
        Narrative(summary="s" * 901, topics=[])
    with pytest.raises(ValidationError):
        Flag(kind="contradiction", detail="d", priority=0)
    with pytest.raises(ValidationError):
        Attendee(name="")


def test_template_spec_holds_model_type() -> None:
    spec = TemplateSpec(
        model=MancomMinutes,
        prompt_file="templates/mancom.md",
        render_template="mancom.md.j2",
        indexable=True,
        default_private=False,
    )
    assert spec.model is MancomMinutes and issubclass(spec.model, Minutes)
    with pytest.raises(ValidationError):
        TemplateSpec(
            model=MancomMinutes,
            prompt_file="x",
            render_template="y",
            indexable=True,
            default_private=False,
            extra=1,
        )


# --------------------------------------------------------------------------- Transcript


def test_render_for_llm_format_and_determinism(roster: list[Attendee]) -> None:
    t = make_transcript("mixed", n=12)
    out = t.render_for_llm(roster)
    again = Transcript.model_validate(t.model_dump()).render_for_llm(roster)
    assert out == again
    lines = out.split("\n")
    assert len(lines) == 12
    for i, line in enumerate(lines):
        m = LINE_RE.match(line)
        assert m, line
        assert m.group(1) == f"S{i + 1:04d}"
    assert lines[0].startswith("[S0001 00:00:00-00:00:09 F. Khalid|")
    assert lines[11].startswith("[S0012 00:01:50-00:01:59 ")
    assert "\n\n" not in out


def test_render_for_llm_resolves_aliases_and_collapses_newlines(roster: list[Attendee]) -> None:
    seg = Segment(
        id="S0001",
        start=3723.0,
        end=3730.5,
        speaker="فيصل",
        speaker_kind="identity",
        language="ar",
        text="سطر أول\nسطر  ثانٍ",
        track="vtt",
    )
    other = seg.model_copy(update={"id": "S0002", "speaker": "SPEAKER_03", "speaker_kind": "label"})
    t = Transcript(meeting_id="M-20260916-a1b2", source="vtt", engines={}, segments=[seg, other])
    out = t.render_for_llm(roster).split("\n")
    assert out[0] == "[S0001 01:02:03-01:02:10 F. Khalid|ar] سطر أول سطر ثانٍ"
    assert out[1].startswith("[S0002 01:02:03-01:02:10 SPEAKER_03|ar] ")
    assert t.render_for_llm([]).split("\n")[0].startswith("[S0001 01:02:03-01:02:10 فيصل|ar]")


def test_transcript_sha256_stable_and_content_sensitive() -> None:
    a, b = make_transcript("en", n=5), make_transcript("en", n=5)
    assert a.sha256() == b.sha256() and len(a.sha256()) == 64
    c = b.model_copy(update={"segments": b.segments[:-1]})
    assert a.sha256() != c.sha256()
    d = b.model_copy(update={"redacted": not b.redacted})
    assert a.sha256() != d.sha256()


def test_transcript_by_id_and_language_profile() -> None:
    t = make_transcript("mixed", n=12)
    idx = t.by_id()
    assert set(idx) == {f"S{i:04d}" for i in range(1, 13)}
    assert idx["S0007"] is t.segments[6]
    profile = t.language_profile()
    assert set(profile) == {"en", "ar", "mixed"}
    assert abs(sum(profile.values()) - 1.0) < 1e-3
    assert profile["en"] == pytest.approx(4 / 12, abs=1e-3)
    empty = Transcript(meeting_id="M-20260916-a1b2", source="vtt", engines={}, segments=[])
    assert empty.language_profile() == {} and empty.by_id() == {}
