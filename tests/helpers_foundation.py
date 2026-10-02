"""Shared builders for the foundation model tests (valid synthetic instances, schema helper)."""

from __future__ import annotations

import copy
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from conftest import FROZEN_NOW, make_transcript
from pydantic import BaseModel

from praktika.models import (
    ActionDraft,
    ActionItem,
    AgendaItem,
    Attendee,
    AuditEvent,
    ChunkFindings,
    Classification,
    ClassificationSuggestion,
    ConsentRecord,
    Decision,
    DecisionDraft,
    Figure,
    Flag,
    KeyPoint,
    LanguageMode,
    MancomMinutes,
    MatterArising,
    Meeting,
    MeetingType,
    MergedFindings,
    Minutes,
    ModelRecord,
    ModelRegister,
    Narrative,
    OneToOneMinutes,
    OpenQuestion,
    Platform,
    Provenance,
    QuestionDraft,
    RawSegment,
    Ref,
    RetractionVerdict,
    Review,
    ReviewItem,
    Risk,
    RiskDraft,
    SpeechChunk,
    TopicDraft,
    TopicSummary,
    Word,
)

NOW = datetime.fromisoformat(FROZEN_NOW)
LINE_RE = re.compile(
    r"^\[(S\d{4,5}) (\d\d:\d\d:\d\d)-(\d\d:\d\d:\d\d) ([^|\]]+)\|(en|ar|mixed|unknown)\] (.*)$"
)


# --------------------------------------------------------------------------- builders


def ref(seg: str = "S0001", quote: str = "Agreed.") -> Ref:
    return Ref(segment_id=seg, start_s=0.0, end_s=9.0, speaker="F. Khalid", quote=quote)


def consent_kwargs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "meeting_id": "M-20260916-a1b2",
        "notified": True,
        "objections": False,
        "method": "spoken",
        "teams_transcription_started": True,
        "script_version": "2026-10-01",
        "purpose": "Minutes of the data team weekly meeting",
        "scope_checks": {"not_board": True, "not_foreign_hosted": True, "not_hr": True},
        "recorded_by": "f.khalid@acme.test",
        "recorded_by_source": "session",
        "recorded_at": NOW,
    }
    base.update(over)
    return base


def provenance() -> Provenance:
    return Provenance(
        generator_model="qwen2.5:14b",
        model_digest="sha256:fake",
        prompt_version="v1",
        prompt_sha256="a" * 64,
        glossary_sha256="b" * 64,
        template="general",
        git_sha="deadbeef",
        transcript_sha256="c" * 64,
        stt_engines={"stt_en": "fake"},
        model_hashes={},
        generated_at=NOW,
    )


def minutes(flags: list[Flag] | None = None, **over: Any) -> Minutes:
    kwargs: dict[str, Any] = {
        "meeting_id": "M-20260916-a1b2",
        "title": "Data team weekly",
        "meeting_type": MeetingType.general,
        "date": date(2026, 9, 16),
        "attendees": [Attendee(name="F. Khalid", role="Chair")],
        "language_profile": {"en": 0.7, "ar": 0.3},
        "summary": "The pilot was approved.",
        "topics": [
            TopicSummary(title="Pilot", summary="Scope agreed.", key_points=["x"], refs=[ref()])
        ],
        "decisions": [
            Decision(
                id="D1",
                statement="Pilot approved",
                kind="approved",
                decided_by="Committee",
                refs=[ref()],
            )
        ],
        "actions": [
            ActionItem(
                id="A1",
                description="Draft the notice",
                owner="Omar Nasser",
                owner_confidence="explicit",
                due_date=date(2026, 9, 18),
                due_text="by Thursday",
                source_language="en",
                refs=[ref("S0005")],
            )
        ],
        "open_questions": [
            OpenQuestion(id="Q1", question="Keep audio?", raised_by=None, owner=None, refs=[])
        ],
        "risks": [
            Risk(
                id="R1",
                description="Customer names",
                severity="medium",
                owner=None,
                mitigation="redact",
                refs=[],
            )
        ],
        "follow_ups": ["Retention question"],
        "flags": flags or [],
        "classification": Classification.internal,
        "provenance": provenance(),
    }
    kwargs.update(over)
    return Minutes(**kwargs)


def meeting() -> Meeting:
    return Meeting(
        id="M-20260916-a1b2",
        title="Data team weekly",
        meeting_type=MeetingType.general,
        classification=Classification.internal,
        language_mode=LanguageMode.ar_mixed,
        platform=Platform.teams,
        started_at=NOW,
        organiser="f.khalid@acme.test",
        roster=[Attendee(name="F. Khalid", aliases=["فيصل"])],
        room_identities=["AI Lab Meeting Room"],
        tags={"weekly"},
    )


def audit_event(**over: Any) -> AuditEvent:
    kwargs: dict[str, Any] = {
        "ts": NOW,
        "actor": "f.khalid@acme.test",
        "actor_source": "session",
        "event": "consent.recorded",
        "meeting_id": "M-20260916-a1b2",
        "classification": "internal",
        "object": None,
        "detail": {"method": "spoken", "n": 3},
        "prev_hash": "0" * 64,
    }
    kwargs.update(over)
    return AuditEvent(**kwargs)


def sample_instances() -> list[BaseModel]:
    """One valid instance of every public model (TemplateSpec excluded: it holds a type)."""
    seg = make_transcript("mixed", n=3).segments[0]
    raw = RawSegment(start=0.0, end=1.5, text="hello", language="en", confidence=0.8, engine="fake")
    dd = DecisionDraft(
        statement="s", kind="approved", decided_by="Chair", refs=["S0001"], quote="q"
    )
    ad = ActionDraft(
        description="d", owner=None, owner_confidence="unknown", refs=["S0002"], quote="q"
    )
    return [
        Attendee(name="R. Haddad", aliases=["رانيا"], upn="r.haddad@acme.test"),
        ConsentRecord(**consent_kwargs()),
        meeting(),
        SpeechChunk(index=0, track="file", start=0.0, end=28.0),
        Word(start=0.0, end=0.4, text="hi", prob=0.9),
        seg,
        raw,
        make_transcript("ar", n=4),
        dd,
        ad,
        QuestionDraft(question="q?", raised_by="Omar", owner=None, refs=[]),
        RiskDraft(description="r", severity="low", owner=None, mitigation=None, refs=["S0001"]),
        KeyPoint(topic="t", point="p", refs=["S0001"]),
        Figure(value="BHD 250,000", context="budget", refs=["S0009"]),
        ChunkFindings(decisions=[dd], actions=[ad]),
        MergedFindings(decisions=[dd], retracted_decisions=[dd]),
        TopicDraft(title="t", summary="s", key_points=["k"], refs=["S0001"]),
        Narrative(summary="s", topics=[TopicDraft(title="t", summary="s", key_points=[], refs=[])]),
        RetractionVerdict(retracted=True, refs=["S0010"], note="deferred"),
        ClassificationSuggestion(
            suggested=Classification.confidential,
            reasons=["S0009"],
            identifiers_seen=["iban"],
            mnpi_keywords=["provision"],
        ),
        ref(),
        Flag(kind="uncited_item_removed", detail="Full text", item_json="{}", priority=1),
        provenance(),
        ReviewItem(
            item_id="D1",
            action="modify",
            reason_code="wording",
            before="a",
            after="b",
            by="F. Khalid",
            at=NOW,
        ),
        Review(status="approved", reviewer="F. Khalid", reviewer_source="session", reviewed_at=NOW),
        minutes(),
        AgendaItem(item_no=1, title="Budget", paper_ref="MC-12", presenter="R. Haddad"),
        MatterArising(previous_action_id="A-old", status="open", note="n", refs=[ref()]),
        MancomMinutes(
            **minutes().model_dump(),
            agenda=[AgendaItem(item_no=1, title="x")],
            figures_mentioned=[Figure(value="1", context="c", refs=[])],
            escalations_to_board=["e"],
        ),
        OneToOneMinutes(**minutes().model_dump(), next_one_to_one=date(2026, 9, 23)),
        audit_event().sealed(),
        ModelRegister(
            models=[
                ModelRecord(
                    role="stt_en",
                    repo="mlx-community/whisper-large-v3-turbo",
                    revision="abc",
                    licence="MIT",
                    files_sha256={"weights.npz": "f" * 64},
                    local_path=Path("~/praktika-models/stt_en").expanduser(),
                    pulled_at=NOW,
                )
            ]
        ),
    ]


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return ``schema`` with ``$defs`` inlined; raise on a self-referencing definition."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if name in stack:
                    raise RecursionError(f"recursive schema via {name}")
                return walk(copy.deepcopy(defs[name]), (*stack, name))
            return {k: walk(v, stack) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, stack) for v in node]
        return node

    return walk(schema, ())


def contains_key(node: Any, key: str) -> bool:
    if isinstance(node, dict):
        return key in node or any(contains_key(v, key) for v in node.values())
    return isinstance(node, list) and any(contains_key(v, key) for v in node)
