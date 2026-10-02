"""Golden set loader and playback assembler.

A golden meeting is a directory with ``spec.json`` (meeting record, language, traps), a valid
``transcript.json``, ``gold_minutes.json`` (the expected decisions, actions, questions, risks and
figures with segment ids) and ``playback.json`` (canned LLM outputs keyed by schema title for
``FakeLLM`` playback). ``assemble_minutes`` turns playback findings into a ``Minutes`` record
without an LLM so the deterministic scoring can run when the pipeline is unavailable; it only
resolves segment ids (unknown ids are dropped and unsupported decisions/actions become
``uncited_item_removed`` flags), consumes retraction verdicts in the pipeline's order (one per
decision that cites a real segment) and leaves quote, number and name checks to ``llm/verify``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from praktika.models import (
    ActionItem,
    Decision,
    Flag,
    MancomMinutes,
    Meeting,
    MergedFindings,
    Minutes,
    Narrative,
    OneToOneMinutes,
    OpenQuestion,
    Provenance,
    Ref,
    RetractionVerdict,
    Risk,
    TopicSummary,
    Transcript,
)

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "golden"
REQUIRED = ("spec.json", "transcript.json", "gold_minutes.json")
REMOVED_KIND = "uncited_item_removed"


class GoldenMeeting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    spec: dict[str, Any]
    transcript: Transcript
    gold: dict[str, Any]
    meeting: Meeting
    playback: dict[str, list[dict[str, Any]]] = {}

    @property
    def language(self) -> str:
        """``"en"`` or ``"ar-mixed"``; from the spec, else the meeting's language mode."""
        return str(self.spec.get("language") or self.meeting.language_mode.value)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load(directory: Path) -> GoldenMeeting:
    """Load one golden meeting; raises ``FileNotFoundError`` when a required file is missing."""
    for name in REQUIRED:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"{directory / name} is missing")
    spec = _read(directory / "spec.json")
    playback_path = directory / "playback.json"
    return GoldenMeeting(
        name=directory.name,
        spec=spec,
        transcript=Transcript.model_validate(_read(directory / "transcript.json")),
        gold=_read(directory / "gold_minutes.json"),
        meeting=Meeting.model_validate(spec["meeting"]),
        playback=_read(playback_path) if playback_path.is_file() else {},
    )


def load_all(directory: Path = GOLDEN_DIR) -> list[GoldenMeeting]:
    """Every golden meeting under ``directory`` (sorted by name); empty when there are none."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"golden directory not found: {directory}")
    return [load(d) for d in sorted(directory.iterdir()) if (d / "spec.json").is_file()]


def _resolver(transcript: Transcript) -> Any:
    by_id = transcript.by_id()

    def resolve(ids: list[str], quote: str | None) -> list[Ref]:
        refs: list[Ref] = []
        for sid in ids:
            seg = by_id.get(sid)
            if seg is None:
                continue
            refs.append(
                Ref(
                    segment_id=sid,
                    start_s=seg.start,
                    end_s=seg.end,
                    speaker=seg.speaker,
                    quote=(quote or seg.text)[:240],
                )
            )
        return refs

    return resolve, by_id


def fake_provenance(gm: GoldenMeeting, generator: str, now: datetime | None = None) -> Provenance:
    return Provenance(
        generator_model=generator,
        model_digest="sha256:fake",
        prompt_version="v1",
        prompt_sha256="0" * 64,
        glossary_sha256="0" * 64,
        template=gm.meeting.meeting_type.value,
        git_sha="golden",
        transcript_sha256=gm.transcript.sha256(),
        stt_engines=dict(gm.transcript.engines),
        model_hashes={},
        generated_at=now or datetime.now(UTC),
    )


def assemble_minutes(
    gm: GoldenMeeting,
    merged: MergedFindings,
    narrative: Narrative,
    *,
    retractions: list[RetractionVerdict] | None = None,
    generator: str = "fake",
    now: datetime | None = None,
) -> Minutes:
    """Build a ``Minutes`` record of the right class from playback findings (see module doc)."""
    resolve, by_id = _resolver(gm.transcript)
    flags: list[Flag] = []
    decisions: list[Decision] = []
    verdicts = list(retractions or [])  # consumed in order by decisions that cite real segments
    for i, d in enumerate(merged.decisions, 1):
        refs = resolve(d.refs, d.quote)
        if not refs:
            flags.append(
                Flag(
                    kind=REMOVED_KIND, detail=d.statement, item_json=d.model_dump_json(), priority=1
                )
            )
            continue
        decisions.append(
            Decision(
                id=f"D{i}",
                statement=d.statement,
                kind=d.kind,
                decided_by=d.decided_by,
                dissent_or_conditions=d.dissent_or_conditions,
                refs=refs,
            )
        )
        verdict = verdicts.pop(0) if verdicts else None
        if verdict is not None and verdict.retracted:
            flags.append(
                Flag(
                    kind="contradiction",
                    detail=f"{d.statement} — {verdict.note}",
                    refs=resolve(verdict.refs, None),
                    priority=2,
                )
            )
    actions: list[ActionItem] = []
    for i, a in enumerate(merged.actions, 1):
        refs = resolve(a.refs, a.quote)
        if not refs:
            flags.append(
                Flag(
                    kind=REMOVED_KIND,
                    detail=a.description,
                    item_json=a.model_dump_json(),
                    priority=1,
                )
            )
            continue
        lang = by_id[refs[0].segment_id].language
        actions.append(
            ActionItem(
                id=f"A{i}",
                description=a.description,
                owner=a.owner,
                owner_confidence=a.owner_confidence,
                due_date=None,
                due_text=a.due_text,
                source_language="en" if lang == "unknown" else lang,
                refs=refs,
            )
        )
    questions = [
        OpenQuestion(
            id=f"Q{i}",
            question=q.question,
            raised_by=q.raised_by,
            owner=q.owner,
            refs=resolve(q.refs, None),
        )
        for i, q in enumerate(merged.questions, 1)
    ]
    risks = [
        Risk(
            id=f"R{i}",
            description=r.description,
            severity=r.severity,
            owner=r.owner,
            mitigation=r.mitigation,
            refs=resolve(r.refs, None),
        )
        for i, r in enumerate(merged.risks, 1)
    ]
    topics = [
        TopicSummary(
            title=t.title, summary=t.summary, key_points=t.key_points, refs=resolve(t.refs, None)
        )
        for t in narrative.topics
    ]
    base: dict[str, Any] = {
        "meeting_id": gm.meeting.id,
        "title": gm.meeting.title,
        "meeting_type": gm.meeting.meeting_type,
        "date": gm.meeting.started_at.date(),
        "attendees": gm.meeting.roster,
        "language_profile": gm.transcript.language_profile(),
        "summary": narrative.summary[:900],
        "topics": topics,
        "decisions": decisions,
        "actions": actions,
        "open_questions": questions,
        "risks": risks,
        "follow_ups": [],
        "flags": sorted(flags, key=lambda f: f.priority),
        "classification": gm.meeting.classification,
        "provenance": fake_provenance(gm, generator, now),
    }
    if gm.meeting.meeting_type.value == "mancom":
        return MancomMinutes(**base, figures_mentioned=list(merged.figures))
    if gm.meeting.meeting_type.value == "one_to_one":
        mine = {str(gm.spec.get("organiser_name", "")).lower(), "me"}
        return OneToOneMinutes(
            **base,
            private=True,
            my_commitments=[a for a in actions if (a.owner or "").lower() in mine],
            their_commitments=[a for a in actions if (a.owner or "").lower() not in mine],
        )
    return Minutes(**base)


def minutes_from_playback(gm: GoldenMeeting, **kw: Any) -> Minutes:
    """Assemble minutes from the meeting's own ``playback.json`` (no LLM, no verifier)."""
    merged = MergedFindings.model_validate(gm.playback["MergedFindings"][0])
    narrative = Narrative.model_validate(gm.playback["Narrative"][0])
    verdicts = [
        RetractionVerdict.model_validate(v) for v in gm.playback.get("RetractionVerdict", [])
    ]
    return assemble_minutes(gm, merged, narrative, retractions=verdicts or None, **kw)
