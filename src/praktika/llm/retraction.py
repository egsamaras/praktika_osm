"""Retraction check of the minutes pipeline: for every merged decision, ask the
model whether it was withdrawn within ±3 segments of its citations and record a
``contradiction`` flag when it was (plus one per earlier position the reduce stage superseded).

Split from ``pipeline`` to keep that module within the house line budget.
"""

from __future__ import annotations

from praktika.llm import assemble as asm
from praktika.llm import prompts as pr
from praktika.llm.base import LLMClient, complete_model
from praktika.models import Attendee, Flag, MergedFindings, RetractionVerdict, Transcript

RETRACTION_NEIGHBOURS = 3


def _neighbourhood(index: dict[str, int], ids: list[str], n_lines: int) -> list[int]:
    rows: set[int] = set()
    for sid in ids:
        if sid in index:
            lo = max(0, index[sid] - RETRACTION_NEIGHBOURS)
            hi = min(n_lines, index[sid] + RETRACTION_NEIGHBOURS + 1)
            rows.update(range(lo, hi))
    return sorted(rows)


def retraction_flags(
    client: LLMClient,
    system: str,
    prompts: pr.PromptSet,
    merged: MergedFindings,
    transcript: Transcript,
    roster: list[Attendee],
) -> list[Flag]:
    """A ``contradiction`` flag for every decision found withdrawn in its ±3-segment
    neighbourhood, plus one per earlier position the reduce stage recorded as superseded."""
    lines = transcript.render_for_llm(roster).split("\n")
    index = {s.id: i for i, s in enumerate(transcript.segments)}
    by_id = transcript.by_id()
    flags: list[Flag] = []
    for d in merged.decisions:
        rows = _neighbourhood(index, d.refs, len(lines))
        if not rows:
            continue
        user = pr.retraction_prompt(
            prompts,
            statement=d.statement,
            kind=d.kind,
            ref_ids=d.refs,
            lines="\n".join(lines[j] for j in rows),
        )
        verdict = complete_model(client, system, user, RetractionVerdict)
        if verdict.retracted:
            refs = asm.resolve_refs(verdict.refs or d.refs, None, by_id)
            refs = [r for r in refs if r.segment_id in by_id]
            detail = f"{d.statement} — {verdict.note}"
            flags.append(Flag(kind="contradiction", detail=detail, refs=refs))
    for d in merged.retracted_decisions:
        refs = [r for r in asm.resolve_refs(d.refs, d.quote, by_id) if r.segment_id in by_id]
        detail = f"Earlier position superseded: {d.statement}"
        flags.append(Flag(kind="contradiction", detail=detail, refs=refs))
    return flags
