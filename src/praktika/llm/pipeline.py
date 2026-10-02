"""Minutes generation pipeline: chunk → map → reduce → narrative → retraction
check → assemble → verify → audit.

Security properties enforced here: the transcript must already be redacted (C-06); the reduce
call never sees transcript text; every LLM call is audited as ``llm.call`` without content;
citations are resolved from the transcript, never trusted from the model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from praktika.errors import PraktikaError
from praktika.llm import assemble as asm
from praktika.llm import prompts as pr
from praktika.llm.base import (
    AuditedClient,
    AuditLike,
    LLMClient,
    complete_model,
    generator_name,
)
from praktika.llm.chunking import Chunk, chunk_transcript, estimate_tokens
from praktika.llm.retraction import retraction_flags
from praktika.llm.verify import VERIFIER_KINDS, flag_section
from praktika.llm.verify import apply as verify_apply
from praktika.logging import get_logger
from praktika.models import (
    ChunkFindings,
    ClassificationSuggestion,
    MancomMinutes,
    Meeting,
    MeetingType,
    MergedFindings,
    Minutes,
    ModelRegister,
    Narrative,
    OneToOneMinutes,
    Provenance,
    Review,
    Transcript,
)

log = get_logger(__name__)
SECTIONS = ("summary", "topics", "decisions", "actions", "open_questions", "risks")
Section = Literal["summary", "topics", "decisions", "actions", "open_questions", "risks"]


class GenerateOptions(BaseModel):
    """Per-run knobs. ``fallback_model`` is used (when the client supports ``with_model``) once
    the transcript exceeds ``long_transcript_tokens``; the run is then marked degraded."""

    model_config = ConfigDict(extra="forbid")

    template: MeetingType
    prompt_version: str = "v1"
    full_context_max_tokens: int = 16000
    language: Literal["en"] = "en"
    map_chunk_tokens: int = 8000
    long_transcript_tokens: int = 60000
    fallback_model: str | None = None
    #: Glossary canonicals and variants: capitalised terms the name check must not flag.
    known_terms: list[str] = []


def _extract(
    client: LLMClient, system: str, prompts: pr.PromptSet, chunks: list[Chunk]
) -> list[ChunkFindings]:
    total = len(chunks)
    return [
        complete_model(client, system, pr.extract_prompt(prompts, c.text, i, total), ChunkFindings)
        for i, c in enumerate(chunks, 1)
    ]


def _merge(
    client: LLMClient, system: str, prompts: pr.PromptSet, found: list[ChunkFindings]
) -> MergedFindings:
    """Single chunk: pass through. Several: the reduce call, which sees findings only."""
    if len(found) == 1:
        return MergedFindings(**found[0].model_dump(), retracted_decisions=[])
    return complete_model(client, system, pr.reduce_prompt(prompts, list(found)), MergedFindings)


def _template_extra(template: MeetingType, merged: MergedFindings) -> dict[str, Any]:
    """Fields specific to ``MancomMinutes`` / ``OneToOneMinutes`` known before verification
    (the one-to-one commitments are derived afterwards by ``sync_derived``)."""
    spec_model = pr.TEMPLATES[template].model
    if spec_model is MancomMinutes:
        return {"figures_mentioned": list(merged.figures)}
    if spec_model is OneToOneMinutes:
        return {"private": True}
    return {}


def sync_derived(minutes: Minutes, transcript: Transcript | None) -> Minutes:
    """Recompute the fields derived from the body after verification or review edits.

    ``follow_ups`` is a view over the verified ``decisions`` of kind ``deferred``, and
    ``OneToOneMinutes.my_commitments`` / ``their_commitments`` are views over the verified
    ``actions``: an item removed by the verifier as uncited or rejected by the reviewer can
    therefore never survive as a follow-up or a commitment in the body or its export (C-07).
    The commitments need the transcript (speaker tracks); without one they are left as
    they are.
    """
    update: dict[str, Any] = {
        "follow_ups": [d.statement for d in minutes.decisions if d.kind == "deferred"]
    }
    if isinstance(minutes, OneToOneMinutes) and transcript is not None:
        mine, theirs = asm.split_commitments(minutes.actions, transcript.by_id())
        update.update({"my_commitments": mine, "their_commitments": theirs})
    return minutes.model_copy(update=update)


def generate(
    transcript: Transcript,
    meeting: Meeting,
    client: LLMClient,
    prompts: pr.PromptSet,
    glossary_sha: str,
    register: ModelRegister | dict[str, str] | None,
    opts: GenerateOptions,
    audit: AuditLike,
) -> Minutes:
    """Draft minutes for ``meeting`` from a redacted ``transcript``.

    Raises ``PraktikaError`` if the transcript is not redacted or is empty. Returns the
    template's ``Minutes`` subclass with verifier flags sorted by priority. Emits ``llm.call``
    per model call and ``minutes.drafted`` once, both carrying the meeting's classification.
    The transcript is never mutated.
    """
    if not transcript.redacted:
        raise PraktikaError("transcript not redacted")
    if not transcript.segments:
        raise PraktikaError("transcript has no segments")
    roster, by_id = meeting.roster, transcript.by_id()
    rendered = transcript.render_for_llm(roster)
    total = estimate_tokens(rendered)
    degraded = total > opts.long_transcript_tokens
    if degraded and opts.fallback_model and hasattr(client, "with_model"):
        log.warning("llm.degrade", tokens=total, fallback=opts.fallback_model)
        client = client.with_model(opts.fallback_model)
    classification = meeting.classification.value
    audited = AuditedClient(
        client, audit, meeting.id, prompts.sha256, classification=classification
    )
    system = pr.system_prompt(
        prompts,
        roster=roster,
        title=meeting.title,
        meeting_type=opts.template,
        meeting_date=meeting.started_at.date(),
        language_mode=meeting.language_mode.value,
    )
    if total <= opts.full_context_max_tokens:
        ids = [s.id for s in transcript.segments]
        chunks = [Chunk(index=0, segment_ids=ids, text=rendered, approx_tokens=total)]
    else:
        chunks = chunk_transcript(transcript, roster, target_tokens=opts.map_chunk_tokens)
    merged = _merge(audited, system, prompts, _extract(audited, system, prompts, chunks))
    narrative = complete_model(audited, system, pr.narrative_prompt(prompts, merged), Narrative)
    flags = retraction_flags(audited, system, prompts, merged, transcript, roster)
    items, uncited = asm.assemble_items(merged, by_id, meeting.started_at.date())
    provenance = Provenance(
        generator_model=generator_name(client),
        model_digest=client.model_digest(),
        prompt_version=prompts.version,
        prompt_sha256=prompts.sha256,
        glossary_sha256=glossary_sha,
        template=opts.template.value,
        git_sha=asm.git_sha(),
        transcript_sha256=transcript.sha256(),
        stt_engines=dict(transcript.engines),
        model_hashes=asm.model_hashes(register),
        degraded=degraded,
        generated_at=datetime.now(UTC),
    )
    draft = pr.TEMPLATES[opts.template].model(
        meeting_id=meeting.id,
        title=meeting.title,
        meeting_type=opts.template,
        date=meeting.started_at.date(),
        attendees=list(roster),
        language_profile=transcript.language_profile(),
        summary=narrative.summary[:900],
        topics=asm.topics_from(narrative, by_id),
        **items,
        follow_ups=[],  # derived from the verified decisions by ``sync_derived``
        flags=flags + uncited,
        classification=meeting.classification,
        provenance=provenance,
        **_template_extra(opts.template, merged),
    )
    minutes = sync_derived(
        verify_apply(draft, transcript, roster, known_terms=opts.known_terms), transcript
    )
    audit.append(
        "minutes.drafted",
        meeting.id,
        classification=classification,
        version=minutes.version,
        template=opts.template.value,
        model=provenance.generator_model,
        prompt_sha=prompts.sha256,
        degraded=degraded,
        flags=len(minutes.flags),
        blocking=len(minutes.blocking_flags()),
    )
    return minutes


def regenerate_section(
    minutes: Minutes,
    transcript: Transcript,
    section: Section,
    instruction: str,
    client: LLMClient,
    prompts: pr.PromptSet,
) -> Minutes:
    """Re-draft one section under a reviewer instruction; return a new, re-verified version.

    Findings sections are re-extracted from the whole transcript; ``summary`` and ``topics`` are
    re-narrated from the current body. The instruction is appended to the user message, never
    to the system rules. The review record is reset. Raises ``ValueError`` for an unknown
    section and ``PraktikaError`` for an unredacted transcript.

    Flags: advisory flags the verifier produced last time are dropped before re-verification
    (the verifier recreates the ones that still apply; cleared flags and ``contradiction``
    flags are kept), and ``uncited_item_removed`` flags belonging to sections other than the
    regenerated one keep their restore JSON. Flag counts are therefore stable across repeated
    regenerations instead of duplicating.
    """
    if section not in SECTIONS:
        raise ValueError(f"unknown section: {section}")
    if not transcript.redacted:
        raise PraktikaError("transcript not redacted")
    roster, by_id = minutes.attendees, transcript.by_id()
    profile = minutes.language_profile
    lang = "ar-mixed" if profile.get("ar", 0) + profile.get("mixed", 0) > 0 else "en"
    system = pr.system_prompt(
        prompts,
        roster=roster,
        title=minutes.title,
        meeting_type=minutes.meeting_type,
        meeting_date=minutes.date,
        language_mode=lang,
    )
    suffix = f"\n\nReviewer instruction for this re-draft: {instruction}"
    update: dict[str, Any] = {
        "flags": [f for f in minutes.flags if f.cleared or f.kind not in VERIFIER_KINDS]
    }
    if section in ("summary", "topics"):
        user = pr.narrative_prompt(prompts, asm.findings_from_minutes(minutes)) + suffix
        narrative = complete_model(client, system, user, Narrative)
        if section == "summary":
            update["summary"] = narrative.summary[:900]
        else:
            update["topics"] = asm.topics_from(narrative, by_id)
    else:
        user = pr.extract_prompt(prompts, transcript.render_for_llm(roster), 1, 1) + suffix
        found = complete_model(client, system, user, ChunkFindings)
        merged = MergedFindings(**found.model_dump(), retracted_decisions=[])
        items, uncited = asm.assemble_items(merged, by_id, minutes.date)
        update[section] = items[section]
        kept = [
            f
            for f in update["flags"]
            if f.kind != "uncited_item_removed" or f.cleared or flag_section(f) != section
        ]
        update["flags"] = kept + [f for f in uncited if flag_section(f) == section]
    update["version"] = minutes.version + 1
    update["provenance"] = minutes.provenance.model_copy(update={"generated_at": datetime.now(UTC)})
    update["review"] = Review()
    fresh = verify_apply(minutes.model_copy(update=update), transcript, roster)
    return sync_derived(fresh, transcript)


def suggest_classification(
    transcript: Transcript,
    client: LLMClient,
    prompts: pr.PromptSet,
    *,
    max_tokens: int = 6000,
) -> ClassificationSuggestion:
    """Advisory classification hint from the first ``max_tokens`` of the redacted transcript.

    Raises ``PraktikaError`` if the transcript is not redacted: raw identifiers never reach the
    model, even for a hint. The organiser, not this function, sets the classification.
    """
    if not transcript.redacted:
        raise PraktikaError("transcript not redacted")
    lines: list[str] = []
    used = 0
    for line in transcript.render_for_llm([]).split("\n"):
        used += estimate_tokens(line)
        if used > max_tokens:
            break
        lines.append(line)
    user = pr.classify_prompt(prompts, "\n".join(lines))
    return complete_model(client, pr.CLASSIFY_SYSTEM, user, ClassificationSuggestion)
