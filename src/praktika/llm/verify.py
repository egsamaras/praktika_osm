"""Deterministic verifier run on every draft before a human sees it (C-07).

Seven checks in a fixed order. A citation survives only if its segment exists, its quote is a
(near-)verbatim part of the segment — compared after Arabic normalisation (tashkeel, alef and
taa-marbuta forms, Arabic-Indic digits) with a length guard and a polarity/figure guard
(``llm.quotes``), so a fabricated quote that merely shares a few tokens with a short segment,
drops a negation or swaps a number cannot pass; a quote that straddles consecutive cited
segments is checked against their joined text — and the segment is not an instruction aimed
at the model; a decision or action with no surviving citation is moved out
of the body into a priority-1 flag that carries its full text and JSON so a reviewer can
restore it (the reason names an instruction-like segment when that is why). The remaining
checks add advisory flags, deduplicated by ``(kind, detail)`` against flags already present;
only a raw identifier in the body is priority 1.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from praktika.llm.quotes import (
    QUOTE_MIN_CHARS,
    QUOTE_MIN_TOKENS,
    QUOTE_TOKEN_SLACK,
    cited_runs,
    neighbour_texts,
    quote_matches,
    supports,
)
from praktika.llm.verify_flags import (
    MNPI_KEYWORDS,
    body_texts,
    check_confidence,
    check_identifiers,
    check_mnpi,
    check_names,
    check_narrative_decisions,
    check_speakers,
    iban_ok,
    luhn_ok,
)
from praktika.models import ActionItem, Attendee, Decision, Flag, Minutes, Ref, Transcript

__all__ = [
    "INJECTION_PATTERNS",
    "INSTRUCTION_REASON",
    "MNPI_KEYWORDS",
    "QUOTE_MIN_CHARS",
    "QUOTE_MIN_TOKENS",
    "QUOTE_TOKEN_SLACK",
    "VERIFIER_KINDS",
    "apply",
    "check_confidence",
    "check_identifiers",
    "check_mnpi",
    "check_names",
    "check_numbers",
    "check_refs",
    "check_speakers",
    "flag_section",
    "iban_ok",
    "is_instruction_like",
    "item_text",
    "luhn_ok",
    "normalise_digits",
    "quote_matches",
    "removed_flag",
]

# Second-person imperatives aimed at the model. Ordinary meeting speech that merely mentions
# instructions ("disregard the previous instructions we sent the vendor"), a system prompt of
# some product, or "you are now the owner" must not match (they are everyday vocabulary in a
# technology team), so the object-less forms are required and the role-play forms are anchored to
# the start of the segment.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:ignore|disregard)\s+(?:(?:all|any|the|your|my|these|those)\s+){0,2}"
        r"(?:previous|prior|above|earlier)\s+instructions?\b"
        r"(?!\s+(?:we|i|they|you|he|she|that|which|from|sent|given|issued|of|about|on))",
        r"\bmark\s+all\s+(?:the\s+)?actions?\s+(?:as\s+)?(?:closed|complete|done)\b",
        r"^\W*(?:system|assistant)\s*:",
        r"^\W*you\s+are\s+now\s+(?:a|an)\b",
        r"^\W*new\s+instructions?\s*:",
        r"تجاهل\s+(?:كل\s+)?التعليمات\s+السابقة",
    )
]
INSTRUCTION_REASON = "cited an instruction-like segment; review manually"
_MAGNITUDE = re.compile(
    r"^\s?(?:million|billion|thousand|mn|bn|k|m|مليون|ألف|الف|مليار)\b", re.IGNORECASE
)
_CURRENCY_BEFORE = re.compile(
    r"(?:BHD|SAR|USD|GBP|EUR|KWD|AED|QAR|OMR|\$|£|€|دينار|ريال|دولار)\s?$", re.IGNORECASE
)
VERIFIER_KINDS = frozenset(
    {
        "number_to_verify",
        "name_to_verify",
        "unresolved_speaker",
        "low_confidence_audio",
        "possible_mnpi",
        "identifier_detected",
    }
)
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUMBER = re.compile(r"\d[\d,.٫٬]*\d|\d")
_BODY_KINDS = ("decisions", "actions", "open_questions", "risks")


def is_instruction_like(text: str) -> bool:
    """True when ``text`` reads as an instruction to the model rather than meeting speech."""
    return any(p.search(text) for p in INJECTION_PATTERNS)


def normalise_digits(text: str) -> str:
    """Arabic-Indic digits to Western, thousands separators removed, ``٪`` unified to ``%``."""
    out = text.translate(_ARABIC_INDIC).replace(",", "").replace("٬", "").replace("٫", ".")
    return out.replace("٪", "%")


def item_text(item: BaseModel) -> str:
    """The human-readable text of a minutes item (statement or description plus owner)."""
    if isinstance(item, Decision):
        return f"{item.statement} (decided by {item.decided_by})"
    if isinstance(item, ActionItem):
        owner = item.owner or "unassigned"
        due = f", due {item.due_text}" if item.due_text else ""
        return f"{item.description} (owner: {owner}{due})"
    return str(getattr(item, "question", None) or getattr(item, "description", ""))


def removed_flag(item: BaseModel, *, reason: str = "no verifiable citation") -> Flag:
    """The ``uncited_item_removed`` flag for ``item``: full text, JSON to restore, priority."""
    priority = 1 if isinstance(item, Decision | ActionItem) else 2
    return Flag(
        kind="uncited_item_removed",
        detail=f"{item_text(item)} — removed: {reason}",
        item_json=json.dumps(item.model_dump(mode="json"), ensure_ascii=False),
        priority=priority,
    )


def check_refs(minutes: Minutes, transcript: Transcript, quote_ratio: int) -> Minutes:
    """Drop citations that do not verify; remove and flag items left without any.

    A ref whose segment is instruction-like is dropped for that reason, and an item that loses
    every ref that way is flagged with ``INSTRUCTION_REASON`` rather than "no verifiable
    citation", so the reviewer is told the real cause. A quote that spans two or more
    consecutive cited segments is checked against their joined text (``quotes.cited_runs``),
    so a decision whose evidence straddles a Whisper segment boundary keeps its citations.
    """
    by_id = transcript.by_id()

    def verdict(ref: Ref, runs: dict[str, str] | None = None) -> str:
        """``keep``, ``instruction`` or ``drop``."""
        seg = by_id.get(ref.segment_id)
        if seg is None:
            return "drop"
        if is_instruction_like(seg.text):
            return "instruction"
        run_text = (runs or {}).get(ref.segment_id)
        if supports(ref.quote, seg, run_text, quote_ratio):
            return "keep"
        if any(
            quote_matches(ref.quote, text, quote_ratio)
            for text in neighbour_texts(ref.segment_id, transcript.segments)
        ):
            return "keep"  # verbatim across a segment boundary the model did not cite fully
        return "drop"

    update: dict[str, Any] = {}
    flags = list(minutes.flags)
    for kind in _BODY_KINDS:
        kept = []
        for item in getattr(minutes, kind):
            runs = cited_runs([r.segment_id for r in item.refs], transcript.segments)
            verdicts = [verdict(r, runs) for r in item.refs]
            refs = [r for r, v in zip(item.refs, verdicts, strict=True) if v == "keep"]
            if refs:
                kept.append(item.model_copy(update={"refs": refs}))
            elif "instruction" in verdicts:
                flags.append(removed_flag(item, reason=INSTRUCTION_REASON))
            else:
                flags.append(removed_flag(item))
        update[kind] = kept
    update["topics"] = [
        t.model_copy(update={"refs": [r for r in t.refs if verdict(r) == "keep"]})
        for t in minutes.topics
    ]
    update["flags"] = flags
    return minutes.model_copy(update=update)


def _checkable(norm: str, m: re.Match[str]) -> bool:
    """A figure worth checking: multi-digit, or a single digit that is a percentage, carries a
    magnitude word (``2 million``, ``9 billion``, ``٥ مليون``) or follows a currency code."""
    if len(m.group(0)) >= 2:
        return True
    after, before = norm[m.end() :], norm[: m.start()]
    if after[:1] == "%":
        return True
    return bool(_MAGNITUDE.match(after) or _CURRENCY_BEFORE.search(before))


def check_numbers(minutes: Minutes, transcript: Transcript) -> list[Flag]:
    """Flag any checkable number in the body that the transcript lacks (offsets are taken on
    the normalised text, so thousands separators never shift the percent check)."""
    haystack = normalise_digits(" ".join(s.text for s in transcript.segments))
    present = set(_NUMBER.findall(haystack))
    flags: list[Flag] = []
    seen: set[str] = set()
    for text in body_texts(minutes):
        norm = normalise_digits(text)
        for m in _NUMBER.finditer(norm):
            value = m.group(0)
            if value in seen or not _checkable(norm, m):
                continue
            if value not in present:
                seen.add(value)
                flags.append(Flag(kind="number_to_verify", detail=f"{value} in: {text[:160]}"))
    return flags


def apply(
    minutes: Minutes,
    transcript: Transcript,
    roster: list[Attendee],
    *,
    quote_ratio: int = 85,
    low_conf: float = 0.5,
    known_terms: list[str] | None = None,
) -> Minutes:
    """Run all eight checks in order and return the minutes with flags sorted by priority.

    ``known_terms`` (glossary canonicals and variants such as "ManCom", "ALCO", "KYC") are
    treated like roster names by the name check, so organisation vocabulary is not flagged.

    Existing flags on ``minutes`` are preserved (cleared ones included); a new advisory flag
    whose ``(kind, detail)`` already exists is not added again, so re-verifying is stable.
    The input is not mutated.
    """
    out = check_refs(minutes, transcript, quote_ratio)
    flags = list(out.flags)
    fresh: list[Flag] = []
    fresh += check_numbers(out, transcript)
    fresh += check_names(out, roster, known_terms or [])
    fresh += check_speakers(out)
    fresh += check_confidence(out, transcript, low_conf)
    fresh += check_mnpi(out)
    fresh += check_identifiers(out)
    fresh += check_narrative_decisions(out)
    seen = {(f.kind, f.detail) for f in flags}
    for f in fresh:
        if (f.kind, f.detail) not in seen:
            seen.add((f.kind, f.detail))
            flags.append(f)
    flags.sort(key=lambda f: f.priority)
    return out.model_copy(update={"flags": flags})


def flag_section(flag: Flag) -> str | None:
    """The body list an ``uncited_item_removed`` flag's item belongs to (from its JSON keys),
    or ``None`` for any other flag."""
    if flag.kind != "uncited_item_removed" or not flag.item_json:
        return None
    try:
        data = json.loads(flag.item_json)
    except ValueError:
        return None
    for key, name in (
        ("severity", "risks"),
        ("owner_confidence", "actions"),
        ("statement", "decisions"),
        ("question", "open_questions"),
    ):
        if key in data:
            return name
    return None
