"""Advisory checks of the verifier (C-07): names, speakers, confidence, MNPI
keywords and raw identifiers, plus the shared text walkers and the IBAN/Luhn validators.

Split from ``verify`` to keep both modules within the house line budget; ``verify.apply`` runs
these after the citation and number checks. Every check is pure and returns advisory ``Flag``
objects (only ``identifier_detected`` is priority 1).
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz, process

from praktika.models import Attendee, Flag, Minutes, Ref, Transcript

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
MNPI_KEYWORDS = (
    "results", "earnings", "dividend", "acquisition", "disposal", "capital raise",
    "rating action", "regulatory finding", "incident", "breach", "litigation", "provision",
    "impairment", "restructuring",
)  # fmt: skip
_NAME_STOP = {
    "committee", "chair", "board", "mancom", "alco", "legal", "infosec", "compliance", "finance",
    "risk", "audit", "bahrain", "saudi", "arabic", "english", "gregorian", "hijri",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "phase", "option", "pilot", "teams", "ollama", "whisper", "praktika",
    "management", "group", "bank", "team", "function", "policy", "credit", "data", "engineering",
    "governance", "lead", "head", "the", "and", "for", "with", "this", "that", "not",
}  # fmt: skip
_ARABIC_HONORIFICS = ("السيد", "السيدة", "الأستاذ", "الأستاذة", "الدكتور", "الدكتورة", "المهندس")
_CAP_RUN = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)*")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;\n]\s*|[\"“(]\s*)$")
_AR_NAME = re.compile("(?:" + "|".join(_ARABIC_HONORIFICS) + r")\s+([؀-ۿ]{2,}(?:\s[؀-ۿ]{2,})?)")
_IDENTIFIERS = {
    "iban": re.compile(r"\b[A-Za-z]{2}\d{2}[A-Za-z0-9]{11,30}\b", re.IGNORECASE),
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "phone": re.compile(r"(?:\+|00)(?:973|966|44|971|974|965|968)[ -]?\d(?:[ -]?\d){6,10}\b"),
    # Arabic context words with their possessive suffixes (رقمه الشخصي, إقامته, هويتها) and
    # the spoken "سي بي آر"; taa marbuta becomes taa before a suffix, hence the alternations.
    "cpr": re.compile(
        r"(?:\bCPR\b|سي بي آر|(?:ال)?رقم(?:ه|ها|ي|ك|كم|هم)?\s+(?:ال)?شخصي|رقم الهوية البحرينية)"
        r"\D{0,12}(\d{9})\b",
        re.IGNORECASE,
    ),
    "iqama": re.compile(
        r"(?:\biqama\b|\bnational id\b|\bNID\b|(?:ال)?إقام(?:ة|ت\w{0,2})|(?:ال)?هوي(?:ة|ت\w{0,2}))"
        r"\D{0,12}([12]\d{9})\b",
        re.IGNORECASE,
    ),
    "account": re.compile(
        r"(?:\baccount\b|\ba/c\b|\bacct\b|حساب(?:ه|ها|ي|ك|كم|هم)?)\D{0,12}(\d{8,16})\b",
        re.IGNORECASE,
    ),
}  # fmt: skip


def iban_ok(s: str) -> bool:
    """mod-97 check on an IBAN-shaped string."""
    s = s.replace(" ", "").upper()
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged if c.isalnum())
    return bool(digits) and int(digits) % 97 == 1


def luhn_ok(s: str) -> bool:
    """Luhn check on a digit string (separators ignored)."""
    digits = [int(c) for c in s if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        d = d * 2 if i % 2 else d
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def body_texts(minutes: Minutes, *, include_headings: bool = True) -> list[str]:
    """Every human-readable body string of the minutes (summary, topics, items, follow-ups).

    ``include_headings=False`` leaves out topic titles, which are Title Case by convention and
    would otherwise read as names (a heading such as "2027 Budget Request" yields "Budget Request").
    """
    texts = [minutes.summary, *minutes.follow_ups]
    for t in minutes.topics:
        texts += [t.summary, *t.key_points]
        if include_headings:
            texts.append(t.title)
    for d in minutes.decisions:
        texts += [d.statement, d.decided_by, d.dissent_or_conditions or ""]
    for a in minutes.actions:
        texts += [a.description, a.owner or "", a.due_text or ""]
    for q in minutes.open_questions:
        texts += [q.question, q.raised_by or "", q.owner or ""]
    for r in minutes.risks:
        texts += [r.description, r.owner or "", r.mitigation or ""]
    return [t for t in texts if t]


def all_refs(minutes: Minutes) -> list[Ref]:
    """Every citation on every body item and topic."""
    items: list[Any] = [
        *minutes.decisions, *minutes.actions, *minutes.open_questions, *minutes.risks,
        *minutes.topics,
    ]  # fmt: skip
    return [r for it in items for r in it.refs]


def _normalise_digits(text: str) -> str:
    return text.translate(_ARABIC_INDIC).replace(",", "").replace("٬", "").replace("٫", ".")


def _known_names(roster: list[Attendee]) -> list[str]:
    known: list[str] = []
    for a in roster:
        known += [a.name, a.organisation, *a.aliases]
        known += a.name.replace(".", " ").split()
    return [k for k in known if k]


def check_names(
    minutes: Minutes, roster: list[Attendee], known_terms: list[str] | None = None
) -> list[Flag]:
    """Flag capitalised runs and honorific-led Arabic names not on the roster (fuzzy ≥ 85).

    ``known_terms`` extends the roster with vocabulary that is capitalised but not a person
    (glossary canonicals and variants); each term and each of its words is accepted.
    """
    known = _known_names(roster)
    for term in known_terms or []:
        known += [term, *term.split()]
    flags: list[Flag] = []
    seen: set[str] = set()

    def unknown(candidate: str) -> bool:
        parts = [candidate, *candidate.split()]
        for p in parts:
            hit = process.extractOne(p, known, scorer=fuzz.ratio)
            if hit and hit[1] >= 85:
                return False
            # a CamelCase fragment of a known term ("Dev" from "DevOps") is not a name
            if len(p) >= 3 and any(p in k for k in known):
                return False
        return True

    for text in body_texts(minutes, include_headings=False):
        candidates = [m.group(1) for m in _AR_NAME.finditer(text)]
        for m in _CAP_RUN.finditer(text):
            if _SENTENCE_START.search(text[: m.start()]):
                continue
            words = [w for w in m.group(0).split() if w.lower() not in _NAME_STOP]
            if words and len(words[0]) >= 3:
                candidates.append(" ".join(words))
        for c in candidates:
            if c not in seen and unknown(c):
                seen.add(c)
                flags.append(Flag(kind="name_to_verify", detail=c))
    return flags


def check_speakers(minutes: Minutes) -> list[Flag]:
    """One ``unresolved_speaker`` flag per unnamed speaker label cited in the body."""
    by_label: dict[str, list[Ref]] = {}
    for r in all_refs(minutes):
        if re.fullmatch(r"SPEAKER_\d+|Room|unknown", r.speaker, re.IGNORECASE):
            by_label.setdefault(r.speaker, []).append(r)
    return [
        Flag(kind="unresolved_speaker", detail=f"speaker {label} is not mapped", refs=refs)
        for label, refs in by_label.items()
    ]


def check_confidence(minutes: Minutes, transcript: Transcript, low_conf: float) -> list[Flag]:
    """One ``low_confidence_audio`` flag per cited segment whose STT confidence < ``low_conf``."""
    by_id = transcript.by_id()
    flags: list[Flag] = []
    done: set[str] = set()
    for r in all_refs(minutes):
        seg = by_id.get(r.segment_id)
        if seg is None or seg.confidence is None or seg.id in done or seg.confidence >= low_conf:
            continue
        done.add(seg.id)
        flags.append(
            Flag(
                kind="low_confidence_audio",
                detail=f"{seg.id} confidence {seg.confidence:.2f}: {seg.text[:160]}",
                refs=[r],
            )
        )
    return flags


def check_mnpi(minutes: Minutes) -> list[Flag]:
    """A single ``possible_mnpi`` flag listing MNPI keywords found in the body."""
    body = " ".join(body_texts(minutes)).lower()
    hits = [k for k in MNPI_KEYWORDS if re.search(rf"\b{re.escape(k)}\b", body)]
    if not hits:
        return []
    return [Flag(kind="possible_mnpi", detail="keywords: " + ", ".join(hits))]


def check_identifiers(minutes: Minutes) -> list[Flag]:
    """Priority-1 ``identifier_detected`` flags for raw identifiers anywhere in the minutes."""
    texts = body_texts(minutes) + [r.quote for r in all_refs(minutes)]
    flags: list[Flag] = []
    seen: set[str] = set()
    for text in texts:
        norm = _normalise_digits(text)
        for kind, pattern in _IDENTIFIERS.items():
            for m in pattern.finditer(norm):
                value = m.group(1) if pattern.groups else m.group(0)
                if kind == "iban" and not iban_ok(value):
                    continue
                if kind == "card" and not luhn_ok(value):
                    continue
                if value in seen:
                    continue
                seen.add(value)
                masked = value[:4] + "…" + value[-2:] if len(value) > 8 else value
                flags.append(
                    Flag(kind="identifier_detected", detail=f"{kind}: {masked}", priority=1)
                )
    return flags


_DECISION_CLAIM = re.compile(
    r"\b(?:the\s+)?(?:committee|council|board|meeting|members|management|attendees|group)\s+"
    r"(?:has\s+|have\s+|was\s+|were\s+)?"
    r"(?:approved|agreed|decided|determined|resolved|concluded|confirmed|endorsed|ratified|"
    r"authorised|authorized|rejected)\b",
    re.IGNORECASE,
)


def check_narrative_decisions(minutes: Minutes) -> list[Flag]:
    """Flag a summary that asserts a collective decision when the minutes record none.

    A model output can turn a request put to a committee into "The committee determined that
    the reporting dashboard should be rebuilt" in the narrative while the body correctly holds
    no decision. The narrative is not citation-checked, so this is the one place an
    unsupported decision can still reach a reader.
    """
    if minutes.decisions:
        return []
    flags: list[Flag] = []
    texts = [("summary", minutes.summary)] + [(t.title, t.summary) for t in minutes.topics]
    for where, text in texts:
        m = _DECISION_CLAIM.search(text)
        if m:
            start = max(0, m.start() - 40)
            snippet = text[start : m.end() + 60].strip()
            flags.append(
                Flag(
                    kind="contradiction",
                    detail=(
                        f"{where} asserts a decision ('{snippet}') but the minutes record no "
                        "decision; reword it as a proposal or add the decision with a citation"
                    ),
                    priority=2,
                )
            )
    return flags
