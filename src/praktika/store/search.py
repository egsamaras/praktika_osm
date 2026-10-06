"""Keyword search over minutes.

Contract: ``index_minutes`` writes an FTS row only for approved minutes of indexable templates
and non-restricted classifications (C-07: nothing is searchable before human review; C-09: MNPI
minutes are never indexed); anything else clears any earlier row.
``normalise_ar`` is applied to text at index time and to the query at search time so Arabic
spelling variants (hamza forms of alef, final yaa/alef maqsura, tashkeel, tatweel, Arabic-Indic
digits) match. There is no semantic search.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from praktika.models import MancomMinutes, Minutes, OneToOneMinutes, TemplateSpec
from praktika.store.repo import Store

_TASHKEEL = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ALEF = re.compile("[آأإٱ]")
_ARABIC_INDIC = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_EXTENDED_INDIC = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


class Hit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meeting_id: str
    version: int
    title: str
    snippet: str
    rank: float


def normalise_ar(text: str) -> str:
    """Normalise Arabic for matching: strip tashkeel and tatweel, unify alef forms to bare alef,
    alef maqsura to yaa, taa marbuta to haa, and Arabic-Indic digits to Western digits.

    Latin text passes through unchanged apart from being lower-cased.
    """
    out = _TASHKEEL.sub("", text)
    out = _ALEF.sub("ا", out)
    out = out.replace("ى", "ي").replace("ة", "ه")
    out = out.translate(_ARABIC_INDIC).translate(_EXTENDED_INDIC)
    return out.lower()


_ARABIC_LETTER = re.compile("[\u0600-\u06ff]")
#: A conjunction or preposition Arabic attaches to a name ('وعلي حسن'); not the article, which
#: would turn a first name into someone else's family name ('العلي').
_AR_PREFIX = "[وفبلك]?"
_NAME_PARTS = re.compile(r"[\s-]+")


def _fold(text: str) -> str:
    """``normalise_ar`` without folding alef maqsura into yaa: 'على' (on) must stay apart from
    'علي' (Ali)."""
    out = _TASHKEEL.sub("", text)
    out = _ALEF.sub("ا", out).replace("ة", "ه")
    out = out.translate(_ARABIC_INDIC).translate(_EXTENDED_INDIC)
    return out.lower()


def mentions(text: str | None, name: str) -> bool:
    """Whether a full name of two words or more appears in ``text`` as whole words, a space or a
    hyphen between them in either: 'Ali Hassan' is found in '1:1 with Ali Hassan', 'Fatima Al
    Zayani' in 'Fatima al-Zayani'. A single word is never matched, because 'Ali', 'May' or
    'Bond' would match ordinary titles; a request needs the full name (or the UPN)."""
    words = [w for w in _NAME_PARTS.split(_fold(name)) if w]
    if not text or len(words) < 2:
        return False
    body = r"[\s-]+".join(re.escape(w) for w in words)
    lead = _AR_PREFIX if _ARABIC_LETTER.match(words[0]) else ""
    return re.search(rf"(?<!\w){lead}{body}(?!\w)", _fold(text)) is not None


def indexable(minutes: Minutes, template_spec: TemplateSpec) -> bool:
    """True when this minutes record may be indexed: approved, indexable template, not
    restricted. Drafts, in-review and discarded minutes are never searchable."""
    return (
        minutes.review.status == "approved"
        and bool(template_spec.indexable)
        and minutes.classification.value != "restricted"
    )


def body_text(minutes: Minutes) -> str:
    """The searchable body: topics, decisions, actions, questions, risks, follow-ups and
    type-specific sections, one item per line. Flags and provenance are excluded."""
    lines: list[str] = []
    for t in minutes.topics:
        lines.extend([t.title, t.summary, *t.key_points])
    lines.extend(d.statement for d in minutes.decisions)
    lines.extend(f"{a.description} {a.owner or ''} {a.due_text or ''}" for a in minutes.actions)
    lines.extend(q.question for q in minutes.open_questions)
    lines.extend(f"{r.description} {r.mitigation or ''}" for r in minutes.risks)
    lines.extend(minutes.follow_ups)
    if isinstance(minutes, MancomMinutes):
        lines.extend(a.title for a in minutes.agenda)
        lines.extend(f"{f.value} {f.context}" for f in minutes.figures_mentioned)
        lines.extend(minutes.escalations_to_board)
    if isinstance(minutes, OneToOneMinutes):
        lines.extend(a.description for a in minutes.my_commitments + minutes.their_commitments)
    return "\n".join(line.strip() for line in lines if line and line.strip())


def index_minutes(store: Store, minutes: Minutes, template_spec: TemplateSpec) -> bool:
    """Index ``minutes`` when permitted; otherwise remove any stale row. Returns whether indexed."""
    if not indexable(minutes, template_spec):
        store.clear_index(minutes.meeting_id)
        return False
    store.index_minutes(
        minutes.meeting_id,
        minutes.version,
        normalise_ar(minutes.title),
        normalise_ar(minutes.summary),
        normalise_ar(body_text(minutes)),
    )
    return True


def fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression: each token becomes a quoted term,
    joined with implicit AND. Returns an empty string when there is nothing to search for."""
    tokens = [t.replace('"', "") for t in _TOKEN.findall(normalise_ar(q))]
    return " ".join(f'"{t}"' for t in tokens if t)


def search(store: Store, q: str, *, include_private: bool = False, limit: int = 20) -> list[Hit]:
    """Search indexed minutes. Private meetings are excluded unless ``include_private``."""
    match = fts_query(q)
    if not match:
        return []
    rows = store.search_index(match, include_private=include_private, limit=max(1, limit))
    return [
        Hit(
            meeting_id=r["meeting_id"],
            version=int(r["version"]),
            title=r["title"],
            snippet=r["snippet"],
            rank=float(r["rank"]),
        )
        for r in rows
    ]
