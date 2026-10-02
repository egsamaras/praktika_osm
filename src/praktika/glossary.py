"""Glossary normalisation after STT: known misrenderings become canonical terms.

Contract: ``load`` parses ``glossary.yaml`` and returns the entries with the file's SHA-256 (which
goes into ``Provenance.glossary_sha256``); it refuses a single-word misrendering that is an
ordinary English word (``COMMON_WORDS``), because the transcript it would rewrite is the
evidence behind every citation ("over dinner" must never become "over BHD"). ``apply``
replaces whole-word runs that fuzzy-match a misrendering (rapidfuzz ``partial_ratio`` ≥
``threshold`` to detect, ``ratio`` on word n-grams to locate) with the canonical spelling.
Redaction tokens such as ``«ACC_1»`` are never touched, and segments without a change are
returned as the same object.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from praktika.logging import get_logger
from praktika.models import Segment

log = get_logger(__name__)

TOKEN_RE = re.compile(r"«[^»]*»")
_WORD_RE = re.compile(r"\S+")
_TRAILING_PUNCT = ".,;:!?،؛)»\"'"

#: Ordinary English words that may never be a bare (single-word) misrendering: rewriting them
#: unconditionally would corrupt everyday speech in the transcript.
COMMON_WORDS: frozenset[str] = frozenset(
    {
        "dinner", "meeting", "office", "chair", "point", "share", "team", "file", "board", "note",
        "case", "plan", "rate", "bank", "fund", "eye", "see", "in", "for", "the", "and", "a", "an",
        "of", "to", "is", "it", "we",
    }
)  # fmt: skip


class GlossaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str = Field(min_length=1)
    variants: list[str] = []
    arabic_variants: list[str] = []
    misrenderings: list[str] = []


def load(path: Path) -> tuple[list[GlossaryEntry], str]:
    """Return ``(entries, sha256 of the file bytes)``.

    Raises ``FileNotFoundError`` for a missing file, ``ValueError`` when the document is not a
    mapping with an ``entries`` list, and ``pydantic.ValidationError`` for a malformed entry.
    """
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    doc = yaml.safe_load(raw) or {}
    if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
        raise ValueError(f"{path}: glossary must be a mapping with an 'entries' list")
    entries = [GlossaryEntry.model_validate(e) for e in doc["entries"]]
    for entry in entries:
        for mis in entry.misrenderings:
            if " " not in mis.strip() and mis.strip().lower() in COMMON_WORDS:
                raise ValueError(
                    f"{path}: misrendering {mis!r} for {entry.canonical!r} is an ordinary word; "
                    "use a multi-word pattern (for example 'Bahrain dinner') instead"
                )
    log.debug("glossary.loaded", path=str(path), entries=len(entries))
    return entries, sha


def _replace_in_span(text: str, mis: str, canonical: str, threshold: float) -> str:
    """Replace every whole-word n-gram of ``text`` scoring ≥ ``threshold`` against ``mis``."""
    n = len(mis.split())
    if n == 0:
        return text
    words = list(_WORD_RE.finditer(text))
    out: list[str] = []
    cursor = 0
    i = 0
    while i + n <= len(words):
        start, end = words[i].start(), words[i + n - 1].end()
        candidate = text[start:end]
        stripped = candidate.rstrip(_TRAILING_PUNCT)
        if stripped and fuzz.ratio(stripped, mis) >= threshold:
            out.append(text[cursor:start])
            out.append(canonical + candidate[len(stripped) :])
            cursor = end
            i += n
        else:
            i += 1
    out.append(text[cursor:])
    return "".join(out)


def normalise_text(text: str, entries: list[GlossaryEntry], threshold: float = 92) -> str:
    """Apply every misrendering → canonical replacement to ``text``, skipping «tokens»."""
    pieces = TOKEN_RE.split(text)
    tokens = TOKEN_RE.findall(text)
    for idx, piece in enumerate(pieces):
        if not piece.strip():
            continue
        for entry in entries:
            for mis in entry.misrenderings:
                if mis.strip() and fuzz.partial_ratio(mis, piece) >= threshold:
                    piece = _replace_in_span(piece, mis, entry.canonical, threshold)
        pieces[idx] = piece
    merged = [pieces[0]]
    for token, piece in zip(tokens, pieces[1:], strict=True):
        merged.extend((token, piece))
    return "".join(merged)


def apply(
    segments: list[Segment], entries: list[GlossaryEntry], threshold: float = 92
) -> list[Segment]:
    """Return segments with glossary misrenderings normalised; unchanged segments are reused."""
    out: list[Segment] = []
    changed = 0
    for seg in segments:
        text = normalise_text(seg.text, entries, threshold)
        if text == seg.text:
            out.append(seg)
        else:
            out.append(seg.model_copy(update={"text": text}))
            changed += 1
    log.debug("glossary.applied", segments=len(segments), changed=changed)
    return out
