"""Teams Recap-tab DOCX transcript ingest.

The Recap export is a sequence of paragraphs: a header ``Name  m:ss`` (or ``h:mm:ss``) followed
by one or more text paragraphs, repeated. ``parse_teams_docx`` reads that layout with
python-docx and reuses the VTT ingest's speaker, merge and language rules. The DOCX carries no
end times, so a block ends where the next block starts (or after an estimate for the last one).

A paragraph is a header only when its name part looks like a Teams display name (at most
``MAX_NAME_TOKENS`` tokens, each capitalised, an initial, Arabic, a name particle such as
``al``/``bin``/``van`` or a bracketed suffix such as ``(Guest)``; no sentence punctuation) and
its timestamp is monotonic (not before the previous header and within ``MAX_BLOCK_GAP_S`` of
it). An utterance that happens to end in a clock time ("We meet again at 10:30") therefore
stays text of the current block instead of becoming a bogus speaker at 630 s.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from praktika.errors import PraktikaError
from praktika.ingest.vtt import Cue, build_segments
from praktika.logging import get_logger
from praktika.models import Transcript

log = get_logger(__name__)

HEADER_RE = re.compile(r"^(?P<name>\S.*?)\s+(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\s*$")
UNKNOWN_NAME_RE = re.compile(r"^unknown(\s+speaker)?$", re.IGNORECASE)
LAST_BLOCK_WORDS_PER_S = 2.5
MIN_BLOCK_S = 1.0
MAX_NAME_TOKENS = 5
MAX_BLOCK_GAP_S = 7200.0  # no Recap block lasts two hours; a larger jump is speech, not a header
# One token of a display name: an initial ("F."), a capitalised Latin word with optional
# hyphen/apostrophe parts ("Al-Mahmood", "O'Neil"), an Arabic word, or a bracketed suffix.
_NAME_TOKEN_RE = re.compile(
    r"^(?:[A-Z]\.|[A-Z][A-Za-z]*(?:[-'’][A-Za-z]+)*|[\u0600-\u06FF]+|\([A-Za-z ]+\))$"
)
_NAME_PARTICLES = frozenset(
    {"al", "el", "bin", "bint", "ibn", "abu", "de", "da", "di", "du", "van", "der", "von",
     "la", "le", "y"}
)  # fmt: skip


def _parse_ts(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        m, s = parts
        return float(m * 60 + s)
    h, m, s = parts
    return float(h * 3600 + m * 60 + s)


def looks_like_name(name: str) -> bool:
    """True when ``name`` could be a Teams display name rather than a fragment of speech."""
    tokens = name.split()
    if not tokens or len(tokens) > MAX_NAME_TOKENS:
        return False
    return all(_NAME_TOKEN_RE.match(t) or t.lower() in _NAME_PARTICLES for t in tokens)


def _header(line: str, previous: float | None) -> tuple[str, float] | None:
    """``(name, start_s)`` when ``line`` is a speaker header, else ``None``.

    The name must look like a display name and the timestamp must not run backwards nor jump
    more than ``MAX_BLOCK_GAP_S`` past the previous header.
    """
    m = HEADER_RE.match(line)
    if not m or not looks_like_name(m.group("name").strip()):
        return None
    start = _parse_ts(m.group("ts"))
    if previous is not None and not previous <= start <= previous + MAX_BLOCK_GAP_S:
        log.debug("ingest.docx.header_rejected", text=line[:40], previous=previous, start=start)
        return None
    return m.group("name").strip(), start


def _blocks(paragraphs: list[str]) -> list[tuple[str, float, str]]:
    """Group paragraphs into ``(name, start_s, text)`` blocks; text before a header is skipped."""
    blocks: list[tuple[str, float, str]] = []
    current: tuple[str, float] | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None and buffer:
            blocks.append((current[0], current[1], " ".join(buffer)))

    for raw in paragraphs:
        line = " ".join(raw.split())
        if not line:
            continue
        header = _header(line, current[1] if current else None)
        if header:  # a header always starts a new block; one with no following text is dropped
            flush()
            current, buffer = header, []
            continue
        if current is None:
            log.debug("ingest.docx.skipped_preamble", text=line[:40])
            continue
        buffer.append(line)
    flush()
    return blocks


def _cues(blocks: list[tuple[str, float, str]]) -> list[Cue]:
    cues: list[Cue] = []
    for i, (name, start, text) in enumerate(blocks):
        if i + 1 < len(blocks) and blocks[i + 1][1] > start:
            end = blocks[i + 1][1]
        else:
            end = start + max(MIN_BLOCK_S, len(text.split()) / LAST_BLOCK_WORDS_PER_S)
        speaker = None if UNKNOWN_NAME_RE.match(name) else name
        cues.append(Cue(start, end, speaker, text))
    return cues


def parse_teams_docx(path: Path, meeting_id: str, room_identities: list[str]) -> Transcript:
    """Parse a Teams Recap DOCX transcript at ``path`` into a ``Transcript`` (``source="docx"``).

    Paragraphs that are neither a ``Name  m:ss`` header nor text under one (title lines, dates,
    empty paragraphs) are skipped; a header with no text is dropped; a name of ``Unknown`` or
    ``Unknown Speaker`` becomes an unattributed segment. Raises ``PraktikaError`` when the file
    is missing or is not a Word document. The result is not redacted.
    """
    try:
        doc = Document(str(path))
    except (PackageNotFoundError, FileNotFoundError, ValueError, KeyError) as e:
        raise PraktikaError(f"cannot read DOCX transcript {path}: {e}") from e
    paragraphs = [p.text for p in doc.paragraphs]
    blocks = _blocks(paragraphs)
    segments = build_segments(_cues(blocks), room_identities, track="docx", engine="teams-docx")
    log.info(
        "ingest.docx",
        meeting_id=meeting_id,
        paragraphs=len(paragraphs),
        blocks=len(blocks),
        segments=len(segments),
    )
    return Transcript(
        meeting_id=meeting_id,
        source="docx",
        engines={"docx": "teams-recap"},
        segments=segments,
    )
