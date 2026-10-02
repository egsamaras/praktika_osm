"""Teams WebVTT transcript ingest.

Contract: ``parse_teams_vtt`` turns the text of a Teams transcript export into a ``Transcript``
whose speakers come from ``<v Name>`` voice tags, whose language tags come from the script
ratio of each segment, and whose ids are ``S0001..`` in time order. ``inherit_names`` copies
those speakers onto an STT transcript by maximum time overlap. The helpers ``language_of`` and
``build_segments`` are shared with the DOCX ingest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import webvtt
from webvtt.errors import MalformedFileError

from praktika.errors import PraktikaError
from praktika.ids import segment_id
from praktika.logging import get_logger
from praktika.models import Segment, Transcript
from praktika.models.transcript import Language

log = get_logger(__name__)

MERGE_GAP_S = 1.0
MAX_SEGMENT_CHARS = 4000
ROOM_SPEAKER = "Room"
UNKNOWN_SPEAKER = "unknown"
AR_THRESHOLD = 0.85
EN_THRESHOLD = 0.15

_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TAG_RE = re.compile(r"<[^>]+>")
_TS_RE = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


@dataclass(frozen=True)
class Cue:
    """One raw cue before merging: ``speaker`` is ``None`` when there was no voice tag."""

    start: float
    end: float
    speaker: str | None
    text: str


def language_of(text: str) -> Language:
    """Tag ``text`` by script ratio: Arabic letters / (Arabic + Latin letters).

    Ratio >= 0.85 is ``ar``, <= 0.15 is ``en``, in between is ``mixed``; text with no letters
    of either script (digits, punctuation, empty) is ``unknown``.
    """
    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    total = arabic + latin
    if total == 0:
        return "unknown"
    ratio = arabic / total
    if ratio >= AR_THRESHOLD:
        return "ar"
    if ratio <= EN_THRESHOLD:
        return "en"
    return "mixed"


def parse_timestamp(ts: str) -> float:
    """Convert a WebVTT timestamp (``hh:mm:ss.mmm`` or ``mm:ss.mmm``) to seconds.

    Millisecond precision is kept (webvtt-py's ``start_in_seconds`` truncates to whole seconds,
    which would make a 1.5 s gap look like a mergeable 1 s one). Raises ``PraktikaError`` on
    text that is not a timestamp.
    """
    m = _TS_RE.fullmatch(ts.strip())
    if not m:
        raise PraktikaError(f"malformed WebVTT timestamp: {ts!r}")
    h, mnt, sec, ms = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000


def _resolve_speaker(voice: str | None, room_identities: list[str]) -> tuple[str, str]:
    if voice is None or not voice.strip():
        return UNKNOWN_SPEAKER, "unknown"
    name = " ".join(voice.split())
    rooms = {r.strip().lower() for r in room_identities}
    if name.lower() in rooms:
        return ROOM_SPEAKER, "room"
    return name, "identity"


def build_segments(
    cues: list[Cue],
    room_identities: list[str],
    *,
    track: str,
    engine: str,
) -> list[Segment]:
    """Order, merge and id cues into ``Segment``s.

    Cues are sorted by start (stable, so overlapping cues keep file order among equals).
    Consecutive cues from the same resolved speaker whose gap is at most ``MERGE_GAP_S`` are
    merged into one segment (texts joined by a space) unless the result would exceed
    ``MAX_SEGMENT_CHARS``. Voice tags in ``room_identities`` become speaker ``Room`` with kind
    ``room``; missing tags become ``unknown``. Cues with no text are dropped.
    """
    ordered = sorted((c for c in cues if c.text.strip()), key=lambda c: c.start)
    merged: list[tuple[float, float, str, str, str]] = []  # start, end, speaker, kind, text
    for c in ordered:
        speaker, kind = _resolve_speaker(c.speaker, room_identities)
        text = " ".join(c.text.split())
        if merged:
            ps, pe, pspk, pkind, ptext = merged[-1]
            mergeable = (
                pspk == speaker
                and pkind == kind
                and kind != "unknown"
                and c.start - pe <= MERGE_GAP_S
                and len(ptext) + 1 + len(text) <= MAX_SEGMENT_CHARS
            )
            if mergeable:
                merged[-1] = (ps, max(pe, c.end), pspk, pkind, f"{ptext} {text}")
                continue
        merged.append((c.start, c.end, speaker, kind, text))
    return [
        Segment(
            id=segment_id(i + 1),
            start=start,
            end=max(start, end),
            speaker=speaker,
            speaker_kind=kind,  # type: ignore[arg-type]
            language=language_of(text),
            text=text[:MAX_SEGMENT_CHARS],
            confidence=None,
            track=track,  # type: ignore[arg-type]
            engine=engine,
        )
        for i, (start, end, speaker, kind, text) in enumerate(merged)
    ]


def _cues_from_vtt(text: str) -> list[Cue]:
    if not text.strip():
        return []
    try:
        parsed = webvtt.from_string(text)
    except MalformedFileError as e:
        raise PraktikaError(f"not a WebVTT transcript: {e}") from e
    cues: list[Cue] = []
    for cap in parsed:
        body = _TAG_RE.sub("", cap.text)
        cues.append(Cue(parse_timestamp(cap.start), parse_timestamp(cap.end), cap.voice, body))
    return cues


def parse_teams_vtt(text: str, meeting_id: str, room_identities: list[str]) -> Transcript:
    """Parse a Teams WebVTT export into a ``Transcript`` with ``source="vtt"``.

    Empty or whitespace-only input yields a transcript with no segments; text that is not
    WebVTT raises ``PraktikaError``. See ``build_segments`` for speaker, merge and language
    rules. The result is not redacted.
    """
    cues = _cues_from_vtt(text)
    segments = build_segments(cues, room_identities, track="vtt", engine="teams-vtt")
    log.info("ingest.vtt", meeting_id=meeting_id, cues=len(cues), segments=len(segments))
    return Transcript(
        meeting_id=meeting_id,
        source="vtt",
        engines={"vtt": "teams-transcript"},
        segments=segments,
    )


def _overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def inherit_names(stt: Transcript, vtt: Transcript) -> Transcript:
    """Copy speakers from ``vtt`` onto ``stt`` segments by maximum time overlap.

    An STT segment takes the speaker and kind (``identity`` or ``room``) of the VTT segment it
    overlaps most; segments with no positive overlap, or whose best match is ``unknown``, are
    left unchanged. Ids, times, text and language are never altered. The returned transcript
    records the VTT engine alongside the STT engines.
    """
    named = [s for s in vtt.segments if s.speaker_kind in ("identity", "room")]
    out: list[Segment] = []
    for seg in stt.segments:
        best = max(named, key=lambda v: _overlap(seg, v), default=None)
        if best is not None and _overlap(seg, best) > 0:
            seg = seg.model_copy(
                update={"speaker": best.speaker, "speaker_kind": best.speaker_kind}
            )
        out.append(seg)
    return stt.model_copy(update={"segments": out, "engines": {**stt.engines, **vtt.engines}})
