"""Language routing and engine orchestration.

Per meeting the gate sets a language mode: ``en`` (the English engine, except chunks Whisper
LID calls Arabic with p >= the threshold, which go to the Arabic engine — turbo is never used
for Arabic), ``ar-mixed`` (the Arabic engine unless LID says English with p >= the threshold)
or ``auto`` (LID on up to 24 spread chunks; Arabic share >= 0.3, or any confidently Arabic
sample, -> ``ar-mixed``, because Arabic in mixed meetings tends to cluster in one agenda item;
with no detector, as on the HTTP speech path, ``auto`` is English because nothing can say
otherwise and the deployment serves no Arabic engine). Whisper
engines receive a language hint only when LID agrees with the route (``None`` otherwise, so
Whisper auto-detects instead of transliterating); Cohere always decodes as Arabic. Engines are
loaded, run and unloaded one after another so the host never holds two models at once, and
local weights are verified against the model register before any backend loads them (C-13).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from praktika.audio.vad import speech_chunks
from praktika.config import Settings
from praktika.errors import LanguageRefused, PraktikaError
from praktika.ids import segment_id
from praktika.logging import get_logger
from praktika.models import Attendee, LanguageMode, RawSegment, Segment, SpeechChunk
from praktika.models_registry.weights import ensure_verified
from praktika.stt.base import LanguageDetector, SttLanguage
from praktika.stt.engines import Engines, NullTranscriber, build_transcribers

__all__ = [
    "AUTO_SAMPLE",
    "CONFIDENT",
    "Detections",
    "Engines",
    "NullTranscriber",
    "ResolvedMode",
    "RouteTable",
    "Routed",
    "build_transcribers",
    "detect_all",
    "merge_tracks",
    "resolve_mode",
    "route",
    "route_chunks",
    "transcribe_track",
]

log = get_logger(__name__)

ResolvedMode = Literal["en", "ar-mixed"]
RouteTable = dict[SttLanguage, list[SpeechChunk]]
Detections = dict[int, tuple[str, float]]
AUTO_SAMPLE = 24
CONFIDENT = 0.5  # LID probability below which no language hint is passed to Whisper


class Routed(NamedTuple):
    """One chunk's route: the engine (``en``/``ar``) and the language hint for it."""

    chunk: SpeechChunk
    engine: SttLanguage
    hint: SttLanguage | None


_TRACK_SPEAKER: dict[str, tuple[str, str]] = {
    "mic": ("ME", "self"),
    "system": ("SPEAKER_00", "label"),
    "file": ("unknown", "unknown"),
}


def _spread(n: int, k: int) -> list[int]:
    """``k`` indexes spread evenly over ``range(n)`` (fewer when ``n < k``)."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def detect_all(
    chunks: list[SpeechChunk], detector: LanguageDetector | None, wav: Path
) -> Detections | None:
    """LID for every chunk (index -> ``(code, probability)``), or ``None`` without a detector."""
    if detector is None:
        return None
    return {c.index: detector.detect(wav, c) for c in chunks}


def resolve_mode(
    mode: LanguageMode | str,
    detector: LanguageDetector | None,
    wav: Path,
    chunks: list[SpeechChunk],
    *,
    sample: int = AUTO_SAMPLE,
    ar_share: float = 0.3,
    ar_confident: float = 0.9,
    detections: Detections | None = None,
) -> ResolvedMode:
    """Collapse ``auto`` into ``en`` or ``ar-mixed`` by sampling LID on spread chunks.

    ``en`` and ``ar-mixed`` pass through. ``auto`` with no detector (the HTTP speech path) or
    no chunks resolves to ``en``: meetings are English by default and without language ID
    there is no evidence of Arabic, so no chunk is sent to an Arabic engine that the
    deployment may not serve. Up to
    ``sample`` chunks are examined; the result is ``ar-mixed`` when the Arabic share is at
    least ``ar_share`` *or* any sampled chunk is Arabic with p >= ``ar_confident`` (a single
    Arabic agenda item must never be decoded as English). ``detections`` reuses LID results
    already computed for these chunks.
    """
    if mode == LanguageMode.en:
        return "en"
    if mode == LanguageMode.ar_mixed:
        return "ar-mixed"
    if mode != LanguageMode.auto:
        raise PraktikaError(f"unknown language mode {mode!r}")
    if detector is None or not chunks:
        log.warning(
            "stt.auto_without_lid", resolved="en", detector=detector is not None, chunks=len(chunks)
        )
        return "en"
    picked = [chunks[i] for i in _spread(len(chunks), sample)]
    results = [
        detections[c.index] if detections and c.index in detections else detector.detect(wav, c)
        for c in picked
    ]
    arabic = [p for lang, p in results if lang == "ar"]
    share = len(arabic) / len(picked)
    confident = any(p >= ar_confident for p in arabic)
    resolved: ResolvedMode = "ar-mixed" if (share >= ar_share or confident) else "en"
    log.info(
        "stt.mode_resolved",
        sampled=len(picked),
        ar_share=round(share, 2),
        confident_arabic=confident,
        mode=resolved,
    )
    return resolved


def route_chunks(
    chunks: list[SpeechChunk],
    mode: ResolvedMode,
    detections: Detections | None,
    threshold: float,
    *,
    confident: float = CONFIDENT,
) -> list[Routed]:
    """Route every chunk and decide the language hint its engine receives.

    ``en`` mode: a chunk LID calls ``ar`` with p >= ``threshold`` goes to the Arabic engine
    (never turbo for Arabic); the rest go to the English engine with hint ``en`` when LID
    agrees (p >= ``confident``) and no hint otherwise. ``ar-mixed``: a chunk goes to the
    English engine only when LID says ``en`` with p >= ``threshold``; the rest go to the Arabic
    engine with hint ``ar`` when LID agrees and no hint otherwise. Without detections the
    mode's engine is used with its own language as the hint.
    """
    out: list[Routed] = []
    for chunk in chunks:
        if detections is None:
            engine: SttLanguage = "en" if mode == "en" else "ar"
            out.append(Routed(chunk, engine, engine))
            continue
        lang, prob = detections.get(chunk.index, ("unknown", 0.0))
        other: SttLanguage = "ar" if mode == "en" else "en"
        own: SttLanguage = "en" if mode == "en" else "ar"
        if lang == other and prob >= threshold:
            out.append(Routed(chunk, other, other))
        else:
            out.append(Routed(chunk, own, own if (lang == own and prob >= confident) else None))
    return out


def route(
    chunks: list[SpeechChunk],
    mode: ResolvedMode,
    detector: LanguageDetector | None,
    wav: Path,
    threshold: float,
) -> RouteTable:
    """Engine assignment per chunk (see ``route_chunks``), as ``{"en": [...], "ar": [...]}``."""
    table: RouteTable = {"en": [], "ar": []}
    for r in route_chunks(chunks, mode, detect_all(chunks, detector, wav), threshold):
        table[r.engine].append(r.chunk)
    return table


def require_language(settings: Settings, mode: LanguageMode | str) -> None:
    """Refuse ``ar-mixed`` when the Arabic path is switched off (``stt_ar = "none"``).

    Called first thing by ``start`` and ``ingest`` (before the consent script or any gate
    question), again when the meeting is built, before re-transcription with the mode it will
    run under, and here, so the refusal comes before any audio is captured or touched. Raises
    ``LanguageRefused``, which the CLI maps to exit 2 (refusal).
    """
    if settings.stt_ar == "none" and mode == LanguageMode.ar_mixed:
        raise LanguageRefused(
            "Arabic transcription is switched off in this deployment (stt_ar = none), so "
            "--lang ar-mixed is refused before the consent script and before any audio is read; "
            "use --lang en"
        )


def transcribe_track(
    wav: Path,
    track: Literal["mic", "system", "file"],
    mode: LanguageMode | str,
    settings: Settings,
    audit: Any | None,
    *,
    engines: Engines | None = None,
    roster: Sequence[Attendee] = (),
    glossary_entries: Sequence[Any] = (),
    meeting_id: str | None = None,
    classification: str | None = None,
) -> list[RawSegment]:
    """VAD -> LID -> resolve mode -> route -> run each engine in turn, unloading after each.

    With ``stt_ar = "none"`` there is no LID and no routing: every chunk goes to the English
    engine with hint ``en``, and ``ar-mixed`` raises ``LanguageRefused`` before any audio is
    read.

    ``engines`` overrides ``build_transcribers(settings)`` (tests and callers that already hold
    engines); when it is absent the configured local weights are verified against the model
    register first (``ModelRegisterMismatch`` refuses the run). An engine is unloaded in a
    ``finally`` even when it raises. A backend with ``auto_language = True`` (the Whisper
    backends) receives ``None`` for chunks whose LID disagreed with the route, so it detects
    the language itself; every other backend is given the route's language. Emits
    ``stt.completed`` on ``audit`` (``audit.append(event, meeting_id, classification=...,
    **detail)``) when one is given; ``classification`` is the meeting's classification value,
    which a caller holding the meeting passes so the event is not recorded unclassified.
    Returns segments sorted by start time.
    """
    require_language(settings, mode)
    arabic = settings.stt_ar != "none"
    if engines is None:
        ensure_verified(settings, audit, meeting_id=meeting_id)
    en, ar, detector = (
        engines
        if engines is not None
        else build_transcribers(settings, roster=roster, glossary_entries=glossary_entries)
    )
    chunks = speech_chunks(wav, track=track)
    if arabic:
        detections = detect_all(chunks, detector, wav)
        resolved = resolve_mode(mode, detector, wav, chunks, detections=detections)
        routed = route_chunks(chunks, resolved, detections, settings.stt_lid_threshold)
    else:
        # English only: no language ID, so a chunk it would misread as Arabic cannot be sent
        # to an engine that is not there, and Whisper decodes every chunk as English.
        resolved = "en"
        routed = [Routed(c, "en", "en") for c in chunks]
    segments: list[RawSegment] = []
    used: dict[str, str] = {}
    counts: dict[str, int] = {"en": 0, "ar": 0}
    for language, engine in (("en", en), ("ar", ar)):
        mine = [r for r in routed if r.engine == language]
        counts[language] = len(mine)
        if not mine:
            continue
        auto = bool(getattr(engine, "auto_language", False))
        groups: dict[SttLanguage | None, list[SpeechChunk]] = {}
        for r in mine:
            groups.setdefault(r.hint if auto else language, []).append(r.chunk)
        try:
            for hint, group in groups.items():
                segments.extend(engine.transcribe(wav, group, hint))
            used[f"stt_{language}"] = getattr(engine, "engine", engine.name)
        finally:
            engine.unload()
    segments.sort(key=lambda s: (s.start, s.end))
    log.info("stt.track_completed", track=track, mode=resolved, segments=len(segments))
    if audit is not None:
        audit.append(
            "stt.completed",
            meeting_id,
            classification=classification,
            track=track,
            mode=resolved,
            chunks=len(chunks),
            routed=counts,
            unhinted=sum(1 for r in routed if r.hint is None),
            segments=len(segments),
            engines=used,
        )
    return segments


def merge_tracks(tracks: dict[str, list[RawSegment]]) -> list[Segment]:
    """Interleave per-track segments by time and assign ids ``S0001..``.

    Keys must be ``mic``, ``system`` or ``file``. ``mic`` segments become speaker ``ME``
    (``self``); ``system`` segments get the placeholder label ``SPEAKER_00`` (``label``) for
    diarisation or the reviewer to resolve; ``file`` segments are ``unknown``. Raises
    ``ValueError`` on any other key.
    """
    flat: list[tuple[str, RawSegment]] = []
    for track, segs in tracks.items():
        if track not in _TRACK_SPEAKER:
            raise ValueError(f"unknown track {track!r}; expected mic, system or file")
        flat.extend((track, s) for s in segs)
    flat.sort(key=lambda item: (item[1].start, item[1].end, item[0]))
    out: list[Segment] = []
    for i, (track, s) in enumerate(flat, start=1):
        speaker, kind = _TRACK_SPEAKER[track]
        out.append(
            Segment(
                id=segment_id(i),
                start=s.start,
                end=s.end,
                speaker=speaker,
                speaker_kind=kind,  # type: ignore[arg-type]
                language=s.language,
                text=s.text,
                confidence=s.confidence,
                track=track,  # type: ignore[arg-type]
                engine=s.engine,
                words=list(s.words),
            )
        )
    return out
