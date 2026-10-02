"""STT protocols and the helpers every backend shares.

The Whisper failure modes measured in development are handled here once so each backend
stays thin: vocabulary priming from the roster and glossary (``vocab_prompt``), clipping to the
chunk and to the audio duration, dropping consecutive duplicate segments, and tagging language by
script ratio (Arabic share >= 0.85 -> ``ar``, <= 0.15 -> ``en``, else ``mixed``).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import soundfile as sf

from praktika.models import Attendee, RawSegment, SpeechChunk, Word

SttLanguage = Literal["en", "ar"]
LanguageTag = Literal["en", "ar", "mixed", "unknown"]

SAMPLE_RATE = 16_000
# Whisper keeps only the last n_text_ctx // 2 - 1 = 223 prompt tokens, so the budget is in
# tokens (conservative chars-per-token estimates; Arabic tokenises ~1.6 chars/token) and
# names are emitted first and protected: terms are dropped from the end, never names.
PROMPT_MAX_TOKENS = 200
PROMPT_MAX_CHARS = 700  # secondary guard for the Latin prompt
ARABIC_CHARS_PER_TOKEN = 1.6
LATIN_CHARS_PER_TOKEN = 3.6
FRAME = {
    "en": ("Internal meeting.", "Attendees: ", "Terms: "),
    "ar": ("اجتماع داخلي.", "الحضور: ", "المصطلحات: "),
}
AR_SHARE_AR = 0.85
AR_SHARE_EN = 0.15
_WS = re.compile(r"\s+")


class Transcriber(Protocol):
    """One STT engine. ``transcribe`` returns file-absolute times; ``unload`` frees weights.

    ``language`` is the route's hint; a backend that sets ``auto_language = True`` accepts
    ``None`` and detects the language itself, every other backend is always given a code.
    """

    name: str

    def transcribe(
        self, wav: Path, chunks: list[SpeechChunk], language: SttLanguage | None
    ) -> list[RawSegment]: ...

    def unload(self) -> None: ...


class LanguageDetector(Protocol):
    """Per-chunk language identification: ``(iso_code, probability)``."""

    def detect(self, wav: Path, chunk: SpeechChunk) -> tuple[str, float]: ...


# --------------------------------------------------------------------------- vocabulary priming


def _is_arabic(ch: str) -> bool:
    return "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ" or "ﭐ" <= ch <= "﷿"


def _has_arabic(text: str) -> bool:
    return any(_is_arabic(ch) for ch in text)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key.casefold() not in seen:
            seen.add(key.casefold())
            out.append(key)
    return out


def _glossary_terms(entries: Sequence[Any], language: SttLanguage) -> list[str]:
    """Canonical terms (plus the first Arabic variant for ``ar``) from any entry shape."""
    terms: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            terms.append(entry)
            continue
        canonical = getattr(entry, "canonical", None) or (
            entry.get("canonical") if isinstance(entry, dict) else None
        )
        if canonical:
            terms.append(str(canonical))
        if language == "ar":
            arabic = getattr(entry, "arabic_variants", None) or (
                entry.get("arabic_variants") if isinstance(entry, dict) else None
            )
            if arabic:
                terms.append(str(arabic[0]))
    return terms


def prompt_tokens(text: str) -> int:
    """Conservative Whisper token estimate of ``text`` (Arabic ~1.6, Latin ~3.6 chars/token)."""
    if not text:
        return 0
    arabic = sum(1 for ch in text if _is_arabic(ch))
    other = len(text) - arabic - len(_WS.findall(text))
    return int(arabic / ARABIC_CHARS_PER_TOKEN + max(0, other) / LATIN_CHARS_PER_TOKEN + 0.999)


def _fits(text: str) -> bool:
    return prompt_tokens(text) <= PROMPT_MAX_TOKENS and len(text) <= PROMPT_MAX_CHARS


def vocab_prompt(
    roster: Sequence[Attendee], glossary_entries: Sequence[Any], language: SttLanguage
) -> str:
    """Build the Whisper ``initial_prompt`` that primes names and house vocabulary.

    Names come from the roster (room devices excluded); for ``ar`` an attendee's Arabic-script
    alias is used where one exists, otherwise the Latin name, and the prompt is framed in
    Arabic. Glossary entries contribute their canonical term (and first Arabic variant for
    ``ar``); entries may be ``GlossaryEntry`` objects, dicts or plain strings. Deterministic,
    deduplicated, and budgeted in *tokens* (``PROMPT_MAX_TOKENS``, inside Whisper's 223-token
    prompt window) by dropping terms from the end; names are never dropped unless they alone
    exceed the budget, in which case the last names go. Empty inputs give an empty string.
    """
    names: list[str] = []
    for a in roster:
        if a.status == "room":
            continue
        arabic = next((al for al in a.aliases if _has_arabic(al)), None)
        names.append(arabic if (language == "ar" and arabic) else a.name)
    names = _dedupe(names)
    terms = _dedupe(t for t in _glossary_terms(glossary_entries, language) if t not in names)
    if not names and not terms:
        return ""
    head, attendees, terms_label = FRAME[language]

    def build(n: list[str], t: list[str]) -> str:
        parts = [head]
        if n:
            parts.append(attendees + ", ".join(n) + ".")
        if t:
            parts.append(terms_label + ", ".join(t) + ".")
        return " ".join(parts)

    while terms and not _fits(build(names, terms)):
        terms.pop()
    while names and not _fits(build(names, [])) and not terms:
        names.pop()
    return build(names, terms)


# --------------------------------------------------------------------------- text helpers


def language_tag(text: str) -> LanguageTag:
    """Tag ``text`` by the share of Arabic letters among its letters.

    Text without letters (digits, punctuation, empty) is ``unknown``.
    """
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return "unknown"
    share = sum(1 for ch in letters if _is_arabic(ch)) / len(letters)
    if share >= AR_SHARE_AR:
        return "ar"
    if share <= AR_SHARE_EN:
        return "en"
    return "mixed"


def logprob_to_confidence(avg_logprob: float | None) -> float | None:
    """Map Whisper's ``avg_logprob`` to 0..1 via ``min(1, max(0, 1 + lp))``; None stays None."""
    if avg_logprob is None:
        return None
    return min(1.0, max(0.0, 1.0 + float(avg_logprob)))


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


def clean_segments(segments: list[RawSegment], *, duration_s: float) -> list[RawSegment]:
    """Sort by time, drop empty text, clip to the audio and drop consecutive duplicates.

    A segment starting at or after ``duration_s`` is a padding hallucination and is dropped; an
    end beyond ``duration_s`` is clamped. A segment whose normalised text equals the previous
    kept segment's text is dropped (Whisper's end-of-window repetition and chunk-overlap echoes).
    """
    out: list[RawSegment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        text = seg.text.strip()
        if not text or seg.start >= duration_s:
            continue
        end = min(seg.end, duration_s)
        if end <= seg.start:
            continue
        if out and _norm(out[-1].text) == _norm(text):
            continue
        words = [w for w in seg.words if w.start < duration_s]
        out.append(seg.model_copy(update={"text": text, "end": end, "words": words}))
    return out


# --------------------------------------------------------------------------- audio helpers


def read_chunk(wav: Path, chunk: SpeechChunk, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read ``chunk``'s samples from ``wav`` as mono float32 at ``sample_rate``.

    Raises ``ValueError`` when the file's rate differs (conversion happens upstream in
    ``audio.convert.to_wav16k``; backends never resample silently).
    """
    with sf.SoundFile(str(wav)) as fh:
        if fh.samplerate != sample_rate:
            raise ValueError(f"{wav} is {fh.samplerate} Hz; expected {sample_rate} Hz")
        start = max(0, int(round(chunk.start * fh.samplerate)))
        end = min(fh.frames, int(round(chunk.end * fh.samplerate)))
        fh.seek(start)
        data = fh.read(max(0, end - start), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)


def whisper_result_to_raw(
    segments: Iterable[dict[str, Any]], chunk: SpeechChunk, *, engine: str
) -> list[RawSegment]:
    """Convert Whisper-shaped segment dicts (times relative to ``chunk.start``) to ``RawSegment``.

    Accepts the dict shape shared by mlx-whisper, faster-whisper (after ``_asdict``) and the
    OpenAI ``verbose_json`` response: ``start``, ``end``, ``text``, optional ``avg_logprob`` and
    ``words`` (``word``/``start``/``end``/``probability``). Segments starting at or beyond the
    chunk end are dropped and ends are clipped to the chunk bound.
    """
    length = chunk.end - chunk.start
    out: list[RawSegment] = []
    for seg in segments:
        rel_start = float(seg.get("start", 0.0))
        rel_end = float(seg.get("end", rel_start))
        text = str(seg.get("text", "")).strip()
        if not text or rel_start >= length:
            continue
        rel_end = min(rel_end, length)
        if rel_end <= rel_start:
            continue
        words = [
            Word(
                start=round(chunk.start + float(w["start"]), 3),
                end=round(chunk.start + min(float(w["end"]), length), 3),
                text=str(w.get("word", w.get("text", ""))).strip(),
                prob=w.get("probability", w.get("prob")),
            )
            for w in seg.get("words") or []
            if float(w.get("start", 0.0)) < length
        ]
        out.append(
            RawSegment(
                start=round(chunk.start + rel_start, 3),
                end=round(chunk.start + rel_end, 3),
                text=text,
                language=language_tag(text),
                confidence=logprob_to_confidence(seg.get("avg_logprob")),
                words=words,
                engine=engine,
            )
        )
    return out
