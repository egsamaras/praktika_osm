"""Voice-activity chunking with silero-vad.

Speech chunks are the unit of every downstream step: language identification and STT run per
chunk (Whisper detects language once per call, so short chunks give
per-utterance detection and remove end-of-audio hallucination), and Cohere segment times are the
chunk bounds. The silero model ships inside the ``silero-vad`` wheel, so nothing is downloaded;
it is loaded once per process.

``chunks_from_spans`` holds the pure merge/pad/split arithmetic so it can be tested without a
model; ``speech_chunks`` is the file-level entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from praktika.logging import get_logger
from praktika.models import SpeechChunk

log = get_logger(__name__)

SUPPORTED_RATES = (8_000, 16_000)
_MODEL: Any = None


def _get_model() -> Any:
    """Return the process-wide silero model, loading it on first use (no network)."""
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
        log.info("vad.model_loaded")
    return _MODEL


def speech_spans(audio: np.ndarray, sample_rate: int = 16_000) -> list[tuple[float, float]]:
    """Run silero on mono float32 ``audio`` and return raw (start, end) speech spans in seconds.

    ``sample_rate`` must be 8000 or 16000 (the rates silero supports); anything else raises
    ``ValueError`` rather than silently mis-detecting. Empty audio returns no spans.
    """
    if sample_rate not in SUPPORTED_RATES:
        raise ValueError(f"silero-vad supports {SUPPORTED_RATES} Hz, got {sample_rate}")
    if audio.ndim != 1:
        raise ValueError("speech_spans expects mono audio (1-D array)")
    if audio.size == 0:
        return []
    import torch
    from silero_vad import get_speech_timestamps

    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    stamps = get_speech_timestamps(
        tensor, _get_model(), sampling_rate=sample_rate, return_seconds=True
    )
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def _merge(spans: list[tuple[float, float]], min_gap_s: float) -> list[tuple[float, float]]:
    """Merge spans whose silence gap is shorter than ``min_gap_s`` (or that overlap)."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] < min_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _split(
    start: float, end: float, max_len_s: float, overlap_s: float
) -> list[tuple[float, float]]:
    """Cut one span into windows of at most ``max_len_s`` that overlap by ``overlap_s``."""
    if end - start <= max_len_s:
        return [(start, end)]
    step = max_len_s - overlap_s
    pieces: list[tuple[float, float]] = []
    cursor = start
    while True:
        piece_end = min(cursor + max_len_s, end)
        pieces.append((cursor, piece_end))
        if piece_end >= end:
            return pieces
        cursor += step


def chunks_from_spans(
    spans: list[tuple[float, float]],
    duration_s: float,
    *,
    max_len_s: float = 28.0,
    min_gap_s: float = 0.6,
    pad_s: float = 0.25,
    overlap_s: float = 0.5,
    track: Literal["mic", "system", "file"] = "file",
) -> list[SpeechChunk]:
    """Turn raw speech spans into indexed ``SpeechChunk``s.

    Steps, in order: merge spans separated by less than ``min_gap_s``; pad each by ``pad_s`` on
    both sides, clamped to ``[0, duration_s]``; re-merge any spans the padding made overlap; split
    spans longer than ``max_len_s`` into windows overlapping by ``overlap_s``. Chunks are indexed
    from 0 in time order. Raises ``ValueError`` when ``overlap_s >= max_len_s`` (no progress).
    """
    if overlap_s >= max_len_s:
        raise ValueError("overlap_s must be smaller than max_len_s")
    padded = [
        (max(0.0, s - pad_s), min(duration_s, e + pad_s)) for s, e in _merge(spans, min_gap_s)
    ]
    padded = [(s, e) for s, e in padded if e > s]
    pieces = [p for s, e in _merge(padded, 0.0) for p in _split(s, e, max_len_s, overlap_s)]
    return [
        SpeechChunk(index=i, track=track, start=round(s, 3), end=round(e, 3))
        for i, (s, e) in enumerate(pieces)
    ]


def speech_chunks(
    wav: Path,
    *,
    max_len_s: float = 28.0,
    min_gap_s: float = 0.6,
    pad_s: float = 0.25,
    overlap_s: float = 0.5,
    track: Literal["mic", "system", "file"] = "file",
) -> list[SpeechChunk]:
    """Detect speech in ``wav`` (16 kHz or 8 kHz mono) and return chunks ready for STT.

    Multi-channel input is averaged to mono. Returns an empty list for silence-only audio.
    See ``chunks_from_spans`` for the merge/pad/split rules.
    """
    audio, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / float(rate)
    spans = speech_spans(mono, int(rate))
    chunks = chunks_from_spans(
        spans,
        duration,
        max_len_s=max_len_s,
        min_gap_s=min_gap_s,
        pad_s=pad_s,
        overlap_s=overlap_s,
        track=track,
    )
    log.info(
        "vad.completed",
        file=Path(wav).name,
        duration_s=round(duration, 2),
        spans=len(spans),
        chunks=len(chunks),
    )
    return chunks
