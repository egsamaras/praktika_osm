"""Transcript chunking on speaker-turn boundaries for the map stage.

A chunk is a run of whole segments rendered in the ``render_for_llm`` line format. Chunks are
cut at speaker turns where possible, never inside a segment, and consecutive chunks overlap by
``overlap_turns`` speaker turns so a decision spoken across a cut is seen whole at least once.
"""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, ConfigDict, Field

from praktika.models import Attendee, Transcript

# Arabic block, supplement, presentation forms A and B.
_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_WHITESPACE = re.compile(r"\s")

LATIN_CHARS_PER_TOKEN = 3.6
ARABIC_CHARS_PER_TOKEN = 1.6


class Chunk(BaseModel):
    """A contiguous slice of the transcript prepared for one extraction call."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    segment_ids: list[str] = Field(min_length=1)
    text: str
    approx_tokens: int = Field(ge=0)


def estimate_tokens(text: str) -> int:
    """Rough token count: Arabic characters at 1.6 per token, everything else at 3.6.

    Arabic tokenises far more densely than Latin script in Qwen and Llama vocabularies, so
    Arabic-heavy transcripts get smaller chunks. Whitespace is not counted. Empty text is 0.
    """
    if not text:
        return 0
    arabic = len(_ARABIC.findall(text))
    other = len(text) - arabic - len(_WHITESPACE.findall(text))
    return math.ceil(arabic / ARABIC_CHARS_PER_TOKEN + max(0, other) / LATIN_CHARS_PER_TOKEN)


def _turn_starts(t: Transcript) -> list[bool]:
    segs = t.segments
    return [i == 0 or segs[i].speaker != segs[i - 1].speaker for i in range(len(segs))]


def chunk_transcript(
    t: Transcript,
    roster: list[Attendee],
    *,
    target_tokens: int = 8000,
    overlap_turns: int = 1,
) -> list[Chunk]:
    """Split ``t`` into chunks of at most ``target_tokens`` (estimated), cut on speaker turns.

    Contract: every segment appears whole in at least one chunk; chunks are in transcript order;
    a single segment larger than ``target_tokens`` becomes its own chunk; each chunk after the
    first repeats the last ``overlap_turns`` speaker turns of its predecessor unless that overlap
    would exceed a quarter of the target (or the predecessor consists of one turn). An empty
    transcript yields no chunks. ``target_tokens`` must be positive.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    segs = t.segments
    if not segs:
        return []
    lines = t.render_for_llm(roster).split("\n")
    tokens = [estimate_tokens(line) for line in lines]
    starts = _turn_starts(t)
    n = len(segs)
    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end, total = start, 0
        while end < n and (end == start or total + tokens[end] <= target_tokens):
            total += tokens[end]
            end += 1
        if end < n:
            cut = end
            while cut > start and not starts[cut]:
                cut -= 1
            if cut > start:
                end = cut
        chunks.append(
            Chunk(
                index=len(chunks),
                segment_ids=[s.id for s in segs[start:end]],
                text="\n".join(lines[start:end]),
                approx_tokens=sum(tokens[start:end]),
            )
        )
        if end >= n:
            break
        start = _next_start(start, end, starts, tokens, overlap_turns, target_tokens // 4)
    return chunks


def _next_start(
    start: int,
    end: int,
    starts: list[bool],
    tokens: list[int],
    overlap_turns: int,
    max_overlap_tokens: int,
) -> int:
    """Index where the next chunk begins: ``end`` minus up to ``overlap_turns`` whole turns.

    The overlap never reaches back to ``start`` (progress is guaranteed) and never exceeds
    ``max_overlap_tokens``.
    """
    pos = end
    for _ in range(max(0, overlap_turns)):
        candidate = pos - 1
        while candidate > start and not starts[candidate]:
            candidate -= 1
        if candidate <= start or sum(tokens[candidate:end]) > max_overlap_tokens:
            break
        pos = candidate
    return pos
