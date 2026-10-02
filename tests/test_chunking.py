"""Tests for ``praktika.llm.chunking``."""

from __future__ import annotations

import pytest
from conftest import make_transcript

from praktika.llm.chunking import chunk_transcript, estimate_tokens
from praktika.models import Attendee, Segment, Transcript

LINE_PREFIX_TOKENS = 12  # "[S0001 00:00:00-00:00:09 F. Khalid|en] " is about 12 tokens


def _lines(chunk_text: str) -> list[str]:
    return chunk_text.split("\n")


def test_never_splits_segment(roster: list[Attendee]) -> None:
    t = make_transcript("mixed", n=24)
    chunks = chunk_transcript(t, roster, target_tokens=90, overlap_turns=1)
    assert len(chunks) > 1
    rendered = t.render_for_llm(roster).split("\n")
    by_id = dict(zip([s.id for s in t.segments], rendered, strict=True))
    for c in chunks:
        lines = _lines(c.text)
        assert len(lines) == len(c.segment_ids)
        for sid, line in zip(c.segment_ids, lines, strict=True):
            assert line == by_id[sid], "a segment line must appear whole and unchanged"
    covered = {sid for c in chunks for sid in c.segment_ids}
    assert covered == {s.id for s in t.segments}
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_present(roster: list[Attendee]) -> None:
    t = make_transcript("en", n=16)
    chunks = chunk_transcript(t, roster, target_tokens=120, overlap_turns=1)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        shared = set(prev.segment_ids) & set(nxt.segment_ids)
        assert shared, "consecutive chunks must share the last speaker turn"
        # The shared turn is exactly the tail of the previous chunk and the head of the next.
        k = len(shared)
        assert prev.segment_ids[-k:] == nxt.segment_ids[:k]
    assert chunk_transcript(t, roster, target_tokens=120, overlap_turns=0)[1].segment_ids[
        0
    ] not in (chunks[0].segment_ids)


def test_small_transcript_one_chunk(roster: list[Attendee]) -> None:
    t = make_transcript("en", n=5)
    chunks = chunk_transcript(t, roster)
    assert len(chunks) == 1
    assert chunks[0].segment_ids == [s.id for s in t.segments]
    assert chunks[0].text == t.render_for_llm(roster)
    whole = estimate_tokens(chunks[0].text)
    assert whole <= chunks[0].approx_tokens <= whole + len(t.segments)  # per-line rounding


def test_arabic_token_estimate_heavier() -> None:
    latin = "the budget line is two hundred and fifty thousand dinars"
    arabic = "الميزانية للمرحلة الأولى مئتان وخمسون ألف دينار"
    assert len(arabic) < len(latin)
    assert estimate_tokens(arabic) > estimate_tokens(latin)
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcdefghij") == 3  # 10 chars / 3.6 rounded up


def test_oversize_segment_own_chunk(roster: list[Attendee]) -> None:
    t = make_transcript("en", n=4)
    big = Segment(
        id="S0003",
        start=20.0,
        end=29.0,
        speaker="R. Haddad",
        speaker_kind="identity",
        language="en",
        text="word " * 600,
        track="vtt",
        engine="fake",
    )
    segs = [big if s.id == "S0003" else s for s in t.segments]
    t2 = Transcript(**{**t.model_dump(), "segments": segs})
    chunks = chunk_transcript(t2, roster, target_tokens=100, overlap_turns=1)
    own = [c for c in chunks if c.segment_ids == ["S0003"]]
    assert own, "an oversize segment must become a chunk of its own"
    assert own[0].approx_tokens > 100
    assert {sid for c in chunks for sid in c.segment_ids} == {"S0001", "S0002", "S0003", "S0004"}


def test_empty_transcript_and_bad_target(roster: list[Attendee]) -> None:
    t = make_transcript("en", n=3)
    empty = Transcript(**{**t.model_dump(), "segments": []})
    assert chunk_transcript(empty, roster) == []
    with pytest.raises(ValueError):
        chunk_transcript(t, roster, target_tokens=0)
