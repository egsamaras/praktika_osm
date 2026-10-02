"""Tests for ``praktika.audio.vad``.

The silero model is bundled with the wheel (no download) and loaded once per session. Pure
tones are not speech to silero, so real-model tests use slices of the synthetic TTS fixture
(macOS voices, no real people) arranged with numpy; the merge/pad/split arithmetic is tested on
``chunks_from_spans`` directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from praktika.audio import vad

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_meeting.wav"
RATE = 16_000
# One continuous English sentence of the fixture, edge to edge: "This is the data team weekly on
# the fifteenth of September." (re-measure these bounds whenever make_fixture.py changes).
SPEECH_START_S, SPEECH_END_S = 1.8, 5.4
SPEECH_S = SPEECH_END_S - SPEECH_START_S


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    """About 3.6 s of continuous synthetic English speech from the fixture."""
    audio, rate = sf.read(FIXTURE, dtype="float32")
    assert rate == RATE
    return audio[int(SPEECH_START_S * RATE) : int(SPEECH_END_S * RATE)]


def _write(path: Path, audio: np.ndarray) -> Path:
    sf.write(path, audio.astype(np.float32), RATE)
    return path


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


# --------------------------------------------------------------------------- real model


def test_silence_only_empty(tmp_path: Path) -> None:
    assert vad.speech_chunks(_write(tmp_path / "silence.wav", _silence(5.0))) == []


def test_speech_detected_with_track_and_indexes(tmp_path: Path, speech: np.ndarray) -> None:
    audio = np.concatenate([_silence(1.0), speech, _silence(2.0), speech, _silence(1.0)])
    chunks = vad.speech_chunks(_write(tmp_path / "two.wav", audio), track="mic")

    assert len(chunks) == 2
    assert [c.index for c in chunks] == [0, 1]
    assert all(c.track == "mic" for c in chunks)
    assert chunks[0].start == pytest.approx(1.0, abs=0.35)
    assert chunks[0].end == pytest.approx(1.0 + SPEECH_S, abs=0.4)
    assert chunks[1].start > chunks[0].end + 1.0


def test_gap_merge(tmp_path: Path, speech: np.ndarray) -> None:
    # Two utterances 0.4 s apart: one chunk with the default 0.6 s merge gap, two without.
    audio = np.concatenate([_silence(0.5), speech, _silence(0.4), speech, _silence(0.5)])
    path = _write(tmp_path / "gap.wav", audio)

    merged = vad.speech_chunks(path, min_gap_s=0.6)
    split = vad.speech_chunks(path, min_gap_s=0.0, pad_s=0.0)

    assert len(merged) == 1
    assert len(split) == 2
    assert merged[0].end - merged[0].start == pytest.approx(2 * SPEECH_S + 0.4 + 0.5, abs=0.5)


def test_long_chunk_split_with_overlap(tmp_path: Path, speech: np.ndarray) -> None:
    # Ten utterances 0.2 s apart merge into ~38 s of speech, which must be split at 28 s.
    parts = [_silence(0.3)]
    for _ in range(10):
        parts += [speech, _silence(0.2)]
    chunks = vad.speech_chunks(_write(tmp_path / "long.wav", np.concatenate(parts)))

    assert len(chunks) >= 2
    assert all(c.end - c.start <= 28.0 + 1e-6 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.end - b.start == pytest.approx(0.5, abs=1e-3)
        assert b.start > a.start


def test_padding_clamped(tmp_path: Path, speech: np.ndarray) -> None:
    # Speech touches both file edges: padding cannot go below 0 or beyond the duration.
    path = _write(tmp_path / "edges.wav", speech)
    duration = len(speech) / RATE
    chunks = vad.speech_chunks(path, pad_s=0.25)

    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end <= duration + 1e-6
    assert chunks[0].end == pytest.approx(duration, abs=0.3)


def test_unsupported_sample_rate_raises() -> None:
    with pytest.raises(ValueError, match="supports"):
        vad.speech_spans(np.zeros(44_100, dtype=np.float32), sample_rate=44_100)


def test_stereo_input_averaged(tmp_path: Path, speech: np.ndarray) -> None:
    stereo = np.stack([speech, speech * 0.5], axis=1)
    sf.write(tmp_path / "stereo.wav", stereo, RATE)
    assert len(vad.speech_chunks(tmp_path / "stereo.wav")) == 1


def test_model_loaded_once() -> None:
    first = vad._get_model()
    assert vad._get_model() is first


# --------------------------------------------------------------------------- pure arithmetic


def test_chunks_from_spans_split_exact() -> None:
    chunks = vad.chunks_from_spans([(0.0, 60.0)], 60.0, pad_s=0.0, max_len_s=28.0, overlap_s=0.5)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 28.0), (27.5, 55.5), (55.0, 60.0)]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_chunks_from_spans_merge_then_pad_and_clamp() -> None:
    spans = [(0.1, 2.0), (2.5, 4.0), (5.0, 9.9)]  # first gap 0.5 merges, second gap 1.0 stays
    chunks = vad.chunks_from_spans(spans, 10.0, min_gap_s=0.6, pad_s=0.25)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 4.25), (4.75, 10.0)]


def test_chunks_from_spans_padding_overlap_remerged() -> None:
    # Gap of 0.3 s survives min_gap_s=0.2 but 0.25 s padding makes the spans overlap: re-merged.
    chunks = vad.chunks_from_spans([(1.0, 2.0), (2.3, 3.0)], 5.0, min_gap_s=0.2, pad_s=0.25)
    assert [(c.start, c.end) for c in chunks] == [(0.75, 3.25)]


def test_chunks_from_spans_unsorted_input_and_empty() -> None:
    chunks = vad.chunks_from_spans([(5.0, 6.0), (1.0, 2.0)], 10.0, pad_s=0.0)
    assert [(c.start, c.end) for c in chunks] == [(1.0, 2.0), (5.0, 6.0)]
    assert vad.chunks_from_spans([], 10.0) == []


def test_chunks_from_spans_rejects_overlap_not_smaller_than_max() -> None:
    with pytest.raises(ValueError, match="overlap_s"):
        vad.chunks_from_spans([(0.0, 5.0)], 5.0, max_len_s=2.0, overlap_s=2.0)
