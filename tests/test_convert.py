"""Tests for ``praktika.audio.convert``."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from praktika.audio import convert
from praktika.errors import FfmpegError

RATE = 16_000


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _write(path: Path, audio: np.ndarray, rate: int = RATE) -> Path:
    sf.write(path, audio.astype(np.float32), rate)
    return path


def _tone(seconds: float, rate: int = RATE, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * 440.0 * t)


def test_to_wav16k_tone(tone_wav: Path, tmp_path: Path) -> None:
    dst = tmp_path / "nested" / "out.wav"
    info = convert.to_wav16k(tone_wav, dst)

    assert info.path == dst and dst.exists()
    assert info.sample_rate == RATE and info.channels == 1
    assert info.duration_s == pytest.approx(3.0, abs=0.01)
    assert info.sha256 == hashlib.sha256(dst.read_bytes()).hexdigest()
    assert _mode(dst) == 0o600
    # Round trip: the converted file is 16 kHz mono PCM-16 and still holds the tone.
    audio, rate = sf.read(dst, dtype="float32")
    assert rate == RATE and audio.ndim == 1
    assert float(np.abs(audio).max()) > 0.4


def test_to_wav16k_resamples_stereo_input(tmp_path: Path) -> None:
    src_rate = 44_100
    t = np.arange(int(1.5 * src_rate)) / src_rate
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 300 * t), 0.3 * np.sin(2 * np.pi * 500 * t)], 1)
    src = _write(tmp_path / "stereo.wav", stereo, src_rate)

    info = convert.to_wav16k(src, tmp_path / "mono.wav")

    assert (info.sample_rate, info.channels) == (RATE, 1)
    assert info.duration_s == pytest.approx(1.5, abs=0.01)


def test_to_wav16k_overwrites_existing_world_readable_file(tone_wav: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out.wav"
    dst.write_bytes(b"stale")
    os.chmod(dst, 0o644)
    convert.to_wav16k(tone_wav, dst)
    assert _mode(dst) == 0o600 and dst.stat().st_size > 5


def test_ffmpeg_missing_error(tone_wav: Path, tmp_path: Path) -> None:
    dst = tmp_path / "out.wav"
    with pytest.raises(FfmpegError, match="not executable"):
        convert.to_wav16k(tone_wav, dst, ffmpeg="praktika-no-such-ffmpeg-binary")
    assert not dst.exists(), "a failed conversion must not leave a file behind"


def test_ffmpeg_failure_carries_stderr(tmp_path: Path) -> None:
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is not audio at all")
    dst = tmp_path / "out.wav"
    with pytest.raises(FfmpegError, match="exited with") as exc_info:
        convert.to_wav16k(bad, dst)
    assert "not_audio.wav" in str(exc_info.value)
    assert not dst.exists()


def test_missing_source_is_ffmpeg_error(tmp_path: Path) -> None:
    with pytest.raises(FfmpegError, match="does not exist"):
        convert.to_wav16k(tmp_path / "absent.m4a", tmp_path / "out.wav")


def test_duration_s(tone_wav: Path, tmp_path: Path) -> None:
    assert convert.duration_s(tone_wav) == pytest.approx(3.0, abs=0.01)
    empty = _write(tmp_path / "empty.wav", np.zeros(0))
    assert convert.duration_s(empty) == 0.0


def test_is_mostly_silent(tmp_path: Path) -> None:
    silent = _write(tmp_path / "silent.wav", np.zeros(RATE * 2))
    tone = _write(tmp_path / "tone.wav", _tone(2.0))
    # 96% silence + 4% tone: silent at the default ratio, not at a stricter one.
    mixed = _write(tmp_path / "mixed.wav", np.concatenate([np.zeros(RATE * 24), _tone(1.0)]))
    empty = _write(tmp_path / "empty.wav", np.zeros(0))

    assert convert.is_mostly_silent(silent) is True
    assert convert.is_mostly_silent(tone) is False
    assert convert.is_mostly_silent(mixed) is True
    assert convert.is_mostly_silent(mixed, ratio=0.99) is False
    assert convert.is_mostly_silent(empty) is True


def test_is_mostly_silent_threshold_respected(tmp_path: Path) -> None:
    quiet = _write(tmp_path / "quiet.wav", _tone(2.0, amp=10 ** (-45 / 20)))  # about -48 dBFS
    assert convert.is_mostly_silent(quiet, threshold_db=-50.0) is False
    assert convert.is_mostly_silent(quiet, threshold_db=-40.0) is True


def test_file_sha256_matches_hashlib(tone_wav: Path) -> None:
    assert convert.file_sha256(tone_wav) == hashlib.sha256(tone_wav.read_bytes()).hexdigest()
