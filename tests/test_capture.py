"""Tests for ``praktika.audio.capture`` (control C-04).

No audio device is touched: ``MicCapturer`` gets a fake stream, ``SckCapturer`` gets a Python
stub that speaks the helper's JSON status protocol.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from praktika.audio import capture
from praktika.audio.capture import MicCapturer, SckCapturer, Track, helper_is_signed, purge_file
from praktika.errors import PraktikaError


def _signed_stub(helper: Path, **kw: Any) -> str:
    return "Developer ID Application: Test"


RATE = 16_000


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


class FakeStream:
    """Stands in for ``sounddevice.InputStream``: the test pushes blocks through ``feed``."""

    instances: list[FakeStream] = []

    def __init__(self, callback: Callable[..., None]) -> None:
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def feed(self, block: np.ndarray) -> None:
        assert self.started and not self.stopped
        self.callback(block.reshape(-1, 1), len(block), None, None)


def _tone(seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def mic() -> MicCapturer:
    FakeStream.instances.clear()
    return MicCapturer(device="Fake Mic", sample_rate=RATE, stream_factory=FakeStream)


# --------------------------------------------------------------------------- MicCapturer


def test_mic_capturer_with_fake_stream(mic: MicCapturer, tmp_path: Path) -> None:
    out_dir = tmp_path / "meeting"
    mic.start(out_dir)
    stream = FakeStream.instances[-1]
    path = out_dir / "mic.wav"

    assert stream.started and path.exists() and _mode(path) == 0o600
    assert mic.levels() == (0.0, 0.0)
    for _ in range(3):
        stream.feed(_tone(0.5))
    assert 0.3 < mic.levels()[1] < 0.4, "RMS of a 0.5-amplitude sine is about 0.35"
    assert mic.elapsed_s() == pytest.approx(1.5)

    tracks = mic.stop()

    assert tracks == [Track(name="mic", path=path, sample_rate=RATE)]
    assert stream.stopped and stream.closed
    info = sf.info(path)
    assert (info.samplerate, info.channels, info.subtype) == (RATE, 1, "PCM_16")
    assert info.frames == int(1.5 * RATE)
    assert _mode(path) == 0o600
    assert mic.stop() == tracks, "stop is idempotent"


def test_mic_capturer_refuses_double_start(mic: MicCapturer, tmp_path: Path) -> None:
    mic.start(tmp_path)
    with pytest.raises(PraktikaError, match="already running"):
        mic.start(tmp_path)
    mic.abort()


def test_abort_overwrites_and_unlinks(mic: MicCapturer, tmp_path: Path) -> None:
    mic.start(tmp_path)
    FakeStream.instances[-1].feed(_tone(1.0))
    path = tmp_path / "mic.wav"
    fd = os.open(path, os.O_RDONLY)  # keep the inode open to inspect the overwrite
    try:
        mic.abort()
        assert not path.exists()
        size = os.fstat(fd).st_size
        assert size > 1000
        assert os.read(fd, size) == b"\0" * size, "content zeroed before unlink"
    finally:
        os.close(fd)
    assert mic.stop() == [], "nothing to hand over after an abort"
    assert not path.exists()


def test_purge_file_zeroes_then_unlinks_and_ignores_missing(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"secret audio")
    fd = os.open(path, os.O_RDONLY)
    try:
        purge_file(path)
        assert os.read(fd, 100) == b"\0" * 12 and not path.exists()
    finally:
        os.close(fd)
    purge_file(path)  # already gone: no error


def test_silence_warning(mic: MicCapturer, tmp_path: Path) -> None:
    mic.start(tmp_path)
    stream = FakeStream.instances[-1]
    quiet = (1e-4 * np.random.default_rng(0).standard_normal(RATE)).astype(np.float32)  # -80 dBFS
    for _ in range(59):
        stream.feed(quiet)
    assert mic.check_silence() is False, "no verdict before 60 s of audio"
    stream.feed(quiet)
    stream.feed(quiet)
    assert mic.check_silence() is True
    assert mic.check_silence() is True, "stays true while the track stays silent"
    stream.feed(_tone(1.0))
    assert mic.check_silence() is True, "1 s of tone in 62 s is still > 95 % silence"
    for _ in range(3):
        stream.feed(_tone(1.0))
    assert mic.check_silence() is False, "4 s of tone in 65 s (93.8 % silence) clears the verdict"
    mic.abort()


def test_silence_warning_not_raised_for_speech_level(mic: MicCapturer, tmp_path: Path) -> None:
    mic.start(tmp_path)
    stream = FakeStream.instances[-1]
    for _ in range(61):
        stream.feed(_tone(1.0, amp=0.05))
    assert mic.check_silence() is False
    assert mic.check_silence(ratio=0.0) is False
    mic.abort()


# --------------------------------------------------------------------------- SckCapturer

STUB = '''#!{python}
"""Stub capture helper: speaks the JSON status protocol and writes two WAV files."""
import json, os, signal, sys, time, wave

args = sys.argv[1:]
out = args[args.index("--out") + 1]
rate = int(args[args.index("--rate") + 1])
paths = {{n: os.path.join(out, f"{{n}}.wav") for n in ("system", "mic")}}
files = {{}}
for name, path in paths.items():
    w = wave.open(path, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    files[name] = w
os.chmod(paths["system"], 0o644)  # a careless helper: the parent must fix the mode

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()

tracks = [{{"name": n, "path": p, "sample_rate": rate}} for n, p in paths.items()]
emit({{"event": "started", "tracks": tracks}})
sys.stdout.write("not json noise\\n"); sys.stdout.flush()
stop = False
def on_int(*_):
    global stop
    stop = True
signal.signal(signal.SIGINT, on_int)
while not stop:
    for w in files.values():
        w.writeframes(b"\\x01\\x00" * 160)
    emit({{"event": "level", "system": 0.3, "mic": 0.1}})
    time.sleep(0.02)
for w in files.values():
    w.close()
emit({{"event": "closed", "tracks": tracks}})
'''


@pytest.fixture
def stub_helper(tmp_path: Path) -> Path:
    helper = tmp_path / "praktika-capture"
    helper.write_text(STUB.format(python=sys.executable), encoding="utf-8")
    helper.chmod(0o755)
    return helper


def _wait(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_sck_capturer_parses_status_and_stops(
    stub_helper: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "helper_is_signed", _signed_stub)
    cap = SckCapturer(stub_helper, sample_rate=RATE)
    out_dir = tmp_path / "meeting"

    cap.start(out_dir)
    assert _wait(lambda: cap.levels() == (0.3, 0.1)), "level events parsed while running"
    assert _wait(lambda: (out_dir / "mic.wav").stat().st_size > 1000)

    tracks = cap.stop()

    assert sorted(t.name for t in tracks) == ["mic", "system"]
    for t in tracks:
        assert t.path.parent == out_dir and t.sample_rate == RATE
        info = sf.info(t.path)
        assert info.samplerate == RATE and info.frames > 0, "helper closed the files cleanly"
        assert _mode(t.path) == 0o600, "modes fixed after the helper closed"
    assert cap.errors == []
    assert cap._proc is None
    assert cap.stop() == tracks, "stop is idempotent"


def test_sck_abort_purges_both_tracks(
    stub_helper: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "helper_is_signed", _signed_stub)
    cap = SckCapturer(stub_helper)
    cap.start(tmp_path)
    assert _wait(lambda: (tmp_path / "system.wav").exists())
    cap.abort()
    assert not (tmp_path / "system.wav").exists() and not (tmp_path / "mic.wav").exists()


def test_sck_refuses_unsigned_helper(stub_helper: Path, tmp_path: Path) -> None:
    cap = SckCapturer(stub_helper)
    with pytest.raises(PraktikaError, match="not Developer ID signed"):
        cap.start(tmp_path)
    assert not (tmp_path / "mic.wav").exists(), "helper never launched"
    assert cap.stop() == []


def test_sck_start_failure_when_helper_not_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "helper"
    helper.write_text("not executable", encoding="utf-8")
    monkeypatch.setattr(capture, "helper_is_signed", _signed_stub)
    with pytest.raises(PraktikaError, match="cannot launch"):
        SckCapturer(helper).start(tmp_path)


def test_sck_error_event_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = tmp_path / "errhelper"
    helper.write_text(
        f"#!{sys.executable}\nimport json,sys\n"
        'print(json.dumps({"event":"error","message":"TCC denied"}))\n'
        'print(json.dumps({"event":"closed","tracks":[]}))\n',
        encoding="utf-8",
    )
    helper.chmod(0o755)
    monkeypatch.setattr(capture, "helper_is_signed", _signed_stub)
    cap = SckCapturer(helper)
    cap.start(tmp_path)
    assert _wait(lambda: cap.errors == ["TCC denied"])
    assert cap.stop() == []


# --------------------------------------------------------------------------- helper_is_signed


def _completed(returncode: int, stderr: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["codesign"], returncode=returncode, stdout="", stderr=stderr
    )


GOOD_SHOW = (
    "Executable=/x\nIdentifier=local.praktika.capture\nTeamIdentifier=ABCDE12345\n"
    "Authority=Developer ID Application: Acme Bank (ABCDE12345)\n"
    "Authority=Developer ID Certification Authority\nAuthority=Apple Root CA\n"
)


def _fake_codesign(
    calls: list[list[str]], *, verify_rc: int = 0, show: str = GOOD_SHOW, show_rc: int = 0
) -> Any:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[1] == "--verify":
            return _completed(verify_rc, "" if verify_rc == 0 else "invalid signature")
        return _completed(show_rc, show)

    return fake_run


def test_helper_is_signed_verifies_then_parses_developer_id(
    stub_helper: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(capture.subprocess, "run", _fake_codesign(calls))
    identity = helper_is_signed(stub_helper, team_id="ABCDE12345")
    assert identity == "Developer ID Application: Acme Bank (ABCDE12345)"
    assert calls[0][:4] == [capture.CODESIGN, "--verify", "--strict", "--deep"]
    assert calls[0][0] == "/usr/bin/codesign" and calls[0][-1] == str(stub_helper)
    assert calls[1][:3] == [capture.CODESIGN, "-dv", "--verbose=4"]
    assert helper_is_signed(stub_helper) == identity, "team id optional for the check itself"


def test_helper_is_signed_rejects_tampered_wrong_team_adhoc_and_missing(
    stub_helper: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    # 1. --verify fails: a binary modified after signing still displays its authorities
    monkeypatch.setattr(capture.subprocess, "run", _fake_codesign(calls, verify_rc=1))
    assert helper_is_signed(stub_helper, team_id="ABCDE12345") is None
    assert len(calls) == 1, "no -dv after a failed --verify"
    # 2. another developer's Developer ID signature
    other = GOOD_SHOW.replace("ABCDE12345", "ZZZZZ99999")
    monkeypatch.setattr(capture.subprocess, "run", _fake_codesign([], show=other))
    assert helper_is_signed(stub_helper, team_id="ABCDE12345") is None
    # 3. ad-hoc / unsigned / no codesign / missing file
    monkeypatch.setattr(capture.subprocess, "run", _fake_codesign([], show="Signature=adhoc\n"))
    assert helper_is_signed(stub_helper) is None
    monkeypatch.setattr(capture.subprocess, "run", _fake_codesign([], verify_rc=1, show_rc=1))
    assert helper_is_signed(stub_helper) is None

    def no_codesign(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("codesign")

    monkeypatch.setattr(capture.subprocess, "run", no_codesign)
    assert helper_is_signed(stub_helper) is None
    assert helper_is_signed(tmp_path / "absent") is None


def test_sck_capturer_passes_team_id(stub_helper: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def fake(helper: Path, *, team_id: str | None = None) -> str | None:
        seen.append(team_id)
        return None

    monkeypatch.setattr(capture, "helper_is_signed", fake)
    cap = SckCapturer(stub_helper, team_id="ABCDE12345")
    with pytest.raises(PraktikaError, match="ABCDE12345"):
        cap.start(stub_helper.parent)
    assert seen == ["ABCDE12345"]


def test_helper_is_signed_real_codesign_rejects_python_stub(stub_helper: Path) -> None:
    """Whatever the host (codesign present or not), a plain script is never accepted."""
    assert helper_is_signed(stub_helper) is None
