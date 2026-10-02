"""A failed or refused run never destroys retained audio (C-05 without data loss).

Found in review: ``praktika transcribe`` on a meeting stored as ``ar-mixed`` before Arabic was
switched off, or with the speech server down, overwrote and deleted the meeting's only retained
WAV (the in-place file was purged as if the run had converted it), left the media row live and
the meeting at ``transcribing``, and audited nothing. Everything here runs offline with the
conftest fakes.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import TONE_WAV, FakeCapturer, FakeDetector, FakeLLM, FakeTranscriber
from typer.testing import CliRunner

from praktika import consent
from praktika.audio.convert import AudioInfo
from praktika.cli import app, steps
from praktika.cli import context as ctx
from praktika.cli import gate_prompts as gp
from praktika.config import Settings
from praktika.errors import PraktikaError
from praktika.ingest import audio_file
from praktika.ingest.audio_file import ingest_tracks
from praktika.models import (
    Classification,
    LanguageMode,
    Meeting,
    MeetingState,
    MeetingType,
    Platform,
    RawSegment,
    SpeechChunk,
)
from praktika.store.repo import SqliteStore
from praktika.stt import router

runner = CliRunner()
GATE = [
    "--notified",
    "--no-objections",
    "--method",
    "chat",
    "--teams-transcription-started",
    "--purpose",
    "Minutes for the data team weekly meeting",
    "--ack-all-scope",
]


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def audit_events(settings: Settings) -> list[dict[str, Any]]:
    path = settings.data_dir / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln]


def store_of(settings: Settings) -> SqliteStore:
    return SqliteStore(settings.data_dir / "praktika.db")


class _Holder:
    """The settings every command sees; a test swaps them to model an upgrade."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings


class _SpeechServerDown:
    """An English engine whose server answers 503, as the HTTP backend reports it."""

    name = "http"
    auto_language = True

    def transcribe(self, wav: Path, chunks: list[SpeechChunk], language: str | None) -> list:
        raise PraktikaError("STT server returned 503: unavailable")

    def unload(self) -> None:
        return None


class _Audit:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def append(self, event: str, meeting_id: str | None = None, **detail: Any) -> None:
        self.events.append((event, detail))


@pytest.fixture
def env(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch, fixed_key: bytes) -> _Holder:
    """CLI collaborators, ffmpeg-free conversion, one VAD chunk and fake engines."""
    holder = _Holder(tmp_settings)
    monkeypatch.setattr(ctx, "load_settings", lambda: holder.settings)
    monkeypatch.setattr(ctx, "vault_key", lambda settings=None: fixed_key)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: FakeLLM())

    def convert(src: Path, dst: Path, *, ffmpeg: str | None = None) -> AudioInfo:
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o600)
        return AudioInfo(path=dst, sample_rate=16000, channels=1, duration_s=3.0, sha256="a" * 64)

    en = FakeTranscriber(
        [
            RawSegment(
                start=0.0,
                end=2.9,
                text="Good morning everyone, let us start.",
                language="en",
                confidence=0.9,
                engine="fake-en",
            )
        ]
    )
    monkeypatch.setattr(audio_file, "to_wav16k", convert)
    monkeypatch.setattr(
        router,
        "speech_chunks",
        lambda wav, **kw: [SpeechChunk(index=0, track=kw.get("track", "file"), start=0, end=2.9)],
    )
    monkeypatch.setattr(
        router,
        "build_transcribers",
        lambda settings, **kw: (en, FakeTranscriber([]), FakeDetector("en")),
    )
    return holder


def _ingested(env: _Holder, lang: str = "en") -> tuple[Meeting, Path]:
    """Ingest the tone fixture; returns the stored meeting and its retained WAV."""
    result = invoke("ingest", str(TONE_WAV), "--lang", lang, *GATE)
    assert result.exit_code == 0, result.output
    store = store_of(env.settings)
    (meeting,) = store.list_meetings()
    (media,) = store.list_media(meeting.id)
    assert media.path.exists()
    return meeting, media.path


def _meeting(state: MeetingState = MeetingState.transcribing) -> Meeting:
    return Meeting(
        id="M-20260925-abcd",
        title="t",
        meeting_type=MeetingType.general,
        classification=Classification.internal,
        language_mode=LanguageMode.en,
        platform=Platform.teams,
        started_at=datetime.now(UTC),
        organiser="o",
        roster=[],
        state=state,
    )


# --------------------------------------------------------------------------- transcribe re-runs


def test_legacy_ar_mixed_rerun_is_refused_before_anything_changes(env: _Holder) -> None:
    """A meeting stored as ``ar-mixed`` while the Arabic path was on, re-run after the upgrade
    to ``stt_ar = none`` without ``--lang``: refused, with its audio, row and state untouched."""
    meeting, wav = _ingested(env, lang="ar-mixed")  # tmp_settings: stt_ar = fake
    before = store_of(env.settings).get_meeting(meeting.id)
    assert before is not None and before.state is MeetingState.draft_ready
    env.settings = env.settings.model_copy(update={"stt_ar": "none"})
    n_events = len(audit_events(env.settings))

    result = invoke("transcribe", meeting.id, "--no-diarize")

    assert result.exit_code == 2, "a refusal, not a failure"
    assert "Arabic transcription is switched off" in result.output
    assert wav.exists(), "the retained audio survives a refused re-run"
    store = store_of(env.settings)
    (media,) = store.list_media(meeting.id)
    assert media.deleted_at is None
    after = store.get_meeting(meeting.id)
    assert after is not None and after.state is MeetingState.draft_ready
    assert after.language_mode is LanguageMode.ar_mixed
    assert audit_events(env.settings)[n_events:] == [], "nothing ran, so nothing is audited"

    result = invoke("transcribe", meeting.id, "--lang", "en", "--no-diarize")  # the advice works
    assert result.exit_code == 0, result.output
    assert wav.exists()
    rerun = store_of(env.settings).get_meeting(meeting.id)
    assert rerun is not None and rerun.language_mode is LanguageMode.en


def test_stt_failure_on_rerun_keeps_audio_restores_state_and_audits(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The speech server is down during ``transcribe --lang en`` on a legacy ``ar-mixed``
    meeting: the WAV and its media row survive, the meeting goes back to ``draft_ready`` in
    its old language mode, and ``ingest.failed`` records the stage and the exception type."""
    meeting, wav = _ingested(env, lang="ar-mixed")
    env.settings = env.settings.model_copy(update={"stt_ar": "none"})
    monkeypatch.setattr(
        router,
        "build_transcribers",
        lambda settings, **kw: (_SpeechServerDown(), FakeTranscriber([]), None),
    )

    result = invoke("transcribe", meeting.id, "--lang", "en", "--no-diarize")

    assert result.exit_code == 1 and "503" in result.output
    assert wav.exists(), "a failed re-run never purges the audio it only read"
    store = store_of(env.settings)
    (media,) = store.list_media(meeting.id)
    assert media.deleted_at is None
    after = store.get_meeting(meeting.id)
    assert after is not None and after.state is MeetingState.draft_ready
    assert after.language_mode is LanguageMode.ar_mixed
    failed = [e for e in audit_events(env.settings) if e["event"] == "ingest.failed"]
    assert len(failed) == 1 and failed[0]["meeting_id"] == meeting.id
    assert failed[0]["detail"] == {
        "stage": "transcribe",
        "error": "PraktikaError",
        "purged": [],
    }
    assert "Good morning" not in json.dumps(failed[0]), "no transcript text in the event"


def test_failed_new_ingest_purges_its_own_audio_and_audits(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C-05 still holds for a new ingest: the WAV it converted is purged, and the purge is
    audited; the meeting is not left at ``transcribing``."""
    monkeypatch.setattr(
        router,
        "build_transcribers",
        lambda settings, **kw: (_SpeechServerDown(), FakeTranscriber([]), None),
    )
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "503" in result.output
    store = store_of(env.settings)
    (meeting,) = store.list_meetings()
    assert not list((env.settings.data_dir / "audio" / meeting.id).glob("*.wav"))
    assert store.list_media(meeting.id) == []
    assert meeting.state is MeetingState.created
    (failed,) = [e for e in audit_events(env.settings) if e["event"] == "ingest.failed"]
    assert failed["detail"]["purged"] == ["file.wav"]
    assert failed["detail"]["stage"] == "transcribe"
    assert TONE_WAV.exists(), "the operator's own recording is never touched"


def test_failed_capture_transcription_purges_the_captured_tracks(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``start`` hands its captured tracks to the run (nothing has registered them), so a
    failed transcription still purges them."""
    capturer = FakeCapturer(("mic",))
    monkeypatch.setattr(ctx, "build_capturer", lambda source, helper, device=None: capturer)
    monkeypatch.setattr(
        router,
        "build_transcribers",
        lambda settings, **kw: (_SpeechServerDown(), FakeTranscriber([]), None),
    )
    result = invoke("start", "--title", "Weekly", "--duration", "0.1", *GATE)
    assert result.exit_code == 1 and "503" in result.output
    (meeting,) = store_of(env.settings).list_meetings()
    assert not list((env.settings.data_dir / "audio" / meeting.id).glob("*.wav"))
    (failed,) = [e for e in audit_events(env.settings) if e["event"] == "ingest.failed"]
    assert failed["detail"]["purged"] == ["mic.wav"]


# --------------------------------------------------------------------------- ingest_tracks


def test_purge_on_error_marks_an_existing_media_row_deleted(
    env: _Holder, tmp_settings: Settings
) -> None:
    """A file the failing run wrote is purged, and a live media row naming it is marked
    deleted through ``on_purge``, so the store never lists audio that is gone."""
    rt = ctx.open_runtime(tmp_settings)
    meeting = _meeting()
    rt.store.save_meeting(meeting)
    dst = audio_file.audio_dir(tmp_settings, meeting.id) / "file.wav"
    rt.store.save_media(meeting.id, dst, "0" * 64, kind="file", delete_after=None)
    audit = _Audit()
    with pytest.raises(PraktikaError, match="503"):
        ingest_tracks(
            {"file": TONE_WAV},
            meeting,
            tmp_settings,
            audit,
            engines=(_SpeechServerDown(), FakeTranscriber([]), None),
            on_purge=steps.media_purged(rt, meeting.id),
        )
    assert not dst.exists()
    (row,) = rt.store.list_media(meeting.id)
    assert row.deleted_at is not None
    assert audit.events[-1] == (
        "ingest.failed",
        {"classification": "internal", "stage": "transcribe", "error": "PraktikaError",
         "purged": ["file.wav"]},
    )  # fmt: skip


def test_retained_file_survives_zero_retention_rerun(env: _Holder) -> None:
    """Re-transcribing a retained WAV in place under zero retention leaves it to its own row
    and timer instead of deleting it behind the row's back."""
    settings = env.settings.model_copy(
        update={"retention_audio_hours": {"internal": 0, "confidential": 72, "restricted": 0}}
    )
    meeting = _meeting()
    wav = audio_file.audio_dir(settings, meeting.id) / "file.wav"
    shutil.copyfile(TONE_WAV, wav)
    result = ingest_tracks({"file": wav}, meeting, settings, _Audit())
    assert wav.exists() and result.media == []


def test_engines_keep_stt_ar_none_when_the_english_engine_writes_arabic_script(
    env: _Holder,
) -> None:
    """With the Arabic path off, a code-switched segment the English engine wrote in Arabic
    script is still the English engine's: provenance must not name it as the Arabic engine."""
    settings = env.settings.model_copy(update={"stt_ar": "none", "stt_en": "http"})
    en = FakeTranscriber(
        [
            RawSegment(start=0.0, end=1.4, text="Good morning everyone.", language="en",
                       confidence=0.9, engine="http/whisper-large-v3"),
            RawSegment(start=1.5, end=2.9, text="إن شاء الله by Thursday", language="mixed",
                       confidence=0.9, engine="http/whisper-large-v3"),
        ]
    )  # fmt: skip
    out = ingest_tracks(
        {"file": TONE_WAV},
        _meeting(),
        settings,
        _Audit(),
        engines=(en, FakeTranscriber([]), FakeDetector("ar")),
    )
    assert out.transcript.engines == {"stt_en": "http/whisper-large-v3", "stt_ar": "none"}


# --------------------------------------------------------------------------- refused up front


def test_start_refuses_ar_mixed_before_the_consent_script(env: _Holder) -> None:
    env.settings = env.settings.model_copy(update={"stt_ar": "none"})
    result = invoke("start", "--title", "Weekly", "--lang", "ar-mixed", *GATE)
    assert result.exit_code == 2, "a refusal, not a failure"
    assert "Arabic transcription is switched off" in result.output
    assert "Consent script" not in result.output
    assert consent.SCRIPT_EN[:40] not in result.output and consent.SCRIPT_AR[:20] not in (
        result.output
    )
    assert not (env.settings.data_dir / "praktika.db").exists()


def test_ingest_refuses_ar_mixed_before_asking_the_gate(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a terminal the gate would prompt for every missing answer; the refusal comes first."""
    env.settings = env.settings.model_copy(update={"stt_ar": "none"})
    monkeypatch.setattr(ctx, "is_interactive", lambda: True)

    def no_prompt(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a gate question was asked before the refusal")

    monkeypatch.setattr(gp.Confirm, "ask", no_prompt)
    monkeypatch.setattr(gp.Prompt, "ask", no_prompt)
    result = invoke("ingest", str(TONE_WAV), "--lang", "ar-mixed")
    assert result.exit_code == 2, result.output
    assert "Arabic transcription is switched off" in result.output
    assert not (env.settings.data_dir / "praktika.db").exists()
