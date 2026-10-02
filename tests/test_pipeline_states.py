"""Meeting states through a run, and an abort from another shell always winning (C-04, C-05).

Regression tests for run-state bugs: a failed run never leaves a meeting at ``transcribing``; a
successful ``praktika transcribe`` leaves it there until ``generate``, and ``approve`` refuses the
older draft meanwhile; every pipeline move is a compare-and-set, so an ``abort`` during ingest,
transcription or drafting is never overwritten and the run removes what it had stored; every
retained WAV is registered inside the purge-on-error window or purged, and every purge is audited;
``start`` refuses a host that cannot record before the consent gate. Everything is offline and
synthetic (conftest fakes, the tone fixture).
"""

from __future__ import annotations

import importlib.abc
import json
import os
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import TONE_WAV, FakeCapturer, FakeDetector, FakeLLM, FakeTranscriber
from typer.testing import CliRunner

from praktika import consent
from praktika.audio.capture import MicCapturer, Track
from praktika.audio.convert import AudioInfo
from praktika.cli import app, meeting_ops, meetings, steps
from praktika.cli import context as ctx
from praktika.config import Settings, default_data_dir
from praktika.diarize.assign import apply_names
from praktika.errors import PraktikaError
from praktika.ingest import audio_file
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
LINE = "Good morning everyone, let us start."


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def segment(text: str = LINE) -> RawSegment:
    return RawSegment(
        start=0.0, end=2.9, text=text, language="en", confidence=0.9, engine="fake-en"
    )


class _Holder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings


class _Engine:
    """An English engine that runs ``hook(wav)`` before answering (a second shell acting)."""

    name = "fake-en"
    auto_language = True

    def __init__(self, text: str = LINE, hook: Any = None, error: BaseException | None = None):
        self.text, self.hook, self.error = text, hook, error

    def transcribe(self, wav: Path, chunks: list[SpeechChunk], language: str | None) -> list:
        if self.hook is not None:
            self.hook(wav)
        if self.error is not None:
            raise self.error
        return [segment(self.text)]

    def unload(self) -> None:
        return None


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

    monkeypatch.setattr(audio_file, "to_wav16k", convert)
    monkeypatch.setattr(
        router,
        "speech_chunks",
        lambda wav, **kw: [SpeechChunk(index=0, track=kw.get("track", "file"), start=0, end=2.9)],
    )
    use_engine(monkeypatch, _Engine())
    return holder


def use_engine(monkeypatch: pytest.MonkeyPatch, engine: Any) -> None:
    monkeypatch.setattr(
        router,
        "build_transcribers",
        lambda settings, **kw: (engine, FakeTranscriber([]), FakeDetector("en")),
    )


def store_of(settings: Settings) -> SqliteStore:
    return SqliteStore(settings.data_dir / "praktika.db")


def only_meeting(settings: Settings) -> Meeting:
    (meeting,) = store_of(settings).list_meetings()
    return meeting


def state_of(settings: Settings, meeting_id: str) -> MeetingState:
    meeting = store_of(settings).get_meeting(meeting_id)
    assert meeting is not None
    return meeting.state


def wavs(settings: Settings, meeting_id: str) -> list[str]:
    return sorted(p.name for p in (settings.data_dir / "audio" / meeting_id).glob("*.wav"))


def live_media(settings: Settings, meeting_id: str) -> list[Any]:
    return [m for m in store_of(settings).list_media(meeting_id) if m.deleted_at is None]


def audit(settings: Settings, event: str | None = None) -> list[dict[str, Any]]:
    path = settings.data_dir / "audit.jsonl"
    rows = [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln]
    return [r for r in rows if event is None or r["event"] == event]


def ingested(env: _Holder) -> Meeting:
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 0, result.output
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.draft_ready
    return meeting


def fake_mic(monkeypatch: pytest.MonkeyPatch, capturer: Any = None) -> Any:
    capturer = capturer or FakeCapturer(("mic",))
    monkeypatch.setattr(ctx, "build_capturer", lambda source, helper, device=None: capturer)
    return capturer


def start(*extra: str) -> Any:
    return invoke("start", "--title", "Weekly", "--duration", "0.1", *extra, *GATE)


# --------------------------------------------------------------------------- store: compare-and-set


def _meeting(state: MeetingState = MeetingState.created) -> Meeting:
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


def test_transition_is_a_compare_and_set(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "p.db")
    m = _meeting()
    store.save_meeting(m)
    assert store.transition(
        m.id,
        MeetingState.transcribing,
        expected=[MeetingState.created],
        language_mode=LanguageMode.ar_mixed,
    )
    got = store.get_meeting(m.id)
    assert got is not None and got.state is MeetingState.transcribing
    assert got.language_mode is LanguageMode.ar_mixed
    other = SqliteStore(tmp_path / "p.db")  # a second process: the abort
    other.set_state(m.id, MeetingState.discarded)
    assert not store.transition(m.id, MeetingState.drafting, expected=[MeetingState.transcribing])
    assert store.get_meeting(m.id).state is MeetingState.discarded  # type: ignore[union-attr]
    with pytest.raises(KeyError):
        store.transition("M-20260925-ffff", MeetingState.drafting, expected=[])


def test_hold_changes_only_the_hold_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hold set from another process never puts back a state a run has moved on since."""
    store, other = SqliteStore(tmp_path / "p.db"), SqliteStore(tmp_path / "p.db")
    m = _meeting(MeetingState.drafting)
    store.save_meeting(m)
    stale = store.get_meeting(m.id)
    fresh_read = SqliteStore.get_meeting
    reads: list[str] = []

    def first_read_is_stale(self: Any, mid: str) -> Meeting | None:
        """The hold command's first read races the run finishing in another process."""
        reads.append(mid)
        if len(reads) == 1:
            other.set_state(m.id, MeetingState.draft_ready)
            return stale
        return fresh_read(self, mid)

    monkeypatch.setattr(SqliteStore, "get_meeting", first_read_is_stale)
    store.set_hold(m.id, True, "litigation", "legal")
    monkeypatch.undo()
    got = other.get_meeting(m.id)
    assert got is not None and got.legal_hold and got.state is MeetingState.draft_ready


# --------------------------------------------------------------------------- no stuck states


def test_ingest_with_no_speech_returns_to_created(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch, FakeTranscriber([]))
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "no segments" in result.output
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.created
    (media,) = live_media(env.settings, meeting.id)  # registered, so on its retention timer
    assert media.delete_after is not None and media.path.exists()


def test_empty_vtt_ingest_returns_to_created(env: _Holder, tmp_path: Path) -> None:
    vtt = tmp_path / "empty.vtt"
    vtt.write_text("WEBVTT\n\n", "utf-8")
    result = invoke("ingest", str(vtt), "--lang", "en", *GATE)
    assert result.exit_code == 1, result.output
    assert only_meeting(env.settings).state is MeetingState.created


def test_missing_teams_transcript_is_a_clean_error(env: _Holder, tmp_path: Path) -> None:
    result = invoke(
        "ingest", str(TONE_WAV), "--vtt", str(tmp_path / "nope.vtt"), "--lang", "en", *GATE
    )
    assert result.exit_code == 1 and "input file not found" in result.output
    assert isinstance(result.exception, SystemExit), "no traceback"
    assert only_meeting(env.settings).state is MeetingState.created


def test_failed_draft_returns_to_created_and_generate_recovers(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Down(FakeLLM):
        def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise PraktikaError("LLM request failed: connection refused")

    monkeypatch.setattr(ctx, "llm_client", lambda settings: _Down())
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "connection refused" in result.output
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.created
    assert f"praktika generate {meeting.id}" in result.output
    assert store_of(env.settings).get_transcript(meeting.id) is not None
    monkeypatch.setattr(ctx, "llm_client", lambda settings: FakeLLM())
    assert invoke("generate", meeting.id).exit_code == 0
    assert state_of(env.settings, meeting.id) is MeetingState.draft_ready


def test_failure_before_drafting_starts_returns_to_created(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(*args: Any, **kwargs: Any) -> Any:
        raise PraktikaError("prompt template not readable")

    monkeypatch.setattr(steps.pr, "load", unreadable)
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "not readable" in result.output
    assert only_meeting(env.settings).state is MeetingState.created


def test_failed_capture_transcription_ends_discarded_and_audited(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mic(monkeypatch)
    use_engine(monkeypatch, _Engine(error=PraktikaError("STT server returned 503")))
    result = start()
    assert result.exit_code == 1 and "503" in result.output
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.discarded
    assert wavs(env.settings, meeting.id) == []
    (failed,) = audit(env.settings, "ingest.failed")
    assert failed["detail"]["purged"] == ["mic.wav"]
    assert invoke("transcribe", meeting.id).exit_code == 1  # nothing left to re-run


def test_capture_with_no_track_ends_discarded(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mic(monkeypatch, FakeCapturer(()))
    result = start()
    assert result.exit_code == 1 and "no audio track" in result.output
    assert only_meeting(env.settings).state is MeetingState.discarded


def test_successful_transcribe_awaits_generate_and_approve_refuses_meanwhile(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingested(env)
    use_engine(monkeypatch, _Engine(text="Good morning everyone, the budget is approved."))
    result = invoke("transcribe", meeting.id, "--no-diarize")
    assert result.exit_code == 0 and "praktika generate" in result.output
    assert state_of(env.settings, meeting.id) is MeetingState.transcribing
    refused = invoke("approve", meeting.id, "--reason", "accurate")
    assert refused.exit_code == 1 and f"praktika generate {meeting.id}" in refused.output
    assert "review.approved" not in [e["event"] for e in audit(env.settings)]
    # A failed re-run afterwards puts it back to transcribing, still awaiting generate.
    use_engine(monkeypatch, _Engine(error=PraktikaError("STT server returned 503")))
    assert invoke("transcribe", meeting.id, "--no-diarize").exit_code == 1
    assert state_of(env.settings, meeting.id) is MeetingState.transcribing
    assert invoke("generate", meeting.id).exit_code == 0
    assert state_of(env.settings, meeting.id) is MeetingState.draft_ready
    assert invoke("approve", meeting.id, "--reason", "accurate").exit_code == 0
    store = store_of(env.settings)
    approved = store.latest_minutes(meeting.id)
    latest = store.get_transcript(meeting.id)
    assert approved is not None and latest is not None
    assert approved.provenance.transcript_sha256 == latest.sha256()


def test_approve_refuses_a_draft_of_an_older_transcript_after_the_state_moved_on(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review page can move the meeting to in_review (a speaker mapping) after a
    re-transcription; the draft is still built from the older words."""
    meeting = ingested(env)
    use_engine(monkeypatch, _Engine(text="Good morning everyone, the budget is approved."))
    assert invoke("transcribe", meeting.id, "--no-diarize").exit_code == 0
    store_of(env.settings).set_state(meeting.id, MeetingState.in_review)
    refused = invoke("approve", meeting.id)
    assert refused.exit_code == 1 and "older transcript" in refused.output


def test_approve_accepts_a_draft_after_a_speaker_mapping(env: _Holder) -> None:
    meeting = ingested(env)
    store = store_of(env.settings)
    transcript = store.get_transcript(meeting.id)
    assert transcript is not None
    label = transcript.segments[0].speaker
    renamed = transcript.model_copy(
        update={"segments": apply_names(transcript.segments, {label: "F. Khalid"})}
    )
    assert renamed.sha256() != transcript.sha256()
    store.save_transcript(renamed, delete_after=None)
    store.set_state(meeting.id, MeetingState.in_review)
    result = invoke("approve", meeting.id)
    assert result.exit_code == 0, result.output


def test_transcribe_refuses_a_closed_meeting(env: _Holder) -> None:
    meeting = ingested(env)
    assert invoke("approve", meeting.id).exit_code == 0
    result = invoke("transcribe", meeting.id)
    assert result.exit_code == 1 and "approved" in result.output
    assert state_of(env.settings, meeting.id) is MeetingState.approved


# --------------------------------------------------------------------------- C-05: registration


def _locked(self: Any, *args: Any, **kwargs: Any) -> Any:
    raise sqlite3.OperationalError("database is locked")


def _interrupted(self: Any, *args: Any, **kwargs: Any) -> Any:
    raise KeyboardInterrupt


@pytest.mark.parametrize("failure", [_locked, _interrupted])
def test_registration_failure_leaves_no_unregistered_wav(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, failure: Any
) -> None:
    monkeypatch.setattr(SqliteStore, "save_media", failure)
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code != 0
    meeting = only_meeting(env.settings)
    assert wavs(env.settings, meeting.id) == []
    assert store_of(env.settings).list_media(meeting.id) == []
    assert meeting.state is MeetingState.created
    (failed,) = audit(env.settings, "ingest.failed")
    assert failed["detail"]["stage"] == "register" and failed["detail"]["purged"] == ["file.wav"]


def test_start_registration_failure_leaves_no_unregistered_wav(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mic(monkeypatch)
    monkeypatch.setattr(SqliteStore, "save_media", _locked)
    result = start()
    assert result.exit_code != 0
    meeting = only_meeting(env.settings)
    assert wavs(env.settings, meeting.id) == []
    assert meeting.state is MeetingState.discarded  # its only recording is gone


def test_a_registered_row_is_retired_when_a_later_registration_fails(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mic(monkeypatch, FakeCapturer(("system", "mic")))
    original = SqliteStore.save_media
    calls = {"n": 0}

    def second_fails(self: Any, *args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SqliteStore, "save_media", second_fails)
    assert start().exit_code != 0
    meeting = only_meeting(env.settings)
    assert wavs(env.settings, meeting.id) == []
    assert len(store_of(env.settings).list_media(meeting.id)) == 1
    assert live_media(env.settings, meeting.id) == []


# --------------------------------------------------------------------------- own_sources


class _NamedCapturer(FakeCapturer):
    """A capture helper that names its file ``mic-48k.wav``, not ``mic.wav``."""

    def start(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / "mic-48k.wav"
        shutil.copyfile(self.source, dst)
        os.chmod(dst, 0o600)
        self._paths.append(dst)
        self.events.append("start")

    def stop(self) -> list[Any]:
        self.events.append("stop")
        return [Track(name="mic", path=p, sample_rate=48000) for p in self._paths if p.exists()]


def test_failed_capture_purges_a_handed_over_source_under_another_name(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mic(monkeypatch, _NamedCapturer(("mic",)))
    use_engine(monkeypatch, _Engine(error=PraktikaError("STT server returned 503")))
    assert start().exit_code == 1
    assert wavs(env.settings, only_meeting(env.settings).id) == []


def test_successful_capture_keeps_only_the_registered_copy(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mic(monkeypatch, _NamedCapturer(("mic",)))
    result = start()
    assert result.exit_code == 0, result.output
    meeting = only_meeting(env.settings)
    assert wavs(env.settings, meeting.id) == ["mic.wav"]
    (media,) = live_media(env.settings, meeting.id)
    assert media.path.name == "mic.wav"
    (event,) = audit(env.settings, "ingest.file")
    assert event["object"] == "mic-48k.wav" and event["detail"]["source_purged"] is True


@pytest.mark.parametrize("command", ["start", "ingest"])
def test_zero_retention_leaves_no_audio_and_audits_the_purge(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    env.settings = env.settings.model_copy(
        update={"retention_audio_hours": {"internal": 0, "confidential": 0, "restricted": 0}}
    )
    fake_mic(monkeypatch, FakeCapturer(("system", "mic")))
    args = ["ingest", str(TONE_WAV), *GATE] if command == "ingest" else ["start", "--title", "R"]
    extra = [] if command == "ingest" else ["--duration", "0.1", *GATE]
    result = invoke(*args, *extra)
    assert result.exit_code == 0, result.output
    meeting = only_meeting(env.settings)
    assert wavs(env.settings, meeting.id) == []
    assert store_of(env.settings).list_media(meeting.id) == []
    deleted = audit(env.settings, "retention.deleted")
    assert len(deleted) == (1 if command == "ingest" else 2)
    assert all(e["detail"]["kind"] == "audio" and e["detail"]["file_removed"] for e in deleted)


# --------------------------------------------------------------------------- an abort wins


def _abort(meeting_id: str) -> None:
    meeting_ops.abort(meeting_id)  # what a second shell runs


def test_abort_during_ingest_transcription_wins(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch, _Engine(hook=lambda wav: _abort(wav.parent.name)))
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "stopped" in result.output
    meeting = only_meeting(env.settings)
    store = store_of(env.settings)
    assert meeting.state is MeetingState.discarded
    assert store.latest_minutes(meeting.id) is None
    assert store.get_transcript(meeting.id) is None
    assert live_media(env.settings, meeting.id) == []
    assert wavs(env.settings, meeting.id) == []


def test_abort_during_transcribe_rerun_stores_no_transcript(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingested(env)
    before = store_of(env.settings).get_transcript(meeting.id)
    use_engine(monkeypatch, _Engine(text="Something else.", hook=lambda wav: _abort(meeting.id)))
    result = invoke("transcribe", meeting.id, "--no-diarize")
    assert result.exit_code == 1 and "Transcribed" not in result.output
    assert state_of(env.settings, meeting.id) is MeetingState.discarded
    assert store_of(env.settings).get_transcript(meeting.id) == before


def test_abort_just_after_the_transcript_is_stored_removes_it_and_restores_the_vault(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingested(env)
    store = store_of(env.settings)
    before, vault = store.get_transcript(meeting.id), store.get_vault(meeting.id)
    original = SqliteStore.save_transcript

    def then_abort(self: Any, transcript: Any, **kwargs: Any) -> int:
        row = original(self, transcript, **kwargs)
        _abort(meeting.id)
        return row

    monkeypatch.setattr(SqliteStore, "save_transcript", then_abort)
    use_engine(monkeypatch, _Engine(text="Something else entirely."))
    result = invoke("transcribe", meeting.id, "--no-diarize")
    assert result.exit_code == 1 and "removed what it had stored" in result.output
    store = store_of(env.settings)
    assert store.get_meeting(meeting.id).state is MeetingState.discarded  # type: ignore[union-attr]
    assert store.get_transcript(meeting.id) == before
    assert store.get_vault(meeting.id) == vault
    failed = audit(env.settings, "ingest.failed")[-1]
    assert failed["detail"] == {
        "stage": "redact",
        "error": "RunStoppedError",
        "purged": [],
        "stopped_by": "abort",
    }


class _AbortingLLM(FakeLLM):
    """Aborts the meeting from 'another shell' on its first model call."""

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings, self.done = settings, False

    def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not self.done:
            self.done = True
            (m,) = store_of(self.settings).list_meetings()
            assert m.state is MeetingState.drafting
            _abort(m.id)
        return super().complete_json(*args, **kwargs)


def test_abort_during_drafting_wins_and_removes_the_run(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _AbortingLLM(env.settings)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: llm)
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "stopped" in result.output
    meeting = only_meeting(env.settings)
    store = store_of(env.settings)
    assert meeting.state is MeetingState.discarded
    assert store.latest_minutes(meeting.id) is None
    assert store.get_transcript(meeting.id) is None and store.get_vault(meeting.id) is None
    assert live_media(env.settings, meeting.id) == [] and wavs(env.settings, meeting.id) == []
    failed = audit(env.settings, "ingest.failed")[-1]
    assert failed["detail"]["stage"] == "draft" and failed["detail"]["error"] == "RunStoppedError"


def test_abort_during_generate_removes_only_the_new_version(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingested(env)
    llm = _AbortingLLM(env.settings)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: llm)
    assert invoke("generate", meeting.id).exit_code == 1
    store = store_of(env.settings)
    assert store.get_meeting(meeting.id).state is MeetingState.discarded  # type: ignore[union-attr]
    latest = store.latest_minutes(meeting.id)
    assert latest is not None and latest.version == 1


def test_abort_leaves_an_approved_meeting_approved(env: _Holder) -> None:
    meeting = ingested(env)
    assert invoke("approve", meeting.id).exit_code == 0
    refused = invoke("abort", meeting.id)
    assert refused.exit_code == 1 and "nothing was changed" in refused.output
    assert state_of(env.settings, meeting.id) is MeetingState.approved


def test_abort_removes_the_meeting_a_failed_ingest_left_at_created(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch, _Engine(error=PraktikaError("STT request failed")))
    assert invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE).exit_code == 1
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.created
    result = invoke("abort", meeting.id)
    assert result.exit_code == 0, result.output
    assert state_of(env.settings, meeting.id) is MeetingState.discarded


# --------------------------------------------------------------------------- start: capture checks


class _NoPortAudio(importlib.abc.MetaPathFinder):
    """Importing ``sounddevice`` fails as it does on a host without ``libportaudio2``."""

    def find_spec(self, name: str, path: Any, target: Any = None) -> None:
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return None


def test_start_mic_without_portaudio_is_refused_before_the_consent_gate(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NoPortAudio(), *sys.meta_path])
    result = start()
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), "a clean error, not a traceback"
    assert "cannot record from a microphone" in result.output
    assert "PortAudio library not found" in result.output
    assert consent.SCRIPT_EN[:40] not in result.output
    assert not (env.settings.data_dir / "praktika.db").exists(), "no meeting, no consent record"


def test_mic_preflight_is_skipped_with_an_injected_stream() -> None:
    MicCapturer(stream_factory=lambda callback: None).preflight()


class _BrokenCapturer(FakeCapturer):
    """Writes a WAV header, then the device fails."""

    def start(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mic.wav").write_bytes(b"RIFF" + b"\0" * 40)
        raise OSError("Error opening InputStream: Device unavailable")


def test_a_capture_that_cannot_start_is_discarded_cleanly(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mic(monkeypatch, _BrokenCapturer(("mic",)))
    result = start()
    assert result.exit_code == 1 and "cannot start the capture" in result.output
    assert isinstance(result.exception, SystemExit), "a clean error, not a traceback"
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.discarded
    audio = env.settings.data_dir / "audio" / meeting.id
    assert wavs(env.settings, meeting.id) == [] and not (audio / meetings.PID_FILE).exists()
    (failed,) = audit(env.settings, "ingest.failed")
    assert failed["detail"] == {"stage": "capture", "error": "OSError", "purged": ["mic.wav"]}


def test_default_capture_helper_follows_the_platform_data_directory() -> None:
    assert meetings.DEFAULT_HELPER == default_data_dir() / "bin" / "praktika-capture"
    linux = default_data_dir("linux", {"XDG_DATA_HOME": "/srv/xdg"})
    assert "Library" not in str(linux) and linux == Path("/srv/xdg/praktika")


def test_steps_stale_reason_is_none_for_a_fresh_draft(env: _Holder) -> None:
    meeting = ingested(env)
    store = store_of(env.settings)
    minutes = store.latest_minutes(meeting.id)
    assert minutes is not None
    assert steps.stale_draft_reason(store, meeting, minutes) is None
