"""Per-meeting run locks (R1) and a failed first ingest that is audited and retryable (R2).

Regression tests: an abort mid-transcription is a clean stop, not a read error; ``abort`` refuses
an unknown or approved meeting and one under legal hold; review-page writes are compare-and-set and
refused while a run works on the meeting; a second run on a meeting is refused and a run only ever
removes what it stored itself, and a draft whose transcript is missing is not approvable; the vault
and the transcript are written together; an abort during ``generate --reopen`` keeps the approved
record and its search row.

Two groups: reproductions run in one process with a hook standing in for the second shell or the
review page, and tests with two processes on one database: a child process runs a command that
pauses at a known point (the model call, or the speech-server call) while this process acts as the
second shell or the review page. Everything is offline and synthetic (conftest fakes, the tone
fixture).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf
from conftest import REPO, TONE_WAV, FakeDetector, FakeIdentity, FakeLLM, FakeTranscriber
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from praktika import server
from praktika.audio.convert import AudioInfo
from praktika.audit import AuditLog, JsonlAuditSink
from praktika.cli import app, meeting_ops
from praktika.cli import context as ctx
from praktika.config import Settings
from praktika.drafts import stale_draft_reason
from praktika.ingest import audio_file
from praktika.models import Meeting, MeetingState, RawSegment, SpeechChunk
from praktika.redact.tokenise import decrypt_vault, tokens_in
from praktika.store import db
from praktika.store.locks import RunLock, RunLockedError, holder_alive, new_run_lock, this_host
from praktika.store.repo import SqliteStore
from praktika.stt import router

runner = CliRunner()
TESTS = Path(__file__).resolve().parent
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
OMAR = "Please email omar.test@acme.test the pack by Thursday."
LAYLA = "Please email layla.test@acme.test the budget by Friday."
KEY = base64.urlsafe_b64encode(bytes(range(1, 33)))  # the conftest ``fixed_key``


# --------------------------------------------------------------------------- fakes and helpers


def _segment(text: str) -> RawSegment:
    return RawSegment(start=0.0, end=2.9, text=text, language="en", confidence=0.9, engine="fake")


def _wait_for(go: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not go.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"{go} never appeared")
        time.sleep(0.02)


class _Engine:
    """An English engine that runs ``hook(wav)`` first (a second shell acting meanwhile), or
    pauses (``ready`` written, then waits for ``go``); with ``read_wav`` it then reads the WAV,
    as the speech-server client does for every chunk."""

    name = "fake-en"
    auto_language = True

    def __init__(
        self,
        text: str = LINE,
        hook: Any = None,
        *,
        pause: tuple[Path, Path] | None = None,
        read_wav: bool = False,
        silent: bool = False,
    ) -> None:
        self.text, self.hook, self.pause = text, hook, pause
        self.read_wav, self.silent = read_wav, silent

    def transcribe(self, wav: Path, chunks: list[SpeechChunk], language: str | None) -> list:
        if self.hook is not None:
            self.hook(wav)
        if self.pause is not None:
            ready, go = self.pause
            ready.write_text(str(os.getpid()), encoding="utf-8")
            _wait_for(go)
        if self.read_wav:
            sf.read(str(wav))  # raises once `praktika abort` has removed the file
        return [] if self.silent else [_segment(self.text)]

    def unload(self) -> None:
        return None


class _HookLLM(FakeLLM):
    """Runs ``hook(meeting_id)`` on its first model call, or pauses there like ``_Engine``."""

    def __init__(
        self, settings: Any = None, hook: Any = None, pause: tuple[Path, Path] | None = None
    ) -> None:
        super().__init__()
        self.settings, self.hook, self.pause = settings, hook, pause
        self.done, self.out = False, None

    def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not self.done:
            self.done = True
            if self.hook is not None:
                (meeting,) = SqliteStore(self.settings.data_dir / "praktika.db").list_meetings()
                self.out = self.hook(meeting.id)
            if self.pause is not None:
                ready, go = self.pause
                ready.write_text(str(os.getpid()), encoding="utf-8")
                _wait_for(go)
        return super().complete_json(*args, **kwargs)


def _copy_convert(src: Path, dst: Path, *, ffmpeg: str | None = None) -> AudioInfo:
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o600)
    return AudioInfo(path=dst, sample_rate=16000, channels=1, duration_s=3.0, sha256="a" * 64)


def _one_chunk(wav: Path, **kw: Any) -> list[SpeechChunk]:
    return [SpeechChunk(index=0, track=kw.get("track", "file"), start=0, end=2.9)]


def use_engine(monkeypatch: pytest.MonkeyPatch, engine: Any) -> None:
    monkeypatch.setattr(
        router,
        "build_transcribers",
        lambda settings, **kw: (engine, FakeTranscriber([]), FakeDetector("en")),
    )


class _Holder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings


@pytest.fixture
def env(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch, fixed_key: bytes) -> _Holder:
    """CLI collaborators, ffmpeg-free conversion, one VAD chunk and fake engines."""
    holder = _Holder(tmp_settings)
    monkeypatch.setattr(ctx, "load_settings", lambda: holder.settings)
    monkeypatch.setattr(ctx, "vault_key", lambda settings=None: fixed_key)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: FakeLLM())
    monkeypatch.setattr(audio_file, "to_wav16k", _copy_convert)
    monkeypatch.setattr(router, "speech_chunks", _one_chunk)
    use_engine(monkeypatch, _Engine())
    return holder


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


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
    if not path.exists():
        return []
    rows = [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln]
    return [r for r in rows if event is None or r["event"] == event]


def locks(settings: Settings) -> list[Any]:
    return store_of(settings).conn.execute("SELECT * FROM run_locks").fetchall()


def ingest_with(env: _Holder, monkeypatch: pytest.MonkeyPatch, text: str = OMAR) -> Meeting:
    use_engine(monkeypatch, _Engine(text=text))
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 0, result.output
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.draft_ready
    return meeting


def review_client(settings: Settings, meeting: Meeting, llm: Any = None) -> TestClient:
    """The review page as a reviewer's browser uses it, over its own store connection."""
    store = SqliteStore(settings.data_dir / "praktika.db")
    audit_log = AuditLog(JsonlAuditSink(settings.data_dir / "audit.jsonl"), store, None)
    application = server.create_app(
        settings,
        store,
        FakeIdentity(user=meeting.organiser, source="session"),
        audit_log,
        llm_client=llm if llm is not None else FakeLLM(),
    )
    return TestClient(application, base_url="http://127.0.0.1", headers={"X-Praktika-Review": "1"})


def vault_values(settings: Settings, meeting_id: str) -> list[str]:
    blob = store_of(settings).get_vault(meeting_id)
    assert blob is not None
    return sorted(decrypt_vault(blob, KEY).entries.values())


def second_shell(command: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a command's body as another shell would, returning the error it refused with."""
    try:
        command.__wrapped__(*args, **kwargs)
    except RunLockedError as exc:
        return exc
    return None


# --------------------------------------------------------------------------- a second process

CHILD = (
    "import sys; sys.path[:0] = sys.argv[2:4]; "
    "import test_run_lock; test_run_lock.child_main(sys.argv[1])"
)


def child_main(spec_path: str) -> None:  # pragma: no cover - runs in the child process
    """Entry point of the child process: the same fakes as ``env``, then one CLI command."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    settings = Settings(
        _env_file=None,
        data_dir=Path(spec["data_dir"]),
        models_dir=Path(spec["models_dir"]),
        prompts_dir=REPO / "prompts",
        glossary_path=REPO / "glossary.yaml",
        allowed_hosts=["localhost", "127.0.0.1"],
        llm_provider="fake",
        stt_en="fake",
        stt_ar="fake",
        diarize_backend="fake",
        identity_provider="fake",
        audit_sink="jsonl",
        pilot_smoke=True,
    )
    pause = (Path(spec["ready"]), Path(spec["go"]))
    llm = _HookLLM(pause=pause if spec["pause"] == "llm" else None)
    engine = _Engine(
        spec["text"], pause=pause if spec["pause"] == "stt" else None, read_wav=spec["read_wav"]
    )
    ctx.load_settings = lambda: settings
    ctx.checked_env_file = lambda: None  # type: ignore[assignment,return-value]
    ctx.vault_key = lambda settings=None: KEY
    ctx.llm_client = lambda s: llm
    audio_file.to_wav16k = _copy_convert  # type: ignore[assignment]
    router.speech_chunks = _one_chunk  # type: ignore[assignment]
    router.build_transcribers = lambda s, **kw: (engine, FakeTranscriber([]), FakeDetector("en"))
    app(args=spec["argv"], prog_name="praktika")


class Child:
    """A second process running one CLI command on the same database; it pauses at the model
    call (``pause="llm"``) or the speech-server call (``pause="stt"``) until released."""

    def __init__(
        self,
        tmp_path: Path,
        settings: Settings,
        argv: list[str],
        *,
        pause: str,
        text: str = LINE,
        read_wav: bool = False,
    ) -> None:
        work = tmp_path / f"child-{uuid.uuid4().hex[:8]}"
        work.mkdir()
        self.ready, self.go = work / "ready", work / "go"
        spec = {
            "data_dir": str(settings.data_dir),
            "models_dir": str(settings.models_dir),
            "argv": argv,
            "pause": pause,
            "text": text,
            "read_wav": read_wav,
            "ready": str(self.ready),
            "go": str(self.go),
        }
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        environ = {k: v for k, v in os.environ.items() if not k.startswith("PRAKTIKA_")}
        self.proc = subprocess.Popen(  # noqa: S603 - this interpreter, fixed arguments
            [sys.executable, "-c", CHILD, str(spec_path), str(TESTS), str(REPO / "src")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environ,
            cwd=str(tmp_path),
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    def wait_paused(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while not self.ready.exists():
            if self.proc.poll() is not None:
                out, err = self.proc.communicate()
                raise AssertionError(f"the child ended before pausing:\n{out}\n{err}")
            if time.monotonic() > deadline:
                self.kill()
                raise AssertionError("the child never paused")
            time.sleep(0.02)

    def release(self) -> None:
        self.go.write_text("go", encoding="utf-8")

    def finish(self, timeout: float = 60.0) -> tuple[int, str]:
        out, err = self.proc.communicate(timeout=timeout)
        return self.proc.returncode, out + err

    def kill(self) -> None:
        self.proc.kill()
        self.proc.communicate(timeout=30)


@pytest.fixture
def children() -> Any:
    """Every child is killed at the end of the test, whatever happened."""
    started: list[Child] = []
    yield started
    for child in started:
        if child.proc.poll() is None:
            child.kill()


def spawn(children: list[Child], *args: Any, **kwargs: Any) -> Child:
    child = Child(*args, **kwargs)
    children.append(child)
    child.wait_paused()
    return child


def test_generate_from_another_process_while_ingest_drafts_is_refused(
    env: _Holder, tmp_path: Path, children: list[Child]
) -> None:
    child = spawn(
        children,
        tmp_path,
        env.settings,
        ["ingest", str(TONE_WAV), "--lang", "en", *GATE],
        pause="llm",
        text=OMAR,
    )
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.drafting
    lock = store_of(env.settings).run_lock(meeting.id)
    assert lock is not None and lock.command == "ingest" and lock.pid == child.pid
    second = invoke("generate", meeting.id)
    assert second.exit_code == 1
    assert "an ingest run is working on this meeting" in second.output
    assert f"process {child.pid} on {this_host()}" in second.output
    child.release()
    code, out = child.finish()
    assert code == 0, out
    store = store_of(env.settings)
    assert state_of(env.settings, meeting.id) is MeetingState.draft_ready
    latest = store.latest_minutes(meeting.id)
    assert latest is not None and latest.version == 1
    assert store.get_transcript(meeting.id) is not None and store.get_vault(meeting.id)
    assert len(live_media(env.settings, meeting.id)) == 1
    assert locks(env.settings) == [] and audit(env.settings, "ingest.failed") == []
    assert invoke("approve", meeting.id).exit_code == 0


def test_generate_from_another_process_while_transcribe_runs_is_refused(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, children: list[Child]
) -> None:
    meeting = ingest_with(env, monkeypatch)
    child = spawn(
        children,
        tmp_path,
        env.settings,
        ["transcribe", meeting.id, "--no-diarize"],
        pause="stt",
        text=LAYLA,
    )
    assert state_of(env.settings, meeting.id) is MeetingState.transcribing
    second = invoke("generate", meeting.id)
    assert second.exit_code == 1 and "a transcribe run is working on this meeting" in second.output
    assert invoke("approve", meeting.id).exit_code == 1
    child.release()
    code, out = child.finish()
    assert code == 0 and "Transcribed" in out, out
    store = store_of(env.settings)
    assert store.latest_minutes(meeting.id).version == 1  # type: ignore[union-attr]
    assert vault_values(env.settings, meeting.id) == ["layla.test@acme.test"]
    assert state_of(env.settings, meeting.id) is MeetingState.transcribing  # awaits generate
    assert invoke("generate", meeting.id).exit_code == 0
    assert invoke("approve", meeting.id).exit_code == 0


def test_review_page_writes_during_generate_in_another_process_get_409(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, children: list[Child]
) -> None:
    meeting = ingest_with(env, monkeypatch)
    v1 = store_of(env.settings).latest_minutes(meeting.id)
    assert v1 is not None
    child = spawn(children, tmp_path, env.settings, ["generate", meeting.id], pause="llm")
    client = review_client(env.settings, meeting)
    item = (v1.actions or v1.decisions)[0]
    mid = meeting.id
    refused = [
        client.post(f"/api/minutes/{mid}/items/{item.id}", json={"action": "reject",
                    "reason_code": "not_said"}),
        client.post(f"/api/meetings/{mid}/speakers", json={"SPEAKER_00": "Omar"}),
        client.post(f"/api/minutes/{mid}/flags/0/clear"),
        client.post(f"/api/minutes/{mid}/regenerate", json={"section": "summary",
                    "instruction": "shorter please"}),
        client.post(f"/api/minutes/{mid}/approve", json={"reason_code": "accurate"}),
    ]  # fmt: skip
    for r in refused:
        assert r.status_code == 409 and r.json()["detail"] == (
            "a generate run is working on this meeting"
        ), r.text
    run = client.get(f"/api/meetings/{mid}/run").json()
    assert run["running"] and run["command"] == "generate"
    page = client.get(f"/api/meetings/{mid}")
    assert page.status_code == 200 and page.json()["meeting"]["state"] == "drafting"
    child.release()
    code, out = child.finish()
    assert code == 0, out
    store = store_of(env.settings)
    assert store.latest_minutes(mid).version == 2  # type: ignore[union-attr]
    assert store.get_minutes(mid, 1).review.items == []  # type: ignore[union-attr]
    assert state_of(env.settings, mid) is MeetingState.draft_ready
    assert audit(env.settings, "ingest.failed") == [] and audit(env.settings, "review.item") == []
    assert client.get(f"/api/meetings/{mid}/run").json()["running"] is False


def test_review_page_regenerate_during_transcribe_in_another_process_gets_409(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, children: list[Child]
) -> None:
    meeting = ingest_with(env, monkeypatch)
    child = spawn(
        children,
        tmp_path,
        env.settings,
        ["transcribe", meeting.id, "--no-diarize"],
        pause="stt",
        text=LAYLA,
    )
    client = review_client(env.settings, meeting)
    r = client.post(
        f"/api/minutes/{meeting.id}/regenerate",
        json={"section": "summary", "instruction": "shorter please"},
    )
    assert r.status_code == 409 and r.json()["detail"] == (
        "a transcribe run is working on this meeting"
    )
    child.release()
    code, out = child.finish()
    assert code == 0, out
    assert store_of(env.settings).latest_minutes(meeting.id).version == 1  # type: ignore[union-attr]


@pytest.mark.parametrize("command", ["ingest", "transcribe"])
def test_abort_from_another_process_mid_transcription_is_a_clean_stop(
    env: _Holder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    children: list[Child],
    command: str,
) -> None:
    """The speech-server client reads the WAV for every chunk: once `praktika abort` has
    removed it, the read fails. That is reported as the stop it is, with no traceback."""
    if command == "transcribe":
        argv = ["transcribe", ingest_with(env, monkeypatch).id, "--no-diarize"]
    else:
        argv = ["ingest", str(TONE_WAV), "--lang", "en", *GATE]
    child = spawn(children, tmp_path, env.settings, argv, pause="stt", text=LAYLA, read_wav=True)
    meeting = only_meeting(env.settings)
    aborted = invoke("abort", meeting.id)
    assert aborted.exit_code == 0, aborted.output
    child.release()
    code, out = child.finish()
    assert code == 1, out
    assert "Traceback" not in out and "LibsndfileError" not in out, out
    assert (
        f"{meeting.id} was discarded by `praktika abort` while this {command} run was "
        "working on it, so the run stopped"
    ) in out
    failed = audit(env.settings, "ingest.failed")[-1]["detail"]
    assert failed["error"] == "RunStoppedError" and failed["stopped_by"] == "abort"
    assert failed["stage"] == "transcribe"
    assert state_of(env.settings, meeting.id) is MeetingState.discarded
    assert wavs(env.settings, meeting.id) == [] and live_media(env.settings, meeting.id) == []
    assert locks(env.settings) == []


def test_a_dead_runs_lock_is_taken_over_and_audited(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, children: list[Child]
) -> None:
    meeting = ingest_with(env, monkeypatch)
    child = spawn(children, tmp_path, env.settings, ["generate", meeting.id], pause="llm")
    dead = child.pid
    child.kill()  # a crash, or kill -9: the lock is never released
    lock = store_of(env.settings).run_lock(meeting.id)
    assert lock is not None and lock.pid == dead and not holder_alive(lock)
    assert state_of(env.settings, meeting.id) is MeetingState.drafting
    client = review_client(env.settings, meeting)
    assert client.get(f"/api/meetings/{meeting.id}/run").json()["running"] is False
    result = invoke("generate", meeting.id)
    assert result.exit_code == 0, result.output
    (event,) = audit(env.settings, "run.lock_taken_over")
    assert event["meeting_id"] == meeting.id and event["classification"] == "internal"
    assert event["detail"]["command"] == "generate"
    assert event["detail"]["previous_command"] == "generate"
    assert event["detail"]["previous_pid"] == dead
    assert event["detail"]["previous_host"] == this_host()
    assert state_of(env.settings, meeting.id) is MeetingState.draft_ready
    assert locks(env.settings) == []


# --------------------------------------------------------------------------- one process


@pytest.mark.parametrize("edit", ["reject_item", "map_speakers", "clear_flag", "regenerate"])
def test_a_review_page_edit_during_cli_generate_is_refused_and_the_run_completes(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, edit: str
) -> None:
    meeting = ingest_with(env, monkeypatch)
    v1 = store_of(env.settings).latest_minutes(meeting.id)
    assert v1 is not None
    client = review_client(env.settings, meeting)

    def do_edit(mid: str) -> Any:
        if edit == "reject_item":
            item = (v1.actions or v1.decisions)[0]
            body = {"action": "reject", "reason_code": "not_said"}
            return client.post(f"/api/minutes/{mid}/items/{item.id}", json=body)
        if edit == "map_speakers":
            return client.post(f"/api/meetings/{mid}/speakers", json={"SPEAKER_00": "Omar"})
        if edit == "regenerate":
            body = {"section": "summary", "instruction": "shorter please"}
            return client.post(f"/api/minutes/{mid}/regenerate", json=body)
        return client.post(f"/api/minutes/{mid}/flags/0/clear")

    llm = _HookLLM(env.settings, do_edit)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: llm)
    result = invoke("generate", meeting.id)
    assert llm.out.status_code == 409, llm.out.text
    assert llm.out.json()["detail"] == "a generate run is working on this meeting"
    assert result.exit_code == 0, result.output
    assert store_of(env.settings).latest_minutes(meeting.id).version == 2  # type: ignore[union-attr]
    assert audit(env.settings, "ingest.failed") == []


def test_generate_while_transcribe_runs_is_refused(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    seen: dict[str, Any] = {}

    def meanwhile(_wav: Path) -> None:
        seen["refused"] = second_shell(meeting_ops.generate, meeting.id)

    use_engine(monkeypatch, _Engine(text=LAYLA, hook=meanwhile))
    result = invoke("transcribe", meeting.id, "--no-diarize")
    assert result.exit_code == 0, result.output
    assert isinstance(seen["refused"], RunLockedError)
    assert seen["refused"].lock.command == "transcribe"
    latest = store_of(env.settings).get_transcript(meeting.id)
    assert latest is not None and "«EMAIL_1»" in latest.segments[0].text
    assert vault_values(env.settings, meeting.id) == ["layla.test@acme.test"]


def test_generate_during_ingest_drafting_is_refused_and_the_ingest_is_usable(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_engine(monkeypatch, _Engine(text=OMAR))
    llm = _HookLLM(env.settings, lambda mid: second_shell(meeting_ops.generate, mid))
    monkeypatch.setattr(ctx, "llm_client", lambda settings: llm)
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 0, result.output
    assert isinstance(llm.out, RunLockedError) and llm.out.lock.command == "ingest"
    meeting = only_meeting(env.settings)
    store = store_of(env.settings)
    assert store.get_transcript(meeting.id) is not None
    assert vault_values(env.settings, meeting.id) == ["omar.test@acme.test"]
    assert len(live_media(env.settings, meeting.id)) == 1
    assert invoke("approve", meeting.id).exit_code == 0
    out = tmp_path / "export.md"
    assert invoke("export", meeting.id, "--detokenise", "--out", str(out)).exit_code == 0


def test_a_review_page_discard_stops_a_cli_run_and_is_named(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    client = review_client(env.settings, meeting)
    llm = _HookLLM(
        env.settings,
        lambda mid: client.post(f"/api/minutes/{mid}/discard", json={"reason_code": "other"}),
    )
    monkeypatch.setattr(ctx, "llm_client", lambda settings: llm)
    result = invoke("generate", meeting.id)
    assert llm.out.status_code == 200, llm.out.text
    assert result.exit_code == 1
    assert (
        f"{meeting.id} was discarded on the review page while this generate run was working on it"
    ) in result.output
    assert "removed what it had stored: draft v2" in result.output
    failed = audit(env.settings, "ingest.failed")[-1]["detail"]
    assert failed == {
        "stage": "draft",
        "error": "RunStoppedError",
        "purged": [],
        "stopped_by": "discard",
    }
    store = store_of(env.settings)
    assert state_of(env.settings, meeting.id) is MeetingState.discarded
    assert store.latest_minutes(meeting.id).version == 1  # type: ignore[union-attr]


def test_abort_mid_transcription_in_one_process_is_a_clean_stop(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def abort_then_read(wav: Path) -> None:
        second_shell(meeting_ops.abort, wav.parent.name)

    use_engine(monkeypatch, _Engine(hook=abort_then_read, read_wav=True))
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.output
    assert "was discarded by `praktika abort` while this ingest run" in result.output
    (failed,) = audit(env.settings, "ingest.failed")
    assert failed["detail"]["error"] == "RunStoppedError"
    assert failed["detail"]["stopped_by"] == "abort"


def test_a_failed_rerun_leaves_vault_and_transcript_together(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The transcript insert fails after the vault was replaced in the same transaction (a
    locked database, a full disk): both roll back, and the approved minutes still
    de-tokenise to the identifiers of their own transcript."""
    meeting = ingest_with(env, monkeypatch, OMAR)
    before = store_of(env.settings).get_transcript(meeting.id)

    def locked(self: Any, cols: Any) -> int:
        raise sqlite3.OperationalError("database is locked")

    insert = SqliteStore._insert_transcript_row
    monkeypatch.setattr(SqliteStore, "_insert_transcript_row", locked)
    use_engine(monkeypatch, _Engine(text=LAYLA))
    assert invoke("transcribe", meeting.id, "--no-diarize").exit_code == 1
    monkeypatch.setattr(SqliteStore, "_insert_transcript_row", insert)
    assert vault_values(env.settings, meeting.id) == ["omar.test@acme.test"]
    assert store_of(env.settings).get_transcript(meeting.id) == before
    assert invoke("approve", meeting.id).exit_code == 0
    out = tmp_path / "export.md"
    assert invoke("export", meeting.id, "--detokenise", "--out", str(out)).exit_code == 0
    text = out.read_text("utf-8")
    assert "omar.test@acme.test" in text and "layla.test" not in text


REDACT = ["tokenise", "inside_the_vault_and_transcript_write", "after_save_transcript"]
DRAFT = ["llm", "after_save_minutes", "clear_index"]
#: Each command with the points it passes: ingest all of them, transcribe the redaction ones,
#: generate the drafting ones.
INTERRUPTIONS = [
    *(("ingest", p) for p in [*REDACT, "audit_redact", *DRAFT]),
    *(("transcribe", p) for p in [*REDACT, "audit_redact"]),
    *(("generate", p) for p in DRAFT),
]
FIRED = {"n": 0}


def _interrupt_at(monkeypatch: pytest.MonkeyPatch, point: str) -> None:
    from praktika.redact import tokenise

    def once(fn: Any, after: bool) -> Any:
        state = {"done": False}

        def wrapper(*a: Any, **kw: Any) -> Any:
            if state["done"]:
                return fn(*a, **kw)
            state["done"] = True
            if after:
                fn(*a, **kw)
            FIRED["n"] += 1
            raise KeyboardInterrupt

        return wrapper

    if point == "tokenise":
        monkeypatch.setattr(tokenise.Tokeniser, "apply", once(tokenise.Tokeniser.apply, False))
    elif point == "inside_the_vault_and_transcript_write":
        monkeypatch.setattr(
            SqliteStore, "_insert_transcript_row", once(SqliteStore._insert_transcript_row, True)
        )
    elif point == "after_save_transcript":
        monkeypatch.setattr(SqliteStore, "save_transcript", once(SqliteStore.save_transcript, True))
    elif point == "audit_redact":
        original = AuditLog.append

        def append(self: Any, event: str, *a: Any, **kw: Any) -> Any:
            if event == "redact.applied":
                FIRED["n"] += 1
                raise KeyboardInterrupt
            return original(self, event, *a, **kw)

        monkeypatch.setattr(AuditLog, "append", append)
    elif point == "llm":
        monkeypatch.setattr(FakeLLM, "complete_json", once(FakeLLM.complete_json, False))
    elif point == "after_save_minutes":
        monkeypatch.setattr(SqliteStore, "save_minutes", once(SqliteStore.save_minutes, True))
    elif point == "clear_index":
        monkeypatch.setattr(SqliteStore, "clear_index", once(SqliteStore.clear_index, False))


def _violations(env: _Holder, meeting_id: str, before: dict[str, Any]) -> list[str]:
    store = store_of(env.settings)
    meeting = store.get_meeting(meeting_id)
    assert meeting is not None
    out = []
    if meeting.state in (MeetingState.capturing, MeetingState.drafting):
        out.append(f"left at {meeting.state.value}")
    if meeting.state is MeetingState.transcribing and before.get("state") is not (
        MeetingState.transcribing
    ):
        out.append("left at transcribing after an interrupted run")
    live = {Path(x.path).name for x in live_media(env.settings, meeting_id) if x.delete_after}
    out.extend(f"unregistered WAV {w}" for w in wavs(env.settings, meeting_id) if w not in live)
    transcript, blob = store.get_transcript(meeting_id), store.get_vault(meeting_id)
    if transcript is not None and before.get("t_sha") == transcript.sha256():
        if blob != before.get("vault"):
            out.append("vault replaced although the latest transcript is unchanged")
    if transcript is not None and blob is None:
        out.append("transcript without a vault")
    if transcript is not None and blob is not None:
        known = set(decrypt_vault(blob, KEY).entries)
        if any(t not in known for s in transcript.segments for t in tokens_in(s.text)):
            out.append("the vault does not restore the latest transcript")
    minutes = store.latest_minutes(meeting_id)
    if minutes is not None and stale_draft_reason(store, meeting, minutes) is None:
        source = store.transcript_with_sha256(meeting_id, minutes.provenance.transcript_sha256)
        if source is None or transcript is None or source.segments != transcript.segments:
            out.append("approvable draft whose source transcript differs from the latest")
    if locks(env.settings):
        out.append("run lock left behind")
    return out


@pytest.mark.parametrize(("command", "point"), INTERRUPTIONS)
def test_ctrl_c_at_every_stage_leaves_a_consistent_meeting(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, command: str, point: str
) -> None:
    before: dict[str, Any] = {}
    FIRED["n"] = 0
    if command == "ingest":
        use_engine(monkeypatch, _Engine(text=OMAR))
        _interrupt_at(monkeypatch, point)
        try:
            invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
        except KeyboardInterrupt:
            pass
        meeting = only_meeting(env.settings)
    else:
        meeting = ingest_with(env, monkeypatch, OMAR)
        store = store_of(env.settings)
        transcript = store.get_transcript(meeting.id)
        assert transcript is not None
        before = {
            "state": state_of(env.settings, meeting.id),
            "t_sha": transcript.sha256(),
            "vault": store.get_vault(meeting.id),
        }
        use_engine(monkeypatch, _Engine(text=LAYLA))
        _interrupt_at(monkeypatch, point)
        try:
            invoke(command, meeting.id, *(["--no-diarize"] if command == "transcribe" else []))
        except KeyboardInterrupt:
            pass
    monkeypatch.undo()
    assert FIRED["n"] == 1, f"{command} never reached {point}"
    problems = _violations(env, meeting.id, before)
    assert not problems, f"{command} interrupted at {point}: {problems}"
    if command == "ingest":  # a failed first ingest is always audited, once
        assert len(audit(env.settings, "ingest.failed")) == 1


def test_abort_during_a_reopened_generate_keeps_the_approved_record(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    meeting = ingest_with(env, monkeypatch)
    assert invoke("approve", meeting.id).exit_code == 0
    llm = _HookLLM(env.settings, lambda mid: invoke("abort", mid))
    monkeypatch.setattr(ctx, "llm_client", lambda settings: llm)
    result = invoke("generate", meeting.id, "--reopen")
    assert llm.out.exit_code == 0 and "back to approved" in llm.out.output, llm.out.output
    assert result.exit_code == 1
    assert "was put back to approved by `praktika abort` while this generate run" in (result.output)
    store = store_of(env.settings)
    assert state_of(env.settings, meeting.id) is MeetingState.approved
    latest = store.latest_minutes(meeting.id)
    assert latest is not None and latest.version == 1 and latest.review.status == "approved"
    indexed = store.conn.execute(
        "SELECT COUNT(*) FROM minutes_fts WHERE meeting_id = ?", (meeting.id,)
    ).fetchone()[0]
    assert indexed == 1
    (aborted,) = audit(env.settings, "capture.aborted")
    assert aborted["detail"]["restored"] == "approved"
    failed = audit(env.settings, "ingest.failed")[-1]["detail"]
    assert failed["stage"] == "draft" and failed["stopped_by"] == "abort"
    out = tmp_path / "export.md"
    assert invoke("export", meeting.id, "--out", str(out)).exit_code == 0


def test_a_review_page_discard_never_discards_approved_minutes_being_redrafted(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    assert invoke("approve", meeting.id).exit_code == 0
    client = review_client(env.settings, meeting)
    llm = _HookLLM(
        env.settings,
        lambda mid: client.post(f"/api/minutes/{mid}/discard", json={"reason_code": "other"}),
    )
    monkeypatch.setattr(ctx, "llm_client", lambda settings: llm)
    result = invoke("generate", meeting.id, "--reopen")
    assert llm.out.status_code == 409, llm.out.text
    assert result.exit_code == 0, result.output
    store = store_of(env.settings)
    assert store.get_minutes(meeting.id, 1).review.status == "approved"  # type: ignore[union-attr]
    assert state_of(env.settings, meeting.id) is MeetingState.draft_ready


def test_abort_clears_the_search_row_of_a_meeting_it_discards(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    store = store_of(env.settings)
    store.index_minutes(meeting.id, 1, "t", "s", "reopened minutes body")
    assert invoke("abort", meeting.id).exit_code == 0
    rows = store.conn.execute(
        "SELECT COUNT(*) FROM minutes_fts WHERE meeting_id = ?", (meeting.id,)
    ).fetchone()[0]
    assert rows == 0 and state_of(env.settings, meeting.id) is MeetingState.discarded


@pytest.mark.parametrize("case", ["unknown", "approved", "purged", "legal_hold"])
def test_abort_refuses_and_changes_nothing(
    env: _Holder, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    if case == "unknown":
        mid = "M-20260925-ffff"
    else:
        meeting = ingest_with(env, monkeypatch)
        mid = meeting.id
        if case == "approved":
            assert invoke("approve", mid).exit_code == 0
        elif case == "purged":
            store_of(env.settings).set_state(mid, MeetingState.purged)
        else:
            assert invoke("hold", "set", mid, "--reason", "litigation").exit_code == 0
    assert invoke("retention", "run").exit_code == 0  # approval and purge retire the audio
    audio = wavs(env.settings, mid)
    assert bool(audio) is (case == "legal_hold")
    state = store_of(env.settings).get_meeting(mid)
    result = invoke("abort", mid)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.output
    assert "Aborted" not in result.output
    expected = {
        "unknown": "no meeting M-20260925-ffff",
        "approved": "abort never discards approved minutes, so nothing was changed",
        "purged": "already purged",
        "legal_hold": "under legal hold; abort would overwrite its retained audio",
    }[case]
    assert expected in result.output
    assert audit(env.settings, "capture.aborted") == []
    assert wavs(env.settings, mid) == audio
    assert store_of(env.settings).get_meeting(mid) == state


def test_review_regenerate_after_transcribe_does_not_make_the_old_draft_approvable(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    use_engine(monkeypatch, _Engine(text=LAYLA))
    assert invoke("transcribe", meeting.id, "--no-diarize").exit_code == 0
    client = review_client(env.settings, meeting)
    r = client.post(
        f"/api/minutes/{meeting.id}/regenerate",
        json={"section": "summary", "instruction": "shorter"},
    )
    assert r.status_code == 409 and "praktika generate" in r.json()["detail"]
    web = client.post(f"/api/minutes/{meeting.id}/approve", json={"reason_code": "reviewed"})
    assert web.status_code == 409 and invoke("approve", meeting.id).exit_code == 1


def test_speaker_mapping_edit_and_regenerate_do_not_block_approval(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    assert invoke("generate", meeting.id).exit_code == 0  # drafted from the stored transcript
    client = review_client(env.settings, meeting)
    mid = meeting.id
    assert client.post(f"/api/meetings/{mid}/speakers", json={"SPEAKER_00": "Omar"}).status_code
    body = {"section": "summary", "instruction": "shorter"}
    assert client.post(f"/api/minutes/{mid}/regenerate", json=body).status_code == 200
    latest = store_of(env.settings).latest_minutes(mid)
    assert latest is not None
    item = (latest.actions or latest.decisions)[0]
    body = {"action": "modify", "after": "Circulate the pack", "reason_code": "wording"}
    assert client.post(f"/api/minutes/{mid}/items/{item.id}", json=body).status_code == 200
    web = client.post(f"/api/minutes/{mid}/approve", json={"reason_code": "reviewed"})
    assert web.status_code == 200, web.text


# --------------------------------------------------------------------------- review page CAS


def test_a_review_page_regenerate_aborted_meanwhile_keeps_nothing(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug: an abort during a regenerate was lost, and the aborted meeting could then be
    approved and exported."""
    meeting = ingest_with(env, monkeypatch)
    llm = _HookLLM(env.settings, lambda mid: invoke("abort", mid))
    client = review_client(env.settings, meeting, llm=llm)
    r = client.post(
        f"/api/minutes/{meeting.id}/regenerate",
        json={"section": "summary", "instruction": "shorter please"},
    )
    assert llm.out.exit_code == 0, llm.out.output
    assert r.status_code == 409
    assert r.json()["detail"] == (
        "the meeting was discarded by `praktika abort` meanwhile; nothing was kept"
    )
    assert state_of(env.settings, meeting.id) is MeetingState.discarded
    assert store_of(env.settings).latest_minutes(meeting.id).version == 1  # type: ignore[union-attr]
    approve = client.post(f"/api/minutes/{meeting.id}/approve", json={"reason_code": "accurate"})
    assert approve.status_code == 409 and locks(env.settings) == []


def test_a_review_page_approve_is_compare_and_set(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    from praktika import server_review

    meeting = ingest_with(env, monkeypatch)
    client = review_client(env.settings, meeting)
    assert client.get(f"/api/meetings/{meeting.id}").json()["meeting"]["state"] == "in_review"
    checked = server_review.stale_draft_reason

    def aborted_meanwhile(store: Any, m: Meeting, minutes: Any) -> str | None:
        invoke("abort", m.id)  # between the request's read and its write
        return checked(store, m, minutes)

    monkeypatch.setattr(server_review, "stale_draft_reason", aborted_meanwhile)
    r = client.post(f"/api/minutes/{meeting.id}/approve", json={"reason_code": "accurate"})
    assert r.status_code == 409 and "now discarded" in r.json()["detail"]
    store = store_of(env.settings)
    assert store.latest_minutes(meeting.id).review.status != "approved"  # type: ignore[union-attr]
    assert audit(env.settings, "review.approved") == []


def test_approve_is_all_or_nothing(env: _Holder, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the review status cannot be written after the state moved to approved, the state is
    put back: never an approved meeting whose minutes are not marked approved."""
    from praktika.store.repo import SqliteStore

    meeting = ingest_with(env, monkeypatch)
    before = state_of(env.settings, meeting.id)

    def locked(self: Any, minutes: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SqliteStore, "set_review_status", locked)
    result = invoke("approve", meeting.id, "--reason", "accurate")
    assert result.exit_code != 0
    assert state_of(env.settings, meeting.id) is before
    assert audit(env.settings, "review.approved") == []


def test_a_meeting_left_mid_run_by_a_killed_process_says_how_to_recover(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No live lock but a running state (the run was killed): approve names the way out
    instead of asking the operator to wait for a run that no longer exists."""
    meeting = ingest_with(env, monkeypatch)
    store_of(env.settings).set_state(meeting.id, MeetingState.drafting)
    result = invoke("approve", meeting.id, "--reason", "accurate")
    assert result.exit_code != 0
    assert "no run is working on it" in result.output
    assert f"praktika generate {meeting.id}" in result.output


def test_opening_the_page_while_a_run_holds_the_lock_leaves_the_state(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    store = store_of(env.settings)
    lock = new_run_lock(meeting.id, "generate")
    assert store.take_run_lock(lock) is None
    client = review_client(env.settings, meeting)
    page = client.get(f"/api/meetings/{meeting.id}")
    assert page.status_code == 200 and page.json()["meeting"]["state"] == "draft_ready"
    assert state_of(env.settings, meeting.id) is MeetingState.draft_ready
    store.release_run_lock(meeting.id, lock.token)
    page = client.get(f"/api/meetings/{meeting.id}")
    assert page.json()["meeting"]["state"] == "in_review"


def test_the_page_pauses_its_edits_while_a_run_works() -> None:
    script = (REPO / "src" / "praktika" / "static" / "app.js").read_text("utf-8")
    assert "/run`" in script and "editing is paused until it finishes" in script
    assert "const frozen = closed || busy" in script


# --------------------------------------------------------------------------- approval checks


def test_a_draft_whose_transcript_is_gone_or_busy_is_not_approvable(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    store = store_of(env.settings)
    minutes = store.latest_minutes(meeting.id)
    source = store.get_transcript(meeting.id)
    assert minutes is not None and source is not None
    meeting = store.get_meeting(meeting.id)  # type: ignore[assignment]
    assert stale_draft_reason(store, meeting, minutes) is None
    lock = new_run_lock(meeting.id, "transcribe")
    store.take_run_lock(lock)
    assert stale_draft_reason(store, meeting, minutes) == (
        f"{meeting.id}: a transcribe run is working on this meeting; wait for it to finish, "
        "then review the draft it leaves"
    )
    store.release_run_lock(meeting.id, lock.token)
    other = source.model_copy(
        update={"segments": [source.segments[0].model_copy(update={"text": "Other words."})]}
    )
    store.save_transcript(other, delete_after=None)
    first = store.conn.execute("SELECT id FROM transcripts ORDER BY id LIMIT 1").fetchone()
    store.erase_transcript(first["id"], datetime.now(UTC))  # what a stopped run does
    reason = stale_draft_reason(store, meeting, minutes)
    assert reason is not None and "is no longer stored" in reason
    store.conn.execute("UPDATE transcripts SET deleted_at = 'x'")
    store.conn.commit()
    reason = stale_draft_reason(store, meeting, minutes)
    assert reason is not None and "no transcript" in reason
    assert invoke("approve", meeting.id).exit_code == 1


# --------------------------------------------------------------------------- R2: first ingests


def test_no_speech_ingest_is_audited_keeps_its_wav_and_can_be_retried(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch, _Engine(silent=True))
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "no segments" in result.output
    meeting = only_meeting(env.settings)
    assert meeting.state is MeetingState.created
    (failed,) = audit(env.settings, "ingest.failed")
    assert failed["classification"] == "internal"
    assert failed["detail"] == {"stage": "redact", "error": "PraktikaError", "purged": []}
    (media,) = live_media(env.settings, meeting.id)
    assert media.delete_after is not None and media.path.exists()
    use_engine(monkeypatch, _Engine(text=LINE))
    assert invoke("transcribe", meeting.id, "--no-diarize").exit_code == 0
    assert invoke("generate", meeting.id).exit_code == 0
    assert state_of(env.settings, meeting.id) is MeetingState.draft_ready


def test_no_speech_ingest_is_removed_by_abort(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_engine(monkeypatch, _Engine(silent=True))
    assert invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE).exit_code == 1
    meeting = only_meeting(env.settings)
    assert invoke("abort", meeting.id).exit_code == 0
    assert state_of(env.settings, meeting.id) is MeetingState.discarded
    assert wavs(env.settings, meeting.id) == [] and live_media(env.settings, meeting.id) == []


def test_empty_vtt_ingest_is_audited(env: _Holder, tmp_path: Path) -> None:
    vtt = tmp_path / "empty.vtt"
    vtt.write_text("WEBVTT\n\n", "utf-8")
    assert invoke("ingest", str(vtt), "--lang", "en", *GATE).exit_code == 1
    assert only_meeting(env.settings).state is MeetingState.created
    (failed,) = audit(env.settings, "ingest.failed")
    assert failed["detail"] == {"stage": "redact", "error": "PraktikaError", "purged": []}
    assert locks(env.settings) == []


def test_a_drafting_failure_in_a_first_ingest_is_audited(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    from praktika.errors import LLMError

    class _Down(FakeLLM):
        def complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise LLMError("LLM request failed: connection refused")

    monkeypatch.setattr(ctx, "llm_client", lambda settings: _Down())
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "praktika generate" in result.output
    (failed,) = audit(env.settings, "ingest.failed")
    assert failed["detail"] == {"stage": "draft", "error": "LLMError", "purged": []}
    assert only_meeting(env.settings).state is MeetingState.created


# --------------------------------------------------------------------------- the lock itself


def test_the_lock_is_released_on_success_failure_and_refusal(
    env: _Holder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting = ingest_with(env, monkeypatch)
    assert locks(env.settings) == []
    use_engine(monkeypatch, _Engine(silent=True))
    assert invoke("transcribe", meeting.id, "--no-diarize").exit_code == 1
    assert locks(env.settings) == []
    assert invoke("approve", meeting.id).exit_code == 0
    assert invoke("generate", meeting.id).exit_code == 1  # closed without --reopen
    assert locks(env.settings) == []


def test_a_second_lock_is_refused_until_the_first_is_released(tmp_path: Path) -> None:
    from helpers_foundation import meeting as base_meeting

    store, other = SqliteStore(tmp_path / "p.db"), SqliteStore(tmp_path / "p.db")
    meeting = base_meeting()
    store.save_meeting(meeting)
    first = new_run_lock(meeting.id, "ingest")
    assert store.take_run_lock(first) is None
    with pytest.raises(RunLockedError, match="an ingest run is working on this meeting"):
        other.take_run_lock(new_run_lock(meeting.id, "generate"))
    other.release_run_lock(meeting.id, "not-its-token")  # someone else's release is ignored
    assert store.live_run_lock(meeting.id) == first
    assert store.live_run_lock(meeting.id, exempt=first.token) is None
    store.release_run_lock(meeting.id, first.token)
    assert other.take_run_lock(new_run_lock(meeting.id, "generate")) is None
    with pytest.raises(KeyError):
        store.take_run_lock(new_run_lock("M-20260925-ffff", "generate"))


def test_holder_liveness() -> None:
    mine = new_run_lock("M-20260925-abcd", "generate")
    assert holder_alive(mine)
    assert holder_alive(mine.model_copy(update={"host": "another-host"})), "cannot be checked"
    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 - this interpreter
    proc.wait()
    assert not holder_alive(mine.model_copy(update={"pid": proc.pid})), "that process is gone"
    assert not holder_alive(mine.model_copy(update={"pid": 0}))


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc start times")
def test_a_reused_process_id_does_not_keep_a_lock_alive() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # noqa: S603
    try:
        lock = new_run_lock("M-20260925-abcd", "generate").model_copy(update={"pid": proc.pid})
        assert holder_alive(lock)
        earlier = lock.started_at.replace(year=lock.started_at.year - 1)
        assert not holder_alive(lock.model_copy(update={"started_at": earlier}))
    finally:
        proc.kill()
        proc.wait()


def test_a_version_1_database_is_upgraded_and_keeps_its_data(tmp_path: Path) -> None:
    from helpers_foundation import meeting as base_meeting

    path = tmp_path / "old.db"
    store = SqliteStore(path)
    store.save_meeting(base_meeting())
    store.conn.executescript(
        "DROP TABLE run_locks; DELETE FROM schema_version;"
        " INSERT INTO schema_version (version, applied_at) VALUES (1, 'then');"
    )
    store.close()
    conn = sqlite3.connect(path)
    assert db.current_version(conn) == 1
    assert (
        conn.execute("SELECT name FROM sqlite_master WHERE name = 'run_locks'").fetchone() is None
    )
    conn.close()
    upgraded = SqliteStore(path)
    assert db.current_version(upgraded.conn) == db.SCHEMA_VERSION == 2
    assert [m.id for m in upgraded.list_meetings()] == [base_meeting().id]
    lock = new_run_lock(base_meeting().id, "generate")
    assert upgraded.take_run_lock(lock) is None
    assert isinstance(upgraded.run_lock(base_meeting().id), RunLock)
    fresh = SqliteStore(tmp_path / "new.db")
    columns = "SELECT name, type, \"notnull\", pk FROM pragma_table_info('run_locks')"
    assert upgraded.conn.execute(columns).fetchall() == fresh.conn.execute(columns).fetchall()
