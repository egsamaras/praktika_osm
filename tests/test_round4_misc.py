"""Round 4, miscellaneous fixes (offline, synthetic).

- ``llm.call``, ``minutes.drafted`` and ``stt.completed`` carry the meeting's classification, so
  a SIEM rule keyed on classification sees every model call, draft and transcription; the legal
  hold and DSAR export events do too.
- Console log lines carry ANSI colour codes only when stderr is a terminal: a pipe, a file, the
  journal or captured test evidence gets plain text (and ``NO_COLOR`` turns colour off).
- ``retention install`` pins an absolute ``PRAKTIKA_ENV_FILE`` (and data directory), and refuses
  a named env file that does not exist, instead of writing a unit that refuses every run.
- ``doctor``'s ``no_egress`` line lists ``allowed_hosts`` as plain comma-separated patterns.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import FIXTURES, FROZEN_NOW, REPO, FakeLLM, FakeTranscriber, make_transcript
from typer.testing import CliRunner

from praktika import config
from praktika.cli import app, doctor, ops
from praktika.cli import context as ctx
from praktika.config import Settings
from praktika.errors import LLMError
from praktika.llm import pipeline
from praktika.llm.base import AuditedClient
from praktika.llm.prompts import PromptSet, load
from praktika.logging import configure_logging, get_logger, stderr_wants_colour
from praktika.models import (
    Attendee,
    Classification,
    LanguageMode,
    Meeting,
    MeetingType,
    Platform,
    RawSegment,
)
from praktika.store.repo import SqliteStore
from praktika.stt import router

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
VTT_EN = FIXTURES / "synthetic_en.vtt"
ROSTER = FIXTURES / "roster_data_team.yaml"
WAV_FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_meeting.wav"
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
MEETING_ID = "M-20260916-a1b2"


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def audit_lines(settings: Settings) -> list[dict[str, Any]]:
    path = settings.data_dir / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text("utf-8").splitlines() if ln]


@pytest.fixture
def cli_env(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch, fixed_key: bytes) -> Settings:
    monkeypatch.setattr(ctx, "load_settings", lambda: tmp_settings)
    monkeypatch.setattr(ctx, "vault_key", lambda settings=None: fixed_key)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: FakeLLM())
    return tmp_settings


class Recorder:
    """An ``AuditLike`` that keeps every call, keyword arguments included."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, dict[str, Any]]] = []

    def append(self, event: str, meeting_id: str | None, **detail: Any) -> None:
        self.events.append((event, meeting_id, detail))

    def named(self, event: str) -> list[dict[str, Any]]:
        return [d for e, _, d in self.events if e == event]


# --------------------------------------------------------------------------- classification


@pytest.fixture(scope="module")
def prompts() -> PromptSet:
    return load(REPO / "prompts", "v1", MeetingType.general)


def _meeting(roster: list[Attendee], classification: Classification) -> Meeting:
    return Meeting(
        id=MEETING_ID,
        title="Data team weekly",
        meeting_type=MeetingType.general,
        classification=classification,
        language_mode=LanguageMode.en,
        platform=Platform.teams,
        started_at=datetime.fromisoformat(FROZEN_NOW),
        organiser="f.khalid@acme.test",
        roster=roster,
    )


@pytest.mark.parametrize("classification", [Classification.internal, Classification.confidential])
def test_generate_audits_every_model_call_and_the_draft_with_the_classification(
    roster: list[Attendee], prompts: PromptSet, classification: Classification
) -> None:
    audit = Recorder()
    meeting = _meeting(roster, classification)
    options = pipeline.GenerateOptions(template=MeetingType.general)
    pipeline.generate(
        make_transcript("en"), meeting, FakeLLM(), prompts, "b" * 64, None, options, audit
    )
    calls, drafted = audit.named("llm.call"), audit.named("minutes.drafted")
    assert calls and len(drafted) == 1
    for detail in [*calls, *drafted]:
        assert detail["classification"] == classification.value, detail
    assert all(mid == MEETING_ID for _, mid, _ in audit.events)


def test_audited_client_classifies_the_call_even_when_the_model_fails() -> None:
    class Broken(FakeLLM):
        def complete_json(self, *a: Any, **k: Any) -> dict[str, Any]:
            raise LLMError("connection refused")

    audit = Recorder()
    client = AuditedClient(Broken(), audit, MEETING_ID, "a" * 64, classification="internal")
    with pytest.raises(LLMError):
        client.complete_json("s", "u", {"title": "ChunkFindings"})
    (detail,) = audit.named("llm.call")
    assert detail["classification"] == "internal" and detail["schema"] == "ChunkFindings"

    # the four positional arguments other callers use still work; nothing is invented
    plain = AuditedClient(FakeLLM(), audit, MEETING_ID, "a" * 64)
    assert plain.classification is None


def test_stt_completed_carries_the_classification(tmp_settings: Settings) -> None:
    en = FakeTranscriber(
        [RawSegment(start=0.0, end=2.0, text="hello", language="en", confidence=0.9, engine="e")]
    )
    audit = Recorder()
    router.transcribe_track(
        WAV_FIXTURE,
        "file",
        "en",
        tmp_settings,
        audit,
        engines=(en, FakeTranscriber([]), None),
        meeting_id=MEETING_ID,
        classification="internal",
    )
    (detail,) = audit.named("stt.completed")
    assert detail["classification"] == "internal"


def test_cli_ingest_and_generate_audit_model_calls_with_the_classification(
    cli_env: Settings,
) -> None:
    result = invoke(
        "ingest", str(VTT_EN), "--title", "Data team weekly", "--roster", str(ROSTER),
        "--lang", "en", *GATE,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (meeting,) = SqliteStore(cli_env.data_dir / "praktika.db").list_meetings()
    result = invoke("generate", meeting.id, "--prompt-version", "v1")
    assert result.exit_code == 0, result.output
    assert invoke("hold", "set", meeting.id, "--reason", "litigation").exit_code == 0
    assert invoke("hold", "clear", meeting.id).exit_code == 0
    assert invoke("dsar", "export", "--participant", "Omar Nasser").exit_code == 0

    lines = audit_lines(cli_env)
    want = {"llm.call", "minutes.drafted", "hold.set", "hold.released", "dsar.export"}
    seen = {e["event"] for e in lines}
    assert want <= seen, seen
    for line in lines:
        if line["event"] in want and line["meeting_id"] == meeting.id:
            assert line["classification"] == "internal", line
    drafted = [e for e in lines if e["event"] == "minutes.drafted"]
    assert len(drafted) == 2, "the ingest's draft and the generate's draft"


# --------------------------------------------------------------------------- colour on stderr


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.fixture
def restore_logging() -> Iterator[None]:
    saved = sys.stderr
    try:
        yield
    finally:
        sys.stderr = saved
        configure_logging(json=False, level="INFO")


def _warn_to(stream: _Stream) -> str:
    sys.stderr = stream
    configure_logging(json=False, level="INFO")
    get_logger("praktika.test").warning("ingest.audio_purged_on_error", file="file.wav")
    return stream.getvalue()


def test_console_log_is_plain_when_stderr_is_not_a_terminal(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    text = _warn_to(_Stream(tty=False))
    assert "ingest.audio_purged_on_error" in text and "file=file.wav" in text
    assert not ANSI.search(text), repr(text)


def test_console_log_is_coloured_only_on_a_terminal_without_no_color(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert ANSI.search(_warn_to(_Stream(tty=True))), "a terminal keeps its colours"
    monkeypatch.setenv("NO_COLOR", "1")
    text = _warn_to(_Stream(tty=True))
    assert "ingest.audio_purged_on_error" in text and not ANSI.search(text)


def test_stderr_that_cannot_answer_counts_as_not_a_terminal(
    restore_logging: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    class Closed(io.StringIO):
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    sys.stderr = Closed()
    assert stderr_wants_colour() is False
    sys.stderr = object()  # type: ignore[assignment]  # no isatty at all
    assert stderr_wants_colour() is False


def test_a_real_redirected_stderr_gets_no_escape_sequences(tmp_path: Path) -> None:
    """The process a test harness or the journal captures: stderr is a pipe, not a terminal."""
    env = {k: v for k, v in os.environ.items() if k != "NO_COLOR" and k != "FORCE_COLOR"}
    code = (
        "from praktika.logging import get_logger\n"
        "get_logger('praktika.test').warning('run.stopped', meeting_id='M-20260916-a1b2')\n"
    )
    done = subprocess.run(  # noqa: S603 - fixed interpreter and code
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert "run.stopped" in done.stderr and not ANSI.search(done.stderr), repr(done.stderr)


# --------------------------------------------------------------------------- retention install


def _service_env(unit_dir: Path) -> dict[str, str]:
    text = (unit_dir / "praktika-retention.service").read_text("utf-8")
    return dict(re.findall(r'^Environment="([A-Z_]+)=([^"]*)"$', text, flags=re.MULTILINE))


@pytest.fixture
def linux_install(cli_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A Linux host whose user unit directory is under ``tmp_path``; returns that directory."""
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config" / "systemd" / "user"


def test_retention_install_pins_a_relative_env_file_as_an_absolute_path(
    linux_install: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    etc = tmp_path / "etc" / "praktika"
    etc.mkdir(parents=True)
    env_file = etc / "praktika.env"
    env_file.write_text("PRAKTIKA_PILOT=true\n", encoding="utf-8")
    monkeypatch.chdir(etc)
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", "praktika.env")

    result = invoke("retention", "install")

    assert result.exit_code == 0, result.output
    pinned = _service_env(linux_install)["PRAKTIKA_ENV_FILE"]
    assert Path(pinned).is_absolute() and Path(pinned) == env_file.absolute()
    assert f"Pinned PRAKTIKA_ENV_FILE={pinned}" in " ".join(result.output.split())
    # what ExecStart sees: a system unit runs from /, a user unit from the home directory
    monkeypatch.chdir("/")
    assert config.env_file_path(environ={"PRAKTIKA_ENV_FILE": pinned}) == Path(pinned)


def test_env_file_to_pin_makes_relative_names_absolute_and_keeps_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = tmp_path / "real.env"
    real.write_text("PRAKTIKA_PILOT=true\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "link.env").symlink_to(real)
    monkeypatch.chdir(tmp_path / "sub")
    pinned = ops.env_file_to_pin({"PRAKTIKA_ENV_FILE": "../link.env"})
    assert pinned is not None and pinned.is_absolute()
    assert pinned.name == "link.env", "the name the operator gave, not the link's target"
    assert pinned.resolve() == real.resolve()
    assert ops.env_file_to_pin({"PRAKTIKA_ENV_FILE": str(real)}) == real


def test_retention_install_refuses_a_named_env_file_that_does_not_exist(
    linux_install: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    for named in ("praktika.env", str(tmp_path / "missing.env"), str(tmp_path)):
        monkeypatch.setenv("PRAKTIKA_ENV_FILE", named)
        result = invoke("retention", "install")
        assert result.exit_code == 1, (named, result.output)
        assert result.exc_info is None or result.exc_info[0] is SystemExit, named
        assert "PRAKTIKA_ENV_FILE=" in result.output, named
        assert not linux_install.exists(), "no unit is written for an env file that is not there"


def test_retention_install_pins_an_absolute_data_dir(
    cli_env: Settings, linux_install: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "rel").mkdir()
    monkeypatch.chdir(tmp_path)
    relative = cli_env.model_copy(update={"data_dir": Path("rel")})
    monkeypatch.setattr(ctx, "load_settings", lambda: relative)
    result = invoke("retention", "install")
    assert result.exit_code == 0, result.output
    assert _service_env(linux_install)["PRAKTIKA_DATA_DIR"] == str(tmp_path / "rel")


# --------------------------------------------------------------------------- doctor


def test_doctor_no_egress_lists_allowed_hosts_plainly(tmp_settings: Settings) -> None:
    check = doctor.check_egress(tmp_settings)
    assert (check.name, check.status) == ("no_egress", "ok")
    assert check.detail == "allowed_hosts=localhost, 127.0.0.1"
    assert "[" not in check.detail and "'" not in check.detail


def test_doctor_no_egress_line_as_printed(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "which", lambda name: "/opt/bin/ffmpeg")
    result = invoke("doctor")
    line = next(ln for ln in result.output.splitlines() if ln[7:].startswith("no_egress "))
    assert line == "[OK  ] no_egress       allowed_hosts=localhost, 127.0.0.1", line


def test_doctor_missing_ollama_model_lists_the_served_models_plainly(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = tmp_settings.model_copy(update={"llm_provider": "ollama"})

    def tags(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            names = [{"name": "llama3.1:8b"}, {"name": "gemma:2b"}]
            return httpx.Response(200, json={"models": names})
        return httpx.Response(200, json={"parameters": ""})

    monkeypatch.setattr(
        doctor, "http_client", lambda s: httpx.Client(transport=httpx.MockTransport(tags))
    )
    check = doctor.check_ollama(settings)
    assert check.status == "fail"
    assert check.detail.endswith("not pulled (have: gemma:2b, llama3.1:8b)"), check.detail
