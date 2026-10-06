"""CLI behaviour with ``typer.testing.CliRunner``.

Everything runs offline: the LLM, STT, capture and vault key are the conftest fakes injected
through ``praktika.cli.context``; system probes used by ``doctor`` are monkeypatched.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import typer
from conftest import (
    FIXTURES,
    TONE_WAV,
    FakeCapturer,
    FakeDetector,
    FakeLLM,
    FakeTranscriber,
    make_transcript,
)
from cryptography.fernet import Fernet
from typer.testing import CliRunner

from praktika import consent, scope
from praktika.audio.convert import AudioInfo
from praktika.audit import verify_chain
from praktika.cli import app, doctor, meetings
from praktika.cli import context as ctx
from praktika.config import Settings
from praktika.ingest import audio_file
from praktika.ingest.audio_file import IngestedAudio, ingest_tracks
from praktika.models import (
    Attendee,
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
ROSTER = FIXTURES / "roster_data_team.yaml"
VTT_EN = FIXTURES / "synthetic_en.vtt"
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
FUTURE = "2027-06-01T00:00:00Z"


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def events(settings: Settings) -> list[str]:
    path = settings.data_dir / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln)["event"] for ln in path.read_text("utf-8").splitlines() if ln]


def store_of(settings: Settings) -> SqliteStore:
    return SqliteStore(settings.data_dir / "praktika.db")


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def cli_env(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch, fixed_key: bytes) -> Settings:
    monkeypatch.setattr(ctx, "load_settings", lambda: tmp_settings)
    monkeypatch.setattr(ctx, "vault_key", lambda settings=None: fixed_key)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: FakeLLM())
    return tmp_settings


@pytest.fixture
def fake_stt(monkeypatch: pytest.MonkeyPatch) -> FakeTranscriber:
    """ffmpeg-free conversion, one VAD chunk and a fake English engine."""

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
            ),
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
    return en


def ingest_vtt(cli_env: Settings) -> str:
    result = invoke(
        "ingest",
        str(VTT_EN),
        "--title",
        "Data team weekly",
        "--roster",
        str(ROSTER),
        "--lang",
        "en",
        *GATE,
    )
    assert result.exit_code == 0, result.output
    meetings_ = store_of(cli_env).list_meetings()
    assert len(meetings_) == 1
    return meetings_[0].id


# --------------------------------------------------------------------------- doctor


def _doctor_probes(
    monkeypatch: pytest.MonkeyPatch, *, ffmpeg: bool, encryption_status: str
) -> None:
    monkeypatch.setattr(doctor, "which", lambda name: "/opt/bin/ffmpeg" if ffmpeg else None)
    monkeypatch.setattr(doctor, "FDESETUP", sys.executable)
    monkeypatch.setattr(doctor, "FINDMNT", "/nonexistent/findmnt")  # force the fdesetup branch
    monkeypatch.setattr(doctor, "run_cmd", lambda args, timeout=5.0: encryption_status)
    monkeypatch.setattr(doctor, "physical_memory_bytes", lambda: 32 * 1024**3)
    monkeypatch.setattr(doctor, "host_platform", lambda: "darwin")  # the Keychain branch


def test_doctor_reports_missing(cli_env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = cli_env.model_copy(update={"llm_provider": "ollama"})
    monkeypatch.setattr(ctx, "load_settings", lambda: settings)
    _doctor_probes(monkeypatch, ffmpeg=False, encryption_status="Full-disk encryption is On.\n")

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(
        doctor, "http_client", lambda s: httpx.Client(transport=httpx.MockTransport(refused))
    )
    started = time.monotonic()
    result = invoke("doctor", "--json")
    assert time.monotonic() - started < 5.0
    assert result.exit_code == 1, result.output
    checks = {c["name"]: c for c in json.loads(result.stdout)}
    assert checks["ffmpeg"]["status"] == "warn" and "not found" in checks["ffmpeg"]["detail"]
    assert checks["ollama"]["status"] == "fail" and "unreachable" in checks["ollama"]["detail"]
    assert checks["disk_encryption"]["status"] == "ok"
    assert checks["no_egress"]["status"] == "ok"
    assert checks["data_dir"]["status"] == "ok"
    assert checks["models"]["status"] == "warn"  # fake STT: no register is a warning only
    assert checks["identity"]["status"] == "warn"  # fake identity is never trusted as session


def test_doctor_models_absent_warns_but_mismatch_is_hard(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No register yet is a setup warning even for the MLX backends; a register that no
    longer matches the weights on disk is a hard failure for them."""
    from praktika.models_registry import manage

    settings = cli_env.model_copy(update={"stt_en": "mlx_whisper", "stt_ar": "mlx_cohere"})
    monkeypatch.setattr(ctx, "load_settings", lambda: settings)
    _doctor_probes(monkeypatch, ffmpeg=True, encryption_status="Full-disk encryption is On.\n")
    result = invoke("doctor", "--json")
    assert result.exit_code == 0, result.output
    checks = {c["name"]: c for c in json.loads(result.stdout)}
    assert checks["models"]["status"] == "warn" and "models pull" in checks["models"]["detail"]

    weights = tmp_path / "stt_en"
    weights.mkdir()
    (weights / "weights.bin").write_bytes(b"synthetic")
    manage.register(
        "stt_en",
        weights,
        settings,
        repo="example/synthetic",
        revision="r1",
        licence="test",
        conversion=None,
    )
    (weights / "weights.bin").write_bytes(b"tampered")
    result = invoke("doctor", "--json")
    assert result.exit_code == 1
    checks = {c["name"]: c for c in json.loads(result.stdout)}
    assert checks["models"]["status"] == "fail" and "stt_en" in checks["models"]["detail"]


def test_doctor_models_hint_suits_an_offline_http_host(cli_env: Settings) -> None:
    """A server with no internet that loads no weights (speech over HTTP): the missing-register
    hint must not send the operator to ``models pull``, which needs the Hub."""
    http_host = cli_env.model_copy(update={"stt_en": "http", "stt_ar": "none"})
    check = doctor.check_models(http_host)
    assert check.status == "warn"
    assert "speech runs over HTTP" in check.detail and "does not need one" in check.detail
    assert "praktika models register stt_en <dir>" in check.detail
    assert "docs/DEPLOYMENT.md" in check.detail
    assert "models pull" not in check.detail
    local = doctor.check_models(cli_env.model_copy(update={"stt_en": "faster_whisper"}))
    assert "models pull stt_en` on a connected machine" in local.detail
    assert "models register stt_en <dir>` for weights copied in" in local.detail


def test_doctor_models_hint_on_linux_never_pulls_weights_that_cannot_run(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default ``stt_en=mlx_whisper`` cannot run on Linux: the hint points to the HTTP
    speech server and docs/DEPLOYMENT.md, not to ``models pull``; macOS keeps the pull hint."""
    defaults = cli_env.model_copy(update={"stt_en": "mlx_whisper", "stt_ar": "none"})
    monkeypatch.setattr(doctor, "host_platform", lambda: "linux")
    check = doctor.check_models(defaults)
    assert check.status == "warn"
    assert "stt_en=mlx_whisper cannot run on this host" in check.detail
    assert "PRAKTIKA_STT_EN=http" in check.detail and "PRAKTIKA_STT_HTTP_URL" in check.detail
    assert "docs/DEPLOYMENT.md" in check.detail
    assert "models pull" not in check.detail and "PRAKTIKA_STT_AR" not in check.detail
    assert not re.search(r"\b(apple|mac|macos|mlx)\b", check.detail, re.I), check.detail
    both = doctor.check_models(cli_env.model_copy(update={"stt_ar": "mlx_cohere"}))
    assert "stt_ar=mlx_cohere" in both.detail and "PRAKTIKA_STT_AR=http or none" in both.detail
    assert "PRAKTIKA_STT_EN" not in both.detail, "stt_en=fake needs no change"
    monkeypatch.setattr(doctor, "host_platform", lambda: "darwin")
    assert "models pull stt_en` on a connected machine" in doctor.check_models(defaults).detail


def test_missing_prompts_say_praktika_runs_from_an_editable_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel install (or a moved checkout) has no prompts/ or glossary.yaml beside the
    package: the error says to run from an editable checkout or point the two variables at it."""
    env_file = tmp_path / "praktika.env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(env_file))
    monkeypatch.setenv("PRAKTIKA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PRAKTIKA_PROMPTS_DIR", str(tmp_path / "site-packages" / "prompts"))
    monkeypatch.setenv("PRAKTIKA_GLOSSARY_PATH", str(tmp_path / "site-packages" / "glossary.yaml"))
    result = invoke("config", "show")
    assert result.exit_code == 1
    text = " ".join(result.output.split())
    assert "configured path(s) not found: prompts_dir=" in text and "glossary_path=" in text
    assert "git checkout installed in editable mode" in text
    assert "set PRAKTIKA_PROMPTS_DIR and PRAKTIKA_GLOSSARY_PATH to the checkout's" in text


def test_doctor_reports_the_env_file_it_loaded(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from praktika import config

    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    _doctor_probes(monkeypatch, ffmpeg=True, encryption_status="Full-disk encryption is On.\n")
    env_file = tmp_path / "praktika.env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(env_file))
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert f"env_file        {env_file} (named by PRAKTIKA_ENV_FILE)" in result.output
    monkeypatch.delenv("PRAKTIKA_ENV_FILE")
    check = {c["name"]: c for c in json.loads(invoke("doctor", "--json").stdout)}["env_file"]
    assert check["status"] == "warn" and "none loaded" in check["detail"]
    assert "defaults and the process environment apply" in check["detail"]


def test_every_command_refuses_a_missing_named_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PRAKTIKA_ENV_FILE`` set before the file is written (or mistyped) stops every command,
    including those that read no settings, instead of running on the defaults."""
    missing = tmp_path / "etc" / "praktika.env"
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(missing))
    monkeypatch.setenv("PRAKTIKA_DATA_DIR", str(tmp_path / "data"))
    for args in (["config", "show"], ["consent-script"], ["doctor"], ["audit", "verify"]):
        result = invoke(*args)
        assert result.exit_code == 1, (args, result.output)
        assert f"PRAKTIKA_ENV_FILE={missing} does not exist" in result.output, args
    assert not (tmp_path / "data").exists(), "nothing ran"
    with pytest.raises(typer.Exit):
        ctx.load_settings()
    missing.parent.mkdir()
    missing.write_text("PRAKTIKA_REVIEW_PORT=8800\n", encoding="utf-8")
    result = invoke("config", "show")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["review_port"] == 8800


def test_doctor_ok_exits_zero_and_unencrypted_disk_is_hard(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _doctor_probes(monkeypatch, ffmpeg=True, encryption_status="Full-disk encryption is On.\n")
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "[OK  ] ffmpeg" in result.output and "[FAIL]" not in result.output
    _doctor_probes(monkeypatch, ffmpeg=True, encryption_status="Full-disk encryption is Off.\n")
    result = invoke("doctor", "--json")
    assert result.exit_code == 1
    assert {c["name"]: c["status"] for c in json.loads(result.stdout)}["disk_encryption"] == "fail"


def test_doctor_ollama_model_missing(cli_env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = cli_env.model_copy(update={"llm_provider": "ollama"})
    monkeypatch.setattr(ctx, "load_settings", lambda: settings)
    _doctor_probes(monkeypatch, ffmpeg=True, encryption_status="Full-disk encryption is On.\n")

    def tags(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
        return httpx.Response(200, json={"parameters": "num_ctx 32768"})

    monkeypatch.setattr(
        doctor, "http_client", lambda s: httpx.Client(transport=httpx.MockTransport(tags))
    )
    result = invoke("doctor", "--json")
    ollama = {c["name"]: c for c in json.loads(result.stdout)}["ollama"]
    assert result.exit_code == 1 and ollama["status"] == "fail"
    assert "qwen2.5:14b" in ollama["detail"] and "not pulled" in ollama["detail"]


# --------------------------------------------------------------------------- ingest / gate


def test_ingest_vtt_end_to_end_fake_llm(cli_env: Settings) -> None:
    result = invoke(
        "ingest",
        str(VTT_EN),
        "--title",
        "Data team weekly",
        "--roster",
        str(ROSTER),
        "--lang",
        "en",
        *GATE,
    )
    assert result.exit_code == 0, result.output
    assert "Review: http://127.0.0.1:8793/" in result.output
    store = store_of(cli_env)
    (meeting,) = store.list_meetings()
    assert meeting.state is MeetingState.draft_ready
    assert meeting.organiser == "f.khalid@acme.test"
    assert [a.name for a in meeting.roster][:2] == ["F. Khalid", "R. Haddad"]
    assert meeting.room_identities == ["AI Lab Meeting Room"]
    minutes = store.latest_minutes(meeting.id)
    assert minutes is not None and minutes.version == 1 and minutes.review.status == "draft"
    assert minutes.decisions and minutes.actions
    transcript = store.get_transcript(meeting.id)
    assert transcript is not None and transcript.redacted and transcript.source == "vtt"
    assert any("«" in s.text for s in transcript.segments), "identifiers tokenised"
    assert store.get_vault(meeting.id) is not None
    record = store.get_consent(meeting.id)
    assert record is not None and record.method == "chat"
    assert set(record.scope_checks) == scope.scope_check_keys()
    assert all(record.scope_checks.values())
    seen = events(cli_env)
    for name in ("consent.recorded", "ingest.vtt", "redact.applied", "llm.call", "minutes.drafted"):
        assert name in seen, name
    assert seen.index("consent.recorded") < seen.index("ingest.vtt") < seen.index("llm.call")
    assert verify_chain(cli_env.data_dir / "audit.jsonl") == (True, None)
    assert mode(cli_env.data_dir / "audit.jsonl") == 0o600
    assert mode(cli_env.data_dir / "praktika.db") == 0o600
    hit = invoke("search", "pilot")
    assert hit.exit_code == 0 and "No matches" in hit.output, "drafts are never searchable"
    assert invoke("approve", meeting.id).exit_code == 0
    hit = invoke("search", "pilot")
    assert hit.exit_code == 0 and meeting.id in hit.output, "approved minutes are indexed"


def test_ingest_refuses_when_gate_flags_missing(cli_env: Settings) -> None:
    flags = [f for f in GATE if f != "--notified"]
    result = invoke("ingest", str(VTT_EN), *flags)
    assert result.exit_code == 2, result.output
    assert "--notified/--no-notified" in result.output
    assert not (cli_env.data_dir / "praktika.db").exists() or not store_of(cli_env).list_meetings()
    assert "consent.recorded" not in events(cli_env)


def test_ingest_refuses_ar_mixed_when_arabic_is_off(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stt_ar = none`` refuses ``--lang ar-mixed`` at the gate: no meeting, no consent record,
    so a live meeting is never captured only to fail at transcription."""
    off = cli_env.model_copy(update={"stt_ar": "none"})
    monkeypatch.setattr(ctx, "load_settings", lambda: off)
    result = invoke("ingest", str(VTT_EN), "--lang", "ar-mixed", *GATE)
    assert result.exit_code != 0
    assert "Arabic transcription is switched off" in result.output
    assert not (off.data_dir / "praktika.db").exists() or not store_of(off).list_meetings()
    assert "consent.recorded" not in events(off)


def test_ingest_partial_scope_ack_lists_missing_keys(cli_env: Settings) -> None:
    flags = [f for f in GATE if f != "--ack-all-scope"] + ["--scope-ack", "not_board"]
    result = invoke("ingest", str(VTT_EN), *flags)
    assert result.exit_code == 2
    assert (
        "--scope-ack not_hr" in result.output
        and "not_board" not in result.output.split("no terminal to ask:")[1]
    )


def test_ingest_refuses_objections_and_audits(cli_env: Settings) -> None:
    flags = ["--objections" if f == "--no-objections" else f for f in GATE]
    result = invoke("ingest", str(VTT_EN), *flags)
    assert result.exit_code == 2 and "objection" in result.output
    assert events(cli_env) == ["scope.refused"]
    assert not store_of(cli_env).list_meetings()


def test_ingest_refuses_confidential_in_pilot(cli_env: Settings) -> None:
    result = invoke("ingest", str(VTT_EN), "--class", "confidential", *GATE)
    assert result.exit_code == 2 and "internal" in result.output
    assert "scope.refused" in events(cli_env)


def test_ingest_rejects_unsupported_suffix(cli_env: Settings, tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    result = invoke("ingest", str(bad), *GATE)
    assert result.exit_code == 1 and "unsupported input" in result.output


def test_ingest_audio_file_with_vtt_names(cli_env: Settings, fake_stt: FakeTranscriber) -> None:
    before = datetime.now(UTC)
    result = invoke(
        "ingest",
        str(TONE_WAV),
        "--title",
        "Recorded",
        "--roster",
        str(ROSTER),
        "--lang",
        "en",
        "--vtt",
        str(VTT_EN),
        *GATE,
    )
    assert result.exit_code == 0, result.output
    store = store_of(cli_env)
    (meeting,) = store.list_meetings()
    transcript = store.get_transcript(meeting.id)
    assert transcript is not None and transcript.source == "file"
    assert transcript.engines["stt_en"] == "fake-en" and "vtt" in transcript.engines
    assert transcript.segments[0].speaker_kind == "identity", "name inherited from the VTT"
    (media,) = store.list_media(meeting.id)
    wav = cli_env.data_dir / "audio" / meeting.id / "file.wav"
    assert media.path == wav and wav.exists() and mode(wav) == 0o600
    assert media.delete_after is not None
    assert timedelta(hours=23) < media.delete_after - before < timedelta(hours=25)
    assert fake_stt.unloaded == 1
    seen = events(cli_env)
    assert "ingest.file" in seen and "ingest.vtt" in seen and "stt.completed" in seen


def test_ingest_tracks_purges_audio_when_retention_is_zero(
    tmp_settings: Settings, fake_stt: FakeTranscriber
) -> None:
    settings = tmp_settings.model_copy(
        update={"retention_audio_hours": {"internal": 0, "confidential": 72, "restricted": 0}}
    )
    meeting = Meeting(
        id="M-20260916-a1b2",
        title="t",
        meeting_type=MeetingType.general,
        classification=Classification.internal,
        language_mode=LanguageMode.en,
        platform=Platform.in_room,
        started_at=datetime.now(UTC),
        organiser="o",
        roster=[Attendee(name="F. Khalid")],
    )
    log: list[str] = []

    class Audit:
        def append(self, event: str, meeting_id: str | None = None, **detail: Any) -> None:
            log.append(event)

    with pytest.raises(ValueError):
        ingest_tracks({}, meeting, settings, Audit())
    result = ingest_tracks({"file": TONE_WAV}, meeting, settings, Audit())
    assert (
        result.media == [] and not (settings.data_dir / "audio" / meeting.id / "file.wav").exists()
    )
    assert result.transcript.segments and not result.transcript.redacted
    assert log == ["ingest.file", "stt.completed", "retention.deleted"]  # the purge is audited


# --------------------------------------------------------------------------- export / approve


def test_export_refuses_draft(cli_env: Settings, tmp_path: Path) -> None:
    mid = ingest_vtt(cli_env)
    out = tmp_path / "minutes.md"
    result = invoke("export", mid, "--out", str(out))
    assert result.exit_code == 1 and "not approved" in result.output
    assert not out.exists()
    result = invoke("export", mid, "--out", str(out), "--allow-draft")
    assert result.exit_code == 1 and "pilot" in result.output
    assert "export.written" not in events(cli_env)


def test_approve_then_export_markdown_and_docx(cli_env: Settings, tmp_path: Path) -> None:
    mid = ingest_vtt(cli_env)
    result = invoke("approve", mid, "--reason", "reviewed")
    assert result.exit_code == 0, result.output
    store = store_of(cli_env)
    minutes = store.latest_minutes(mid)
    assert minutes is not None and minutes.review.status == "approved"
    assert minutes.review.reviewer == "f.khalid@acme.test"
    assert store.get_meeting(mid).state is MeetingState.approved  # type: ignore[union-attr]
    out = tmp_path / "minutes.md"
    result = invoke("export", mid, "--out", str(out))
    assert result.exit_code == 0, result.output
    text = out.read_text("utf-8")
    assert "Data team weekly" in text and "DRAFT" not in text and mode(out) == 0o600
    docx = tmp_path / "minutes.docx"
    assert invoke("export", mid, "--format", "docx", "--out", str(docx)).exit_code == 0
    assert docx.exists() and mode(docx) == 0o600
    assert events(cli_env).count("export.written") == 2
    assert "review.approved" in events(cli_env)


def test_approve_blocked_by_priority_one_flag(cli_env: Settings) -> None:
    mid = ingest_vtt(cli_env)
    store = store_of(cli_env)
    minutes = store.latest_minutes(mid)
    assert minutes is not None
    from praktika.models import Flag

    flagged = minutes.model_copy(
        update={"flags": [Flag(kind="uncited_item_removed", detail="x", priority=1)]}
    )
    store.set_review_status(flagged)
    result = invoke("approve", mid)
    assert result.exit_code == 1 and "blocking" in result.output
    assert store.latest_minutes(mid).review.status == "draft"  # type: ignore[union-attr]


# --------------------------------------------------------------------------- retention / hold


def test_retention_dry_run(cli_env: Settings) -> None:
    mid = ingest_vtt(cli_env)
    result = invoke("retention", "run", "--dry-run", "--now", FUTURE)
    assert result.exit_code == 0, result.output
    assert "Would delete" in result.output and "transcript" in result.output
    store = store_of(cli_env)
    assert store.get_transcript(mid) is not None and store.get_vault(mid) is not None
    assert "retention.deleted" not in events(cli_env)
    result = invoke("retention", "run", "--now", FUTURE)
    assert result.exit_code == 0, result.output
    assert store.get_transcript(mid) is None and store.get_vault(mid) is None
    assert store.latest_minutes(mid) is None, "unapproved draft removed after the window"
    assert events(cli_env).count("retention.deleted") == 3
    again = invoke("retention", "run", "--now", FUTURE)
    assert "Deleted 0 item(s)" in again.output


def test_hold_blocks_retention(cli_env: Settings) -> None:
    mid = ingest_vtt(cli_env)
    result = invoke("hold", "set", mid, "--reason", "litigation")
    assert result.exit_code == 0
    store = store_of(cli_env)
    assert store.get_meeting(mid).legal_hold is True  # type: ignore[union-attr]
    assert "Deleted 0 item(s)" in invoke("retention", "run", "--now", FUTURE).output
    assert store.get_transcript(mid) is not None
    assert invoke("hold", "clear", mid).exit_code == 0
    assert store.get_meeting(mid).legal_hold is False  # type: ignore[union-attr]
    assert [e for e in events(cli_env) if e.startswith("hold.")] == ["hold.set", "hold.released"]


# --------------------------------------------------------------------------- consent / start


def test_consent_script_outputs_both_languages(cli_env: Settings) -> None:
    result = invoke("consent-script")
    assert result.exit_code == 0
    assert consent.SCRIPT_EN[:60] in result.output
    assert consent.SCRIPT_AR[:30] in result.output
    assert consent.BANNER_EN[:40] in result.output and consent.BANNER_AR[:20] in result.output
    only_en = invoke("consent-script", "--lang", "en")
    assert consent.SCRIPT_EN[:60] in only_en.output and consent.SCRIPT_AR[:30] not in only_en.output
    assert invoke("consent-script", "--lang", "fr").exit_code == 1


def test_start_sck_refuses_unsigned_helper(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "praktika-capture"
    helper.write_text("#!/bin/sh\necho unsigned\n", encoding="utf-8")
    helper.chmod(0o755)
    args = ["start", "--title", "Weekly", "--source", "sck", "--helper", str(helper), *GATE]
    result = invoke(*args)
    assert result.exit_code == 2, result.output
    assert "PRAKTIKA_CAPTURE_TEAM_ID" in result.output, "no pinned team id: refused"
    pinned = cli_env.model_copy(update={"capture_team_id": "ABCDE12345"})
    monkeypatch.setattr(ctx, "load_settings", lambda: pinned)
    result = invoke(*args)
    assert result.exit_code == 2, result.output
    assert "not Developer ID signed by team ABCDE12345" in result.output
    assert not (cli_env.data_dir / "praktika.db").exists()
    assert not (cli_env.data_dir / "audio").exists()
    assert consent.SCRIPT_EN[:40] not in result.output, "refused before the gate"


def test_start_mic_end_to_end(cli_env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    capturer = FakeCapturer(("mic",))
    monkeypatch.setattr(ctx, "build_capturer", lambda source, helper, device=None: capturer)
    seen: dict[str, Any] = {}

    def fake_ingest(sources: dict, meeting: Meeting, settings: Settings, audit: Any, **kw: Any):
        seen.update(sources)
        return IngestedAudio(
            transcript=make_transcript(meeting_id=meeting.id, source="capture", redacted=False)
        )

    monkeypatch.setattr(meetings, "ingest_tracks", fake_ingest)
    result = invoke(
        "start", "--title", "Weekly", "--roster", str(ROSTER), "--duration", "0.1", *GATE
    )
    assert result.exit_code == 0, result.output
    assert consent.SCRIPT_EN[:60] in result.output and consent.SCRIPT_AR[:30] in result.output
    assert "Review: http://127.0.0.1:8793/" in result.output
    assert capturer.events == ["start", "stop"]
    assert list(seen) == ["mic"] and seen["mic"].name == "mic.wav"
    store = store_of(cli_env)
    (meeting,) = store.list_meetings()
    assert meeting.state is MeetingState.draft_ready and meeting.platform is Platform.in_room
    assert store.latest_minutes(meeting.id) is not None
    seen_events = events(cli_env)
    assert seen_events.index("capture.started") < seen_events.index("capture.stopped")
    assert not (cli_env.data_dir / "audio" / meeting.id / "capture.pid").exists()


def test_abort_purges_audio(cli_env: Settings) -> None:
    store = store_of(cli_env)
    meeting = Meeting(
        id="M-20260916-c3d4",
        title="t",
        meeting_type=MeetingType.general,
        classification=Classification.internal,
        language_mode=LanguageMode.en,
        platform=Platform.in_room,
        started_at=datetime.now(UTC),
        organiser="o",
        roster=[],
        state=MeetingState.capturing,
    )
    store.save_meeting(meeting)
    store.close()
    audio_dir = cli_env.data_dir / "audio" / meeting.id
    audio_dir.mkdir(parents=True)
    wav = audio_dir / "mic.wav"
    wav.write_bytes(b"\x7f" * 2048)
    result = invoke("abort", meeting.id)
    assert result.exit_code == 0, result.output
    assert not wav.exists()
    assert store_of(cli_env).get_meeting(meeting.id).state is MeetingState.discarded  # type: ignore[union-attr]
    assert "capture.aborted" in events(cli_env)


# --------------------------------------------------------------------------- re-runs


def test_transcribe_and_generate_new_version(cli_env: Settings, fake_stt: FakeTranscriber) -> None:
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 0, result.output
    store = store_of(cli_env)
    (meeting,) = store.list_meetings()
    result = invoke("transcribe", meeting.id, "--lang", "en", "--no-diarize")
    assert result.exit_code == 0, result.output
    assert fake_stt.unloaded == 2
    result = invoke("generate", meeting.id, "--prompt-version", "v1")
    assert result.exit_code == 0, result.output
    assert store.latest_minutes(meeting.id).version == 2  # type: ignore[union-attr]
    assert invoke("generate", "M-20260916-ffff").exit_code == 1


# --------------------------------------------------------------------------- misc commands


def test_models_register_and_verify(cli_env: Settings, tmp_path: Path) -> None:
    weights = tmp_path / "stt_en"
    weights.mkdir()
    (weights / "weights.safetensors").write_bytes(b"\x00" * 64)
    result = invoke("models", "register", "stt_en", str(weights))
    assert result.exit_code == 0, result.output
    assert (cli_env.models_dir / "models.yaml").exists()
    assert invoke("models", "verify").exit_code == 0
    (weights / "weights.safetensors").write_bytes(b"\x01" * 64)
    result = invoke("models", "verify")
    assert result.exit_code == 1 and "hash mismatch" in result.output
    assert invoke("models", "register", "stt_en", str(tmp_path / "missing")).exit_code == 1


def test_config_show_prints_settings(cli_env: Settings) -> None:
    result = invoke("config", "show")
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["llm_provider"] == "fake" and data["data_dir"] == str(cli_env.data_dir)


def test_audit_verify_and_tail(cli_env: Settings) -> None:
    ingest_vtt(cli_env)
    assert invoke("audit", "verify").exit_code == 0
    tail = invoke("audit", "tail", "-n", "3")
    assert tail.exit_code == 0 and len(tail.output.strip().splitlines()) == 3
    path = cli_env.data_dir / "audit.jsonl"
    lines = path.read_text("utf-8").splitlines()
    lines[0] = lines[0].replace('"event":"consent.recorded"', '"event":"consent.forged"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = invoke("audit", "verify")
    assert result.exit_code == 1 and "line 1" in result.output


def test_serve_refuses_non_loopback_in_local_mode(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = cli_env.model_copy(update={"review_host": "0.0.0.0"})  # noqa: S104
    monkeypatch.setattr(ctx, "load_settings", lambda: settings)
    result = invoke("serve")
    assert result.exit_code == 1 and "loopback" in result.output


def test_serve_builds_the_review_app(cli_env: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """``praktika serve`` hands ``server.serve`` an app over the CLI's store, identity, audit
    log and LLM client, on the port given; nothing is bound in the test."""
    from fastapi.testclient import TestClient

    from praktika import server

    mid = ingest_vtt(cli_env)
    served: list[tuple[Any, Settings]] = []
    monkeypatch.setattr(server, "serve", lambda app, settings: served.append((app, settings)))
    result = invoke("serve", "--port", "8911")
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8911/?t=" in result.output
    token = re.search(r"\?t=([A-Za-z0-9_-]+)", result.output).group(1)  # type: ignore[union-attr]
    app, settings = served[0]
    assert settings.review_port == 8911 and settings.review_host == "127.0.0.1"
    headers = {"X-Praktika-Review": "1", "X-Praktika-Token": token}
    assert TestClient(app, base_url="http://127.0.0.1").get("/api/meetings").status_code == 401
    client = TestClient(app, base_url="http://127.0.0.1", headers=headers)
    assert [m["id"] for m in client.get("/api/meetings").json()] == [mid]
    detail = client.get(f"/api/meetings/{mid}").json()
    assert detail["regenerate_available"] is True, "the configured LLM client is wired in"
    assert detail["viewer"]["user"] == "f.khalid@acme.test"
    assert "review.opened" in events(cli_env)


def test_dsar_find_export_delete(cli_env: Settings, tmp_path: Path) -> None:
    mid = ingest_vtt(cli_env)
    found = invoke("dsar", "find", "--participant", "Omar")
    assert found.exit_code == 0 and f"{mid}  (roster)" in found.output
    assert "No meetings" in invoke("dsar", "find", "--participant", "Nobody Here").output
    out = tmp_path / "bundle.json"
    result = invoke("dsar", "export", "--participant", "Omar", "--out", str(out))
    assert result.exit_code == 0 and mode(out) == 0o600
    bundle = json.loads(out.read_text("utf-8"))
    assert len(bundle) == 1 and bundle[0]["consent"]["meeting_id"] == mid
    result = invoke("dsar", "delete", "--participant", "Omar", "--yes")
    assert result.exit_code == 0, result.output
    store = store_of(cli_env)
    assert store.get_meeting(mid).state is MeetingState.purged  # type: ignore[union-attr]
    assert store.get_transcript(mid) is None and store.latest_minutes(mid) is None
    assert store.get_vault(mid) is None
    seen = events(cli_env)
    assert "dsar.export" in seen and "dsar.delete" in seen
    audit_text = (cli_env.data_dir / "audit.jsonl").read_text("utf-8")
    assert "Omar" not in audit_text.split("dsar.export")[1], "participant name never audited"


def test_actions_register(cli_env: Settings) -> None:
    mid = ingest_vtt(cli_env)
    assert "No open actions" in invoke("actions").output, "drafts stay off the register"
    assert invoke("approve", mid).exit_code == 0
    result = invoke("actions")
    assert result.exit_code == 0 and "Omar Nasser" in result.output and mid in result.output
    assert "No open actions" in invoke("actions", "--owner", "nobody").output


def test_eval_fake_passes_gates(cli_env: Settings, tmp_path: Path) -> None:
    report = tmp_path / "eval_report.md"
    result = invoke("eval", "--llm", "fake", "--out", str(report))
    assert result.exit_code == 0, result.output
    assert "Gate: PASS" in result.output and report.exists()
    assert "Evaluated 6 meeting(s)" in result.output
    assert invoke("eval", "--llm", "other").exit_code == 1


def test_cli_never_builds_consent_records_outside_the_gate() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "praktika"
    sources = list((root / "cli").glob("*.py")) + [root / "ingest" / "audio_file.py"]
    for path in sources:
        text = path.read_text("utf-8")
        assert "ConsentRecord(" not in text, path
        assert not re.search(r"--(skip|bypass|force)-?(gate|consent)", text), path


# --------------------------------------------------------------------------- closed-state guards


def test_approve_refuses_discarded_and_generate_needs_reopen(cli_env: Settings) -> None:
    mid = ingest_vtt(cli_env)
    store = store_of(cli_env)
    minutes = store.latest_minutes(mid)
    assert minutes is not None
    discarded = minutes.review.model_copy(update={"status": "discarded"})
    store.set_review_status(minutes.model_copy(update={"review": discarded}))
    store.set_state(mid, MeetingState.discarded)
    store.close()
    result = invoke("approve", mid)
    assert result.exit_code == 1 and "discarded" in result.output
    assert store_of(cli_env).latest_minutes(mid).review.status == "discarded"  # type: ignore[union-attr]

    other = invoke("ingest", str(VTT_EN), "--title", "Second", "--lang", "en", *GATE)
    assert other.exit_code == 0, other.output
    mid2 = next(m.id for m in store_of(cli_env).list_meetings() if m.title == "Second")
    assert invoke("approve", mid2).exit_code == 0
    result = invoke("generate", mid2)
    assert result.exit_code == 1 and "--reopen" in result.output
    assert store_of(cli_env).latest_minutes(mid2).version == 1  # type: ignore[union-attr]
    assert store_of(cli_env).get_meeting(mid2).state is MeetingState.approved  # type: ignore[union-attr]
    result = invoke("generate", mid2, "--reopen")
    assert result.exit_code == 0, result.output
    assert store_of(cli_env).latest_minutes(mid2).version == 2  # type: ignore[union-attr]
    assert "review.reopened" in events(cli_env)


# --------------------------------------------------------------------------- export destinations


def test_export_defaults_to_data_dir_and_refuses_icloud(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from praktika.cli import export_paths

    mid = ingest_vtt(cli_env)
    assert invoke("approve", mid).exit_code == 0
    result = invoke("export", mid)
    assert result.exit_code == 0, result.output
    target = cli_env.data_dir / "exports" / f"{mid}-v1.md"
    assert target.exists() and mode(target) == 0o600 and str(target) in result.output
    assert mode(target.parent) == 0o700

    icloud = tmp_path / "Mobile Documents"
    (icloud / "com~apple~CloudDocs").mkdir(parents=True)
    monkeypatch.setattr(export_paths, "ICLOUD_ROOT", icloud)
    cloud_target = icloud / "com~apple~CloudDocs" / "minutes.md"
    result = invoke("export", mid, "--out", str(cloud_target))
    assert result.exit_code == 1 and "iCloud" in result.output and not cloud_target.exists()
    result = invoke("export", mid, "--out", str(cloud_target), "--force")
    assert result.exit_code == 0 and cloud_target.exists()
    result = invoke("dsar", "export", "--participant", "Omar")
    assert result.exit_code == 0 and (cli_env.data_dir / "exports" / "dsar-export.json").exists()
    bad = invoke("dsar", "export", "--participant", "Omar", "--out", str(icloud / "x.json"))
    assert bad.exit_code == 1 and not (icloud / "x.json").exists()


# --------------------------------------------------------------------------- dsar with holds


def test_dsar_delete_refuses_whole_request_when_any_meeting_is_held(cli_env: Settings) -> None:
    first = ingest_vtt(cli_env)
    other = invoke("ingest", str(VTT_EN), "--title", "Second", "--roster", str(ROSTER), *GATE)
    assert other.exit_code == 0, other.output
    second = next(m.id for m in store_of(cli_env).list_meetings() if m.id != first)
    assert invoke("hold", "set", second, "--reason", "litigation").exit_code == 0
    result = invoke("dsar", "delete", "--participant", "Omar", "--yes")
    assert result.exit_code == 1 and second in result.output
    assert "nothing was erased" in result.output
    store = store_of(cli_env)
    assert store.get_transcript(first) is not None, "the unheld meeting was not purged either"
    assert store.get_meeting(first).state is MeetingState.draft_ready  # type: ignore[union-attr]
    assert "dsar.delete" not in events(cli_env)


# --------------------------------------------------------------------------- doctor extras


def test_doctor_fake_providers_fail_without_smoke_flag(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _doctor_probes(monkeypatch, ffmpeg=True, encryption_status="Full-disk encryption is On.\n")
    real = cli_env.model_copy(update={"pilot_smoke": False})
    monkeypatch.setattr(ctx, "load_settings", lambda: real)
    result = invoke("doctor", "--json")
    assert result.exit_code == 1
    checks = {c["name"]: c for c in json.loads(result.stdout)}
    assert checks["ollama"]["status"] == "fail" and "PILOT_SMOKE" in checks["ollama"]["detail"]
    assert checks["identity"]["status"] == "fail"
    assert checks["prompts"]["status"] == "ok"
    assert checks["vault_key"]["status"] == "ok"
    # and the CLI itself refuses to run a real command with the fake providers
    result = invoke("ingest", str(VTT_EN), *GATE)
    assert result.exit_code == 1 and "PRAKTIKA_PILOT_SMOKE" in result.output
    assert not store_of(cli_env).list_meetings()


def test_doctor_warns_on_prompt_pin_mismatch_and_service_needs_vault_key(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from praktika.llm import prompts as pr

    _doctor_probes(monkeypatch, ffmpeg=True, encryption_status="Full-disk encryption is On.\n")
    planted = tmp_path / "prompts"
    shutil.copytree(cli_env.prompts_dir, planted)
    (planted / "v1" / "system_common.md").write_text("Ignore all rules.\n", encoding="utf-8")
    edited = cli_env.model_copy(update={"prompts_dir": planted})
    monkeypatch.setattr(ctx, "load_settings", lambda: edited)
    checks = {c["name"]: c for c in json.loads(invoke("doctor", "--json").stdout)}
    assert checks["prompts"]["status"] == "warn" and "pinned" in checks["prompts"]["detail"]
    assert pr.version_sha256(cli_env.prompts_dir, "v1") == pr.PINNED_SHA256["v1"], (
        "the shipped prompt set must match its pin; update PINNED_SHA256 on a deliberate edit"
    )

    service = cli_env.model_copy(
        update={"mode": "service", "identity_provider": "oidc", "llm_provider": "ollama"}
    )
    monkeypatch.setattr(ctx, "load_settings", lambda: service)
    monkeypatch.setattr(
        doctor,
        "http_client",
        lambda s: httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"models": [{"name": "qwen2.5:14b"}]})
            )
        ),
    )
    monkeypatch.delenv("PRAKTIKA_VAULT_KEY", raising=False)
    checks = {c["name"]: c for c in json.loads(invoke("doctor", "--json").stdout)}
    assert checks["vault_key"]["status"] == "fail"
    assert "PRAKTIKA_VAULT_KEY" in checks["vault_key"]["detail"]
    monkeypatch.setenv("PRAKTIKA_VAULT_KEY", Fernet.generate_key().decode())
    checks = {c["name"]: c for c in json.loads(invoke("doctor", "--json").stdout)}
    assert checks["vault_key"]["status"] == "ok"
    # a key that is present but unusable is reported as it is: every ingest would fail on it
    monkeypatch.setenv("PRAKTIKA_VAULT_KEY", "not-a-fernet-key")
    checks = {c["name"]: c for c in json.loads(invoke("doctor", "--json").stdout)}
    assert (
        checks["vault_key"]["status"] == "fail"
        and "not a Fernet key" in (checks["vault_key"]["detail"])
    )
    # service mode without oidc authenticates nobody, whatever the bind address
    for provider in ("session", "fake"):
        weak = service.model_copy(update={"identity_provider": provider, "pilot_smoke": True})
        monkeypatch.setattr(ctx, "load_settings", lambda weak=weak: weak)
        result = invoke("doctor", "--json")
        checks = {c["name"]: c for c in json.loads(result.stdout)}
        assert result.exit_code == 1
        assert checks["identity"]["status"] == "fail" and "oidc" in checks["identity"]["detail"]


def test_audit_verify_detects_truncation(cli_env: Settings) -> None:
    ingest_vtt(cli_env)
    assert invoke("audit", "verify").exit_code == 0
    path = cli_env.data_dir / "audit.jsonl"
    lines = path.read_text("utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    assert verify_chain(path) == (True, None), "a prefix alone still looks valid"
    result = invoke("audit", "verify")
    assert result.exit_code == 1 and "truncated" in result.output
    checks = {c["name"]: c for c in json.loads(invoke("doctor", "--json").stdout)}
    assert checks["audit_chain"]["status"] == "fail"
    path.unlink()
    result = invoke("audit", "verify")
    assert result.exit_code == 1 and "missing" in result.output


# --------------------------------------------------------------------------- scope tags


def test_ingest_tag_exclusion_is_refused_in_code(cli_env: Settings) -> None:
    result = invoke("ingest", str(VTT_EN), "--title", "Weekly", "--tag", "HR", *GATE)
    assert result.exit_code == 2 and "hr" in result.output.lower()
    assert not store_of(cli_env).list_meetings() and events(cli_env) == ["scope.refused"]
    result = invoke("ingest", str(VTT_EN), "--title", "Weekly", "--foreign-hosted", *GATE)
    assert result.exit_code == 2 and "foreign" in result.output.lower()
    assert "no such option" not in result.output.lower()
    result = invoke(
        "ingest", str(VTT_EN), "--title", "Grievance hearing", "--tag", "weekly", "--lang", "en",
        *GATE,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "WARNING: the title suggests a pilot exclusion (hr)" in result.output
    (meeting,) = store_of(cli_env).list_meetings()
    assert meeting.tags == {"weekly"}


# --------------------------------------------------------------------------- silence (C-04)


class _SilentCapturer(FakeCapturer):
    def check_silence(self) -> bool:
        return True

    def levels(self) -> tuple[float, float]:
        return (0.0, 0.0)


def test_start_warns_loudly_when_no_audio(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    capturer = _SilentCapturer(("mic",))
    monkeypatch.setattr(ctx, "build_capturer", lambda source, helper, device=None: capturer)
    monkeypatch.setattr(
        meetings,
        "ingest_tracks",
        lambda sources, meeting, settings, audit, **kw: IngestedAudio(
            transcript=make_transcript(meeting_id=meeting.id, source="capture", redacted=False),
            silent_tracks=["mic"],
        ),
    )
    result = invoke("start", "--title", "Weekly", "--duration", "0.6", *GATE)
    assert result.exit_code == 0, result.output
    assert "NO AUDIO DETECTED" in result.output
    assert "the mic track is mostly silent" in result.output
    assert meetings._status("M-20260916-a1b2", "en", 3.0, (0.0, 0.0), silent=True).plain.endswith(
        meetings.SILENCE_LINE
    )


# --------------------------------------------------------------------------- ingest failure (C-05)


def test_failed_transcription_leaves_no_wav_behind(
    cli_env: Settings, fake_stt: FakeTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    from praktika.errors import PraktikaError

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise PraktikaError("model weights not pulled")

    monkeypatch.setattr(router, "transcribe_track", boom)
    result = invoke("ingest", str(TONE_WAV), "--lang", "en", *GATE)
    assert result.exit_code == 1 and "weights" in result.output
    store = store_of(cli_env)
    (meeting,) = store.list_meetings()
    wav_dir = cli_env.data_dir / "audio" / meeting.id
    assert not list(wav_dir.glob("*.wav")), "the converted WAV was purged on failure"
    assert store.list_media(meeting.id) == []


def test_retention_sweeps_orphan_audio(cli_env: Settings) -> None:
    orphan_dir = cli_env.data_dir / "audio" / "M-20260901-dead"
    orphan_dir.mkdir(parents=True)
    orphan = orphan_dir / "file.wav"
    orphan.write_bytes(b"\x7f" * 4096)
    old = time.time() - 3 * 24 * 3600
    os.utime(orphan, (old, old))
    fresh = orphan_dir / "mic.wav"
    fresh.write_bytes(b"\x7f" * 16)
    result = invoke("retention", "run", "--dry-run")
    assert result.exit_code == 0 and "orphan" in result.output and orphan.exists()
    result = invoke("retention", "run")
    assert result.exit_code == 0, result.output
    assert not orphan.exists() and fresh.exists(), "only orphans past the window go"
    assert "retention.deleted" in events(cli_env)


def test_retention_run_reports_a_locked_file_and_deletes_the_rest(cli_env: Settings) -> None:
    """One un-wipeable orphan neither aborts the run nor hides the failure: the other orphan
    is deleted, ``retention.failed`` is audited and the command exits 1 naming the file."""
    locked_dir = cli_env.data_dir / "audio" / "M-20260901-aaaa"
    other_dir = cli_env.data_dir / "audio" / "M-20260901-bbbb"
    locked_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    locked, other = locked_dir / "file.wav", other_dir / "file.wav"
    old = time.time() - 3 * 24 * 3600
    for wav in (locked, other):
        wav.write_bytes(b"\x7f" * 4096)
        os.utime(wav, (old, old))
    locked.chmod(0o400)
    locked_dir.chmod(0o500)
    try:
        result = invoke("retention", "run")
    finally:
        locked_dir.chmod(0o700)
        locked.chmod(0o600)
    assert result.exit_code == 1, result.output
    assert "1 retention deletion(s) failed" in result.output and locked.name in result.output
    assert locked.exists() and not other.exists()
    seen = events(cli_env)
    assert "retention.failed" in seen and "retention.deleted" in seen
    assert invoke("retention", "run").exit_code == 0 and not locked.exists()


def test_user_commands_run_retention_opportunistically(cli_env: Settings) -> None:
    mid = ingest_vtt(cli_env)
    store = store_of(cli_env)
    (media_dir := cli_env.data_dir / "audio" / mid).mkdir(parents=True, exist_ok=True)
    wav = media_dir / "file.wav"
    wav.write_bytes(b"\x7f" * 64)
    store.save_media(mid, wav, "a" * 64, kind="file", delete_after=datetime.now(UTC))
    store.close()
    assert invoke("approve", mid).exit_code == 0, "approval closes the meeting"
    assert invoke("actions").exit_code == 0, "any later user command runs the timers"
    assert not wav.exists(), "audio of an approved meeting was deleted without `retention run`"
    seen = events(cli_env)
    assert "retention.deleted" in seen


def test_retention_install_writes_launchd_agent(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from praktika.cli import ops

    monkeypatch.setattr(ops, "host_platform", lambda: "darwin")
    target = tmp_path / "agent.plist"
    result = invoke("retention", "install", "--out", str(target))
    assert result.exit_code == 0, result.output
    text = target.read_text("utf-8")
    assert mode(target) == 0o600 and ops.LAUNCHD_LABEL in text
    assert "<string>retention</string><string>run</string>" in text
    assert "<key>StartInterval</key><integer>3600</integer>" in text
    assert str(cli_env.data_dir / "retention.log") in text and "launchctl load" in result.output


def test_review_url_carries_the_persistent_token(tmp_settings) -> None:
    """The review link must carry the session token: the server refuses a link without it."""
    from praktika.cli import context as ctx
    from praktika.server_auth import session_token_for

    url = ctx.review_url(tmp_settings, "M-20260915-c3d4")
    assert url.startswith(f"http://127.0.0.1:{tmp_settings.review_port}/?t=")
    assert session_token_for(tmp_settings.data_dir) in url
    assert url.endswith("#/meetings/M-20260915-c3d4")


def test_disk_encryption_on_linux_reads_the_data_volume(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux the check resolves the data directory's volume and asks whether it is a
    dm-crypt mapping; a plain volume is a warning (IT's confirmation is the record of truth)."""
    monkeypatch.setattr(doctor, "FINDMNT", sys.executable)
    monkeypatch.setattr(doctor, "LSBLK", sys.executable)

    def probe(kind: str):
        def run(args: list[str], timeout: float = 5.0) -> str | None:
            if "SOURCE" in args:
                return "/dev/mapper/data\n"
            return f"{kind}\n"

        return run

    monkeypatch.setattr(doctor, "run_cmd", probe("crypt"))
    ok = doctor.check_disk_encryption(tmp_settings)
    assert ok.status == "ok" and "/dev/mapper/data" in ok.detail
    monkeypatch.setattr(doctor, "run_cmd", probe("lvm"))
    plain = doctor.check_disk_encryption(tmp_settings)
    assert plain.status == "warn" and "confirm" in plain.detail


def test_dsar_find_says_why_a_meeting_matched(cli_env: Settings) -> None:
    """A title match shows the title, so a false match is visible before anything is erased;
    a single word never matches a title."""
    result = invoke(
        "ingest", str(VTT_EN), "--title", "1:1 with Layla Farouk", "--lang", "en", *GATE
    )
    assert result.exit_code == 0, result.output
    mid = store_of(cli_env).list_meetings()[0].id
    found = invoke("dsar", "find", "--participant", "Layla Farouk")
    assert f"{mid}  (title: 1:1 with Layla Farouk)" in found.output
    assert "No meetings found." in invoke("dsar", "find", "--participant", "Layla").output
    assert (
        f"{mid}  (organiser)"
        in invoke("dsar", "find", "--participant", "f.khalid@acme.test").output
    )


def test_retention_now_without_an_offset_is_utc() -> None:
    from datetime import UTC

    from praktika.cli import ops

    assert ops._dt("2026-10-06T10:00:00").tzinfo is UTC
