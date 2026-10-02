"""The command-line surface an operator reads on a Linux server: exit codes, help, error text.

Covers these regressions on the CLI surface:

- ``--lang ar-mixed`` with the Arabic path off is a refusal (exit 2), refused before the
  consent script (``LanguageRefused``).
- An env-file location this account cannot look into (``sudo -E`` keeping the caller's
  ``HOME``) is a clean error that says to export ``PRAKTIKA_ENV_FILE``; a subcommand's
  ``--help`` still prints; an env file that sets no ``PRAKTIKA_`` variable makes doctor warn;
  doctor always prints its thirteen checks.
- ``retention install`` pins ``PRAKTIKA_ENV_FILE`` only when there is a file to pin.
- ``models verify`` prints the full repo and revision in a narrow, non-terminal output.
- The review page and ``praktika actions`` show the spoken due phrase beside a resolved date.
- Text read on a Linux server names no Apple product; ``serve`` calls the token persistent;
  the chat-banner copy is skipped silently where there is no clipboard tool.

Offline and synthetic throughout; system probes are monkeypatched.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tomllib
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import FIXTURES, TONE_WAV, FakeLLM
from rich.console import Console
from typer.main import get_command
from typer.testing import CliRunner

import praktika
from praktika import config, consent
from praktika.cli import app, doctor, ops, review_cmd
from praktika.cli import context as ctx
from praktika.config import Settings
from praktika.errors import ConfigError, LanguageRefused, PraktikaError
from praktika.models import ActionItem, Ref
from praktika.redact import tokenise
from praktika.store.repo import SqliteStore
from praktika.stt import router

runner = CliRunner()
REPO = Path(__file__).resolve().parent.parent
VTT_EN = FIXTURES / "synthetic_en.vtt"
APP_JS = REPO / "src" / "praktika" / "static" / "app.js"
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
#: Words that must not appear in text a user or engineer reads on a Linux server.
PLATFORM_WORDS = re.compile(
    r"\b(apple|mac|macos|macbook|icloud|filevault|keychain|launchd|mlx|laptop)\b|~/Library",
    re.IGNORECASE,
)
REVISION = "0123456789abcdef0123456789abcdef01234567"  # 40 hex characters, synthetic
REPO_ID = "example-org/synthetic-speech-model-large-v3-turbo"


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def no_traceback(result: Any) -> bool:
    """True when the command ended through ``typer.Exit``/``SystemExit`` (or cleanly), never
    through an uncaught exception."""
    return result.exc_info is None or result.exc_info[0] is SystemExit


@pytest.fixture
def cli(tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch, fixed_key: bytes) -> Settings:
    monkeypatch.setattr(ctx, "load_settings", lambda: tmp_settings)
    monkeypatch.setattr(ctx, "vault_key", lambda settings=None: fixed_key)
    monkeypatch.setattr(ctx, "llm_client", lambda settings: FakeLLM())
    return tmp_settings


@pytest.fixture
def locked_dir(tmp_path: Path) -> Iterator[Path]:
    """A directory this account cannot search (mode 000), like another account's home."""
    if os.geteuid() == 0:
        pytest.skip("root can search any directory")
    locked = tmp_path / "other-home"
    (locked / "praktika").mkdir(parents=True)
    locked.chmod(0)
    try:
        yield locked
    finally:
        locked.chmod(0o700)


@pytest.fixture
def linux_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The Linux default-location rules, with ``XDG_DATA_HOME`` under ``tmp_path``."""
    monkeypatch.setattr(config.sys, "platform", "linux")
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    return xdg / "praktika"


# --------------------------------------------------------------------------- D1: ar-mixed


def test_language_refused_is_a_refusal_type() -> None:
    assert issubclass(LanguageRefused, PraktikaError)
    assert LanguageRefused in ctx.REFUSALS


def test_ar_mixed_with_arabic_off_exits_2_before_the_consent_script(
    cli: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    off = cli.model_copy(update={"stt_ar": "none"})
    monkeypatch.setattr(ctx, "load_settings", lambda: off)
    for args in (
        ["ingest", str(VTT_EN), "--lang", "ar-mixed", *GATE],
        ["ingest", str(TONE_WAV), "--lang", "ar-mixed", *GATE],
        ["start", "--title", "Weekly", "--lang", "ar-mixed", *GATE],
    ):
        result = invoke(*args)
        assert result.exit_code == 2, (args, result.output)
        assert no_traceback(result)
        text = " ".join(result.output.split())
        assert "Arabic transcription is switched off" in text, args
        assert "refused before the consent script" in text and "use --lang en" in text
        assert consent.SCRIPT_EN[:40] not in result.output, "refused before the consent script"
        assert "Consent script" not in result.output
    assert not (off.data_dir / "praktika.db").exists(), "no meeting row, no consent record"
    with pytest.raises(LanguageRefused):
        router.require_language(off, "ar-mixed")


# --------------------------------------------------------------------------- D2: env file


def test_unsearchable_default_env_location_is_a_clean_config_error(locked_dir: Path) -> None:
    """``sudo -E`` keeps the caller's ``HOME``; ``Path.exists`` then raises instead of
    returning False. That is a ``ConfigError`` telling the operator to export the variable."""
    environ = {"XDG_DATA_HOME": str(locked_dir)}
    with pytest.raises(ConfigError) as info:
        config.env_file_path("linux", environ)
    message = str(info.value)
    assert str(locked_dir / "praktika" / ".env") in message
    assert "PRAKTIKA_ENV_FILE is not set" in message
    assert "Export PRAKTIKA_ENV_FILE" in message and "sudo -E" in message
    with pytest.raises(ConfigError, match="PRAKTIKA_ENV_FILE is empty"):
        config.env_file_path("linux", {**environ, "PRAKTIKA_ENV_FILE": ""})


def test_every_command_refuses_the_unsearchable_default_without_a_traceback(
    locked_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(locked_dir))
    for args in (["doctor"], ["audit", "verify"], ["consent-script"], ["retention", "run"]):
        result = invoke(*args)
        assert result.exit_code == 1, (args, result.output)
        assert no_traceback(result), (args, result.exc_info)
        assert "Export PRAKTIKA_ENV_FILE" in " ".join(result.output.split()), args
        assert "Traceback" not in result.output and "PermissionError" not in result.output


def test_subcommand_help_prints_when_the_env_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(tmp_path / "praktika.env.typo"))
    for args in (
        ["ingest", "--help"],
        ["doctor", "--help"],
        ["retention", "install", "--help"],
        ["models", "verify", "--help"],
        ["audit", "verify", "--help"],
        ["--log-level", "WARNING", "transcribe", "--help"],
    ):
        result = invoke(*args)
        assert result.exit_code == 0, (args, result.output)
        assert result.output.startswith("Usage: "), args
    refused = invoke("consent-script")
    assert refused.exit_code == 1 and "does not exist" in refused.output, "no --help: refused"
    passed_through = invoke("consent-script", "--", "--help")
    assert passed_through.exit_code != 0 and "Usage: " not in passed_through.output.split("\n")[0]


def test_subcommand_help_prints_when_the_default_location_is_unsearchable(
    locked_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(locked_dir))
    result = invoke("doctor", "--help")
    assert result.exit_code == 0 and result.output.startswith("Usage: "), result.output


def test_doctor_warns_on_an_env_file_that_sets_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, linux_defaults: Path
) -> None:
    named = tmp_path / "praktika.env"
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(named))
    for placeholder in ("", "# settings to be written here\n\n", "OTHER_TOOL=1\n"):
        named.write_text(placeholder, encoding="utf-8")
        check = doctor.check_env_file()
        assert check.status == "warn", (placeholder, check)
        assert f"{named} (named by PRAKTIKA_ENV_FILE)" in check.detail
        assert "sets no PRAKTIKA_ variable" in check.detail
    named.write_text("PRAKTIKA_REVIEW_PORT=8800\n", encoding="utf-8")
    assert doctor.check_env_file() == doctor.Check(
        "env_file", "ok", f"{named} (named by PRAKTIKA_ENV_FILE)"
    )
    named.write_text("export praktika_data_dir=/srv/x\n", encoding="utf-8")
    assert doctor.check_env_file().status == "ok", "dotenv syntax and case are honoured"

    monkeypatch.delenv("PRAKTIKA_ENV_FILE")
    default = linux_defaults / ".env"
    default.parent.mkdir(parents=True)
    default.write_text("", encoding="utf-8")
    check = doctor.check_env_file()
    assert check.status == "warn" and "default location" in check.detail
    assert "sets no PRAKTIKA_ variable" in check.detail
    default.write_text("PRAKTIKA_PILOT=true\n", encoding="utf-8")
    assert doctor.check_env_file().status == "ok"


def test_doctor_prints_thirteen_checks_even_when_the_data_dir_is_unsearchable(
    cli: Settings, locked_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "which", lambda name: None)
    monkeypatch.setattr(doctor, "run_cmd", lambda args, timeout=5.0: None)
    monkeypatch.setattr(doctor, "physical_memory_bytes", lambda: 32 * 1024**3)
    settings = cli.model_copy(update={"data_dir": locked_dir / "praktika" / "data"})
    checks = doctor.run_checks(settings)
    assert [c.name for c in checks] == [
        "env_file",
        "ffmpeg",
        "ollama",
        "models",
        "disk_encryption",
        "data_dir",
        "no_egress",
        "memory",
        "identity",
        "audit_chain",
        "prompts",
        "vault_key",
        "endpoint_agent",
    ]
    data_dir = next(c for c in checks if c.name == "data_dir")
    assert data_dir.status == "fail" and "ermission denied" in data_dir.detail


def test_a_data_dir_this_account_cannot_use_is_a_config_error(locked_dir: Path) -> None:
    with pytest.raises(ConfigError, match="cannot use the data directory"):
        ctx.ensure_data_dir(locked_dir / "praktika" / "data")


# --------------------------------------------------------------------------- retention install


def _service_env(unit_dir: Path) -> dict[str, str]:
    text = (unit_dir / "praktika-retention.service").read_text("utf-8")
    pairs = re.findall(r'^Environment="([A-Z_]+)=([^"]*)"$', text, flags=re.MULTILINE)
    return dict(pairs)


def test_retention_install_pins_no_env_file_when_there_is_none(
    cli: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, linux_defaults: Path
) -> None:
    """Without ``PRAKTIKA_ENV_FILE`` and without a default ``.env``, pinning the default path
    would make every hourly run refuse. The unit leaves it out, and the run it makes works."""
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = invoke("retention", "install")
    assert result.exit_code == 0, result.output
    unit_dir = tmp_path / "config" / "systemd" / "user"
    env = _service_env(unit_dir)
    assert env == {"PRAKTIKA_DATA_DIR": str(cli.data_dir)}, env
    assert "No env file pinned" in " ".join(result.output.split())
    timer_env = {**os.environ, **env}  # what ExecStart sees: no PRAKTIKA_ENV_FILE at all
    assert config.env_file_path(environ=timer_env) == linux_defaults / ".env", "the run starts"

    default = linux_defaults / ".env"
    default.parent.mkdir(parents=True)
    default.write_text("PRAKTIKA_PILOT=true\n", encoding="utf-8")
    assert invoke("retention", "install").exit_code == 0
    assert _service_env(unit_dir)["PRAKTIKA_ENV_FILE"] == str(default), "an existing default"

    named = tmp_path / "etc" / "praktika.env"
    named.parent.mkdir()
    named.write_text("PRAKTIKA_PILOT=true\n", encoding="utf-8")
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(named))
    result = invoke("retention", "install")
    assert result.exit_code == 0 and "No env file pinned" not in result.output
    assert _service_env(unit_dir)["PRAKTIKA_ENV_FILE"] == str(named), "a named file"


def test_systemd_units_without_an_env_file() -> None:
    service = ops.systemd_units(Path("/py"), None, Path("/var/lib/praktika"))[
        "praktika-retention.service"
    ]
    assert "PRAKTIKA_ENV_FILE" not in service
    assert 'Type=oneshot\nEnvironment="PRAKTIKA_DATA_DIR=/var/lib/praktika"\n' in service


def test_retention_install_help_names_no_platform_product() -> None:
    result = invoke("retention", "install", "--help")
    assert result.exit_code == 0, result.output
    assert not PLATFORM_WORDS.search(result.output), result.output
    command = get_command(app).commands["retention"].commands["install"]  # type: ignore[attr-defined]
    texts = [command.help or ""] + [getattr(p, "help", "") or "" for p in command.params]
    for text in texts:
        assert not PLATFORM_WORDS.search(text), text
    hidden = {p.name: p.hidden for p in command.params}  # type: ignore[attr-defined]
    systemd = ops.SYSTEMD_HOST
    assert hidden == {"out": systemd, "unit_dir": not systemd, "run_as": not systemd}


# --------------------------------------------------------------------------- models verify


def test_models_verify_prints_full_repo_and_revision_when_narrow(
    cli: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / "stt_en"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"synthetic weights")
    registered = invoke(
        "models", "register", "stt_en", str(weights),
        "--repo", REPO_ID, "--revision", REVISION, "--licence", "MIT",
    )  # fmt: skip
    assert registered.exit_code == 0, registered.output
    narrow = Console(width=60, markup=False, highlight=False, emoji=False, soft_wrap=True)
    monkeypatch.setattr(ctx, "console", narrow)
    result = invoke("models", "verify")
    assert result.exit_code == 0, result.output
    assert REPO_ID in result.output and REVISION in result.output
    assert "…" not in result.output
    assert "All registered models match their recorded hashes." in result.output


# --------------------------------------------------------------------------- D5: due phrases


def test_due_display_shows_the_spoken_phrase_beside_the_date() -> None:
    show = review_cmd.due_display
    assert show(date(2026, 10, 2), "November the 2nd") == '2026-10-02 ("November the 2nd")'
    assert show(date(2026, 10, 2), None) == "2026-10-02"
    assert show(date(2026, 10, 2), "2026-10-02") == "2026-10-02"
    assert show(None, "next month") == "next month"
    assert show(None, None) == ""


def test_actions_register_shows_the_spoken_phrase(
    cli: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = Ref(segment_id="S0001", start_s=0.0, end_s=1.0, speaker="F. Khalid", quote="by 2nd")
    action = ActionItem(
        id="A1",
        description="Send the draft",
        owner="Omar Nasser",
        owner_confidence="explicit",
        due_date=date(2026, 10, 2),
        due_text="by November the 2nd",
        source_language="en",
        refs=[ref],
    )
    item = SimpleNamespace(meeting_id="M-20260925-abcd", action=action)
    monkeypatch.setattr(SqliteStore, "open_actions", lambda self, owner=None: [item])
    monkeypatch.setattr(ctx, "console", Console(width=200, markup=False, highlight=False))
    result = invoke("actions")
    assert result.exit_code == 0, result.output
    assert '2026-10-02 ("by November the 2nd")' in result.output


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_review_page_shows_the_spoken_phrase_beside_the_date() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    fn = js[js.index("function dueLabel") :]
    fn = fn[: fn.index("\n}\n") + 3]
    cases = [
        ["2026-10-02", "November the 2nd"],
        ["2026-10-02", None],
        ["2026-10-02", "2026-10-02"],
        [None, "next month"],
        [None, None],
    ]
    call = f"{json.dumps(cases)}.map(([d, t]) => dueLabel(d, t))"
    out = subprocess.run(  # noqa: S603 - fixed local interpreter, script built from the repo
        [shutil.which("node") or "node", "-e", fn + f"console.log(JSON.stringify({call}));"],
        capture_output=True, text=True, timeout=20, check=True,
    )  # fmt: skip
    assert json.loads(out.stdout) == [
        "2026-10-02 (“November the 2nd”)",
        "2026-10-02",
        "2026-10-02",
        "next month",
        "",
    ]
    row = js[js.index("function itemRow") :]
    assert "dueLabel(item.due_date, item.due_text)" in row[: row.index("\n}\n")]


# --------------------------------------------------------------------------- wording


def test_top_level_help_and_metadata_say_english() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    assert "On-premises English meeting minutes" in result.output
    assert "bilingual" not in result.output and "EN/AR" not in result.output
    project = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]
    assert "bilingual" not in project["description"] and "EN/AR" not in project["description"]
    assert "bilingual" not in (praktika.__doc__ or "")


@pytest.mark.parametrize(
    ("mode", "platform"), [("local", "linux"), ("service", "linux"), ("local", "darwin")]
)
def test_doctor_vault_key_text_names_no_platform_product(
    tmp_settings: Settings, monkeypatch: pytest.MonkeyPatch, mode: str, platform: str
) -> None:
    monkeypatch.delenv("PRAKTIKA_VAULT_KEY", raising=False)
    monkeypatch.setattr(doctor, "host_platform", lambda: platform)
    check = doctor.check_vault_key(tmp_settings.model_copy(update={"mode": mode}))
    assert not PLATFORM_WORDS.search(check.detail), check.detail
    if platform == "linux":
        assert check.status == "fail" and "PRAKTIKA_VAULT_KEY" in check.detail


def test_vault_key_error_without_a_key_store_names_no_platform_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(tokenise.VAULT_KEY_ENV, raising=False)
    monkeypatch.setattr(tokenise, "_SECURITY", "/nonexistent/key-store-tool")
    with pytest.raises(PraktikaError) as info:
        tokenise.keychain_key()
    assert "no key store on this host" in str(info.value)
    assert "PRAKTIKA_VAULT_KEY" in str(info.value)
    assert not PLATFORM_WORDS.search(str(info.value)), str(info.value)


def test_review_page_names_no_platform_product() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    strings = re.findall(r'"[^"\n]*"|`[^`\n]*`', js)
    for text in strings:
        assert not PLATFORM_WORDS.search(text), text


def test_serve_calls_the_token_persistent(cli: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from praktika import server

    monkeypatch.setattr(server, "serve", lambda app_, settings: None)
    first = invoke("serve", "--port", "8912")
    assert first.exit_code == 0, first.output
    text = " ".join(first.output.split())
    assert "persistent session key" in text and "this run's" not in text
    assert str(cli.data_dir / "review.token") in text
    token = re.search(r"\?t=([A-Za-z0-9_-]+)", first.output).group(1)  # type: ignore[union-attr]
    again = invoke("serve", "--port", "8912")
    assert f"?t={token}" in again.output, "the same token after a restart"


def test_clipboard_copy_is_skipped_silently_without_a_clipboard_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ran: list[Any] = []
    warned: list[Any] = []
    monkeypatch.setattr(consent, "PBCOPY", str(tmp_path / "no-such-tool"))
    monkeypatch.setattr(consent.subprocess, "run", lambda *a, **kw: ran.append(a))
    monkeypatch.setattr(
        consent, "log", SimpleNamespace(warning=lambda *a, **kw: warned.append((a, kw)))
    )
    assert consent.copy_banner_to_clipboard("both") is False
    assert ran == [] and warned == [], "no attempt, no warning"


def test_env_file_error_never_names_a_platform_product(locked_dir: Path) -> None:
    with pytest.raises(ConfigError) as info:
        config.env_file_path("linux", {"XDG_DATA_HOME": str(locked_dir)})
    assert not PLATFORM_WORDS.search(str(info.value))


def test_locked_fixture_restores_access(locked_dir: Path) -> None:
    """Guard for the fixture itself: the directory really is unsearchable while in use."""
    assert stat.S_IMODE(locked_dir.stat().st_mode) == 0
    with pytest.raises(PermissionError):
        (locked_dir / "praktika" / ".env").exists()
