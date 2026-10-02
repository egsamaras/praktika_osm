"""Linux server behaviour of the CLI: systemd retention units, the vault-key doctor check,
English as the default language mode and ``ingest --organiser``.

Platform-dependent code reads ``host_platform()`` in the module under test, so both branches run
on any developer machine; everything stays offline and synthetic.
"""

from __future__ import annotations

import inspect
import json
import os
import pwd
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import FIXTURES, FakeLLM
from cryptography.fernet import Fernet
from typer.testing import CliRunner

from praktika.cli import app, doctor, meetings, ops
from praktika.cli import context as ctx
from praktika.config import Settings
from praktika.models import LanguageMode
from praktika.store.repo import SqliteStore

runner = CliRunner()
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


# --------------------------------------------------------------------------- systemd retention


def test_systemd_units_text_is_exact(tmp_path: Path) -> None:
    units = ops.systemd_units(
        Path("/opt/praktika/.venv/bin/python"),
        Path("/etc/praktika/praktika.env"),
        Path("/var/lib/praktika"),
    )
    assert set(units) == {"praktika-retention.service", "praktika-retention.timer"}
    service = units["praktika-retention.service"].splitlines()
    assert (
        'ExecStart="/opt/praktika/.venv/bin/python" "-m" "praktika.cli" "retention" "run"'
        in service
    )
    assert 'Environment="PRAKTIKA_ENV_FILE=/etc/praktika/praktika.env"' in service
    assert 'Environment="PRAKTIKA_DATA_DIR=/var/lib/praktika"' in service
    assert "Type=oneshot" in service and "UMask=0077" in service
    timer = units["praktika-retention.timer"].splitlines()
    assert "OnCalendar=hourly" in timer and "Persistent=true" in timer
    assert "Unit=praktika-retention.service" in timer and "WantedBy=timers.target" in timer


def test_systemd_exec_start_quotes_spaces_and_specifiers() -> None:
    line = ops.systemd_exec_start(Path('/srv/my env/100%/$HOME/"q"/python'))
    assert line.startswith('"/srv/my env/100%%/$$HOME/\\"q\\"/python" "-m" "praktika.cli"')


def test_retention_install_on_linux_writes_service_and_timer(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    env_file = tmp_path / "praktika.env"
    env_file.write_text("", encoding="utf-8")  # a named env file must exist
    monkeypatch.setenv("PRAKTIKA_ENV_FILE", str(env_file))
    unit_dir = tmp_path / "units"

    result = invoke("retention", "install", "--unit-dir", str(unit_dir))

    assert result.exit_code == 0, result.output
    service = unit_dir / "praktika-retention.service"
    timer = unit_dir / "praktika-retention.timer"
    assert sorted(p.name for p in unit_dir.iterdir()) == [service.name, timer.name]
    assert stat.S_IMODE(service.stat().st_mode) == 0o600
    assert stat.S_IMODE(timer.stat().st_mode) == 0o600
    text = service.read_text("utf-8")
    assert f'ExecStart="{sys.executable}" "-m" "praktika.cli" "retention" "run"\n' in text
    assert f'Environment="PRAKTIKA_ENV_FILE={env_file}"\n' in text
    assert f'Environment="PRAKTIKA_DATA_DIR={cli_env.data_dir}"\n' in text
    assert "OnCalendar=hourly" in timer.read_text("utf-8")
    assert "plist" not in text and not list(tmp_path.glob("**/*.plist"))
    assert (
        "systemctl daemon-reload" in result.output and "praktika-retention.timer" in result.output
    )


def test_retention_install_on_linux_defaults_to_user_unit_dir(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = invoke("retention", "install")
    assert result.exit_code == 0, result.output
    user_dir = tmp_path / "config" / "systemd" / "user"
    assert (user_dir / "praktika-retention.service").is_file()
    assert (user_dir / "praktika-retention.timer").is_file()
    assert "systemctl --user enable --now praktika-retention.timer" in result.output
    assert ops.systemd_user_unit_dir({}) == Path("~/.config/systemd/user").expanduser()


def test_retention_install_refuses_the_other_platforms_option(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    result = invoke("retention", "install", "--out", str(tmp_path / "agent.plist"))
    assert result.exit_code == 1 and "--unit-dir" in result.output
    assert not (tmp_path / "agent.plist").exists()
    monkeypatch.setattr(ops, "host_platform", lambda: "darwin")
    result = invoke("retention", "install", "--unit-dir", str(tmp_path / "units"))
    assert result.exit_code == 1 and "--out" in result.output
    assert not (tmp_path / "units").exists()


def test_system_unit_runs_as_the_data_directory_owner(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit written outside the user unit directory is a system unit: it gets ``User=``."""
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    cli_env.data_dir.mkdir(parents=True, exist_ok=True)
    me = pwd.getpwuid(os.getuid()).pw_name
    result = invoke("retention", "install", "--unit-dir", str(tmp_path / "system"))
    assert result.exit_code == 0, result.output
    service = (tmp_path / "system" / "praktika-retention.service").read_text("utf-8")
    assert f"\nUser={me}\n" in service
    assert "--user" not in result.output


def test_retention_install_never_writes_a_root_unit(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a system unit without ``User=`` (or a root user unit) runs as root."""
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    system = tmp_path / "system"
    result = invoke("retention", "install", "--unit-dir", str(system), "--run-as", "root")
    assert result.exit_code == 1 and "root" in result.output
    assert not system.exists()
    monkeypatch.setattr(ops, "effective_uid", lambda: 0)
    result = invoke("retention", "install")
    assert result.exit_code == 1 and "root" in result.output
    assert not (tmp_path / "config").exists()


def test_retention_install_run_as_is_checked(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    system = tmp_path / "system"
    for bad in ("x\nExecStartPre=/bin/sh", "no-such-account-praktika"):
        result = invoke("retention", "install", "--unit-dir", str(system), "--run-as", bad)
        assert result.exit_code == 1, result.output
    assert not system.exists()
    result = invoke("retention", "install", "--run-as", "someone")
    assert result.exit_code == 1 and "system unit" in result.output
    units = ops.systemd_units(Path("/py"), Path("/e"), Path("/d"), user="praktika")
    assert "Type=oneshot\nUser=praktika\n" in units["praktika-retention.service"]
    assert (
        "User="
        not in ops.systemd_units(Path("/py"), Path("/e"), Path("/d"))["praktika-retention.service"]
    )


def test_user_unit_install_prints_the_linger_hint(
    cli_env: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ops, "host_platform", lambda: "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    result = invoke("retention", "install")
    assert result.exit_code == 0, result.output
    assert "loginctl enable-linger" in result.output
    service = tmp_path / "config" / "systemd" / "user" / "praktika-retention.service"
    assert "User=" not in service.read_text("utf-8")


# --------------------------------------------------------------------------- doctor vault key


@pytest.mark.parametrize(
    ("platform", "value", "status"),
    [
        ("linux", None, "fail"),
        ("linux", "garbage", "fail"),
        ("linux", "valid", "ok"),
        ("darwin", None, "ok"),
        ("darwin", "garbage", "fail"),
    ],
)
def test_doctor_vault_key_reports_the_real_state(
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    value: str | None,
    status: str,
) -> None:
    """Local mode on Linux has no Keychain: an unset key is a hard failure, not OK."""
    monkeypatch.setattr(doctor, "host_platform", lambda: platform)
    if value is None:
        monkeypatch.delenv("PRAKTIKA_VAULT_KEY", raising=False)
    else:
        key = Fernet.generate_key().decode() if value == "valid" else value
        monkeypatch.setenv("PRAKTIKA_VAULT_KEY", key)
    check = doctor.check_vault_key(tmp_settings)
    assert tmp_settings.mode == "local"
    assert check.status == status, check.detail
    if platform == "linux" and value is None:
        assert "PRAKTIKA_VAULT_KEY" in check.detail and "no key store" in check.detail


# --------------------------------------------------------------------------- English default


def test_start_and_ingest_default_to_english() -> None:
    for command in (meetings.start, meetings.ingest):
        assert inspect.signature(command).parameters["lang"].default == LanguageMode.en


def test_ingest_without_lang_records_english(cli_env: Settings) -> None:
    result = invoke("ingest", str(VTT_EN), "--title", "Weekly", *GATE)
    assert result.exit_code == 0, result.output
    (meeting,) = SqliteStore(cli_env.data_dir / "praktika.db").list_meetings()
    assert meeting.language_mode is LanguageMode.en


# --------------------------------------------------------------------------- --organiser


def test_ingest_organiser_on_behalf_is_recorded_and_audited(cli_env: Settings) -> None:
    result = invoke(
        "ingest", str(VTT_EN), "--title", "Weekly", "--organiser", "R.Haddad@Acme.test",
        *GATE,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    store = SqliteStore(cli_env.data_dir / "praktika.db")
    (meeting,) = store.list_meetings()
    assert meeting.organiser == "r.haddad@acme.test"
    consent = store.get_consent(meeting.id)
    assert consent is not None and consent.recorded_by == "f.khalid@acme.test"
    (named,) = [e for e in audit_lines(cli_env) if e["event"] == "meeting.organiser_named"]
    assert named["meeting_id"] == meeting.id and named["actor"] == "f.khalid@acme.test"
    assert named["detail"]["operator"] == "f.khalid@acme.test"
    assert named["detail"]["organiser"] == "r.haddad@acme.test"
    assert named["detail"]["operator_source"] == "local"


def test_ingest_without_organiser_writes_no_on_behalf_event(cli_env: Settings) -> None:
    assert invoke("ingest", str(VTT_EN), "--title", "Weekly", *GATE).exit_code == 0
    (meeting,) = SqliteStore(cli_env.data_dir / "praktika.db").list_meetings()
    assert meeting.organiser == "f.khalid@acme.test"
    assert "meeting.organiser_named" not in [e["event"] for e in audit_lines(cli_env)]


@pytest.mark.parametrize("bad", ["root", "r.haddad", "r haddad@acme.test", "a@b"])
def test_ingest_organiser_shape_is_validated_before_anything_runs(
    cli_env: Settings, bad: str
) -> None:
    result = invoke("ingest", str(VTT_EN), "--title", "Weekly", "--organiser", bad, *GATE)
    assert result.exit_code == 1 and "UPN" in result.output
    assert (
        not (cli_env.data_dir / "praktika.db").exists()
        or not SqliteStore(cli_env.data_dir / "praktika.db").list_meetings()
    )
    assert audit_lines(cli_env) == []


def test_named_organiser_is_audited_before_the_meeting_is_stored(
    cli_env: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a meeting that names someone else as organiser is never stored without the
    ``meeting.organiser_named`` event, even when storing it fails straight after the gate."""

    def fail_save(self: SqliteStore, meeting: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(SqliteStore, "save_meeting", fail_save)
    result = invoke(
        "ingest", str(VTT_EN), "--title", "Weekly", "--organiser", "r.haddad@acme.test",
        *GATE,
    )  # fmt: skip
    assert result.exit_code != 0
    named = [e for e in audit_lines(cli_env) if e["event"] == "meeting.organiser_named"]
    assert len(named) == 1 and named[0]["detail"]["organiser"] == "r.haddad@acme.test"
