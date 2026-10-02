"""Operations commands: ``retention run|install``, ``hold set|clear``,
``dsar find|export|delete``.

``SqliteStore`` implements the ``retention.RetentionStore`` seam directly and ``purge_meeting``
for the DSAR route; these commands only wire them to the audit log and the console.
``retention install`` writes the hourly scheduler for ``retention run``: a launchd agent on
macOS, a systemd service and timer pair on Linux.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from praktika import retention
from praktika.cli import context as ctx
from praktika.cli.export_paths import resolve_export_target
from praktika.config import ENV_FILE_VAR, env_file_path
from praktika.errors import PraktikaError
from praktika.logging import get_logger

log = get_logger(__name__)

retention_app = typer.Typer(help="Retention job (timers per classification, legal hold).")
hold_app = typer.Typer(help="Legal hold: blocks every retention timer.")
dsar_app = typer.Typer(help="DPO route: locate, export or delete a participant's meetings.")


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@retention_app.command("run")
@ctx.guarded
def retention_run(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only.")] = False,
    now: Annotated[str | None, typer.Option("--now", help="ISO time to evaluate at.")] = None,
) -> None:
    """Delete artefacts whose retention timer has expired (audit receipts per item)."""
    rt = ctx.open_runtime(system_actor=True)
    at = _dt(now) or datetime.now(UTC)
    policy = retention.RetentionPolicy.from_settings(rt.settings)
    deletions = retention.run(
        rt.store,
        rt.audit,
        at,
        dry_run=dry_run,
        policy=policy,
        audio_root=Path(rt.settings.data_dir) / "audio",
    )
    verb = "Would delete" if dry_run else "Deleted"
    ctx.console.print(f"{verb} {len(deletions)} item(s) at {at.isoformat()}.")
    for d in deletions:
        ctx.console.print(f"  {d.meeting_id}  {d.kind:<10} {d.path or ''}  ({d.reason})")


LAUNCHD_LABEL = "local.praktika.retention"
LAUNCHD_INTERVAL_S = 3600
_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array><string>{executable}</string><string>retention</string><string>run</string></array>
  <key>StartInterval</key><integer>{interval}</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def launchd_plist(executable: Path, log_path: Path, interval_s: int = LAUNCHD_INTERVAL_S) -> str:
    """The launchd agent that runs ``praktika retention run`` every ``interval_s`` seconds."""
    return _PLIST.format(
        label=LAUNCHD_LABEL, executable=executable, interval=interval_s, log=log_path
    )


SYSTEMD_UNIT = "praktika-retention"
SYSTEMD_USER_UNIT_DIR = "~/.config/systemd/user"
XDG_CONFIG_HOME_VAR = "XDG_CONFIG_HOME"

_SERVICE = """[Unit]
Description=Praktika retention timers (C-05)

[Service]
Type=oneshot
{user_line}{env_file_line}Environment={data_dir_assignment}
ExecStart={exec_start}
UMask=0077
NoNewPrivileges=yes
"""

_TIMER = """[Unit]
Description=Run Praktika retention timers hourly (C-05)

[Timer]
OnCalendar=hourly
Persistent=true
Unit={unit}.service

[Install]
WantedBy=timers.target
"""


_ACCOUNT_RE = re.compile(r"[a-z_][a-z0-9_-]{0,31}")


def host_platform() -> str:
    """``sys.platform``, behind a function so tests can exercise either scheduler."""
    return sys.platform


def effective_uid() -> int:
    """``os.geteuid()``, behind a function so tests can exercise the root refusal."""
    return os.geteuid()


def _uid_of(account: str) -> int:
    """The uid of ``account``; ``PraktikaError`` when the name is not a plain account name or
    the host has no such account."""
    if not _ACCOUNT_RE.fullmatch(account):
        raise PraktikaError(f"not a valid account name for User=: {account!r}")
    try:
        return pwd.getpwnam(account).pw_uid
    except KeyError:
        raise PraktikaError(f"no account {account!r} on this host") from None


def _owner_of(path: Path) -> str | None:
    """The account that owns ``path``, or ``None`` when it does not exist or has no name."""
    try:
        return pwd.getpwuid(path.stat().st_uid).pw_name
    except (OSError, KeyError):
        return None


def _systemd_quote(value: str, *, command: bool = False) -> str:
    """Quote one systemd ``Environment=`` assignment or (``command=True``) ``ExecStart=`` word:
    double quotes, backslashes and double quotes escaped, ``%`` doubled (specifiers), and in a
    command ``$`` doubled too (``ExecStart=`` expands environment variables)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if command:
        escaped = escaped.replace("$", "$$")
    return f'"{escaped}"'


def systemd_exec_start(interpreter: Path) -> str:
    """The exact ``ExecStart`` line value: ``interpreter -m praktika.cli retention run``."""
    return " ".join(
        _systemd_quote(word, command=True)
        for word in (str(interpreter), "-m", "praktika.cli", "retention", "run")
    )


def systemd_units(
    interpreter: Path,
    env_file: Path | None,
    data_dir: Path,
    unit: str = SYSTEMD_UNIT,
    *,
    user: str | None = None,
) -> dict[str, str]:
    """The ``{filename: text}`` pair that runs ``praktika retention run`` hourly on Linux.

    ``user`` adds ``User=`` for a system unit (``/etc/systemd/system``), which would otherwise
    run as root; a user unit (``user=None``) always runs as the account whose user manager
    loads it, and ``User=`` is not allowed there.

    The service is a oneshot that runs ``interpreter -m praktika.cli retention run`` with
    ``PRAKTIKA_DATA_DIR`` pinned, and ``PRAKTIKA_ENV_FILE`` pinned when ``env_file`` is given,
    to the values the installer resolved, so the timer reads the same configuration and data
    directory as the operator did. ``env_file=None`` leaves ``PRAKTIKA_ENV_FILE`` out (there
    was no env file to pin: pinning a path that does not exist would make every run refuse).
    The timer fires on the hour (``OnCalendar=hourly``) and catches up after downtime
    (``Persistent=true``). Output goes to the journal.
    """
    env_file_line = (
        f"Environment={_systemd_quote(f'{ENV_FILE_VAR}={env_file}')}\n"
        if env_file is not None
        else ""
    )
    service = _SERVICE.format(
        user_line=f"User={user}\n" if user else "",
        env_file_line=env_file_line,
        data_dir_assignment=_systemd_quote(f"PRAKTIKA_DATA_DIR={data_dir}"),
        exec_start=systemd_exec_start(interpreter),
    )
    return {f"{unit}.service": service, f"{unit}.timer": _TIMER.format(unit=unit)}


def env_file_to_pin(environ: dict[str, str] | None = None) -> Path | None:
    """The env file a retention unit should pin, as an absolute path: the one
    ``PRAKTIKA_ENV_FILE`` names (checked by ``env_file_path``), or the default ``.env`` when it
    exists; ``None`` when neither, so the unit runs on the defaults and ``PRAKTIKA_DATA_DIR``
    instead of refusing every hour.

    A relative ``PRAKTIKA_ENV_FILE`` is made absolute against the installer's current
    directory (symbolic links are kept, not followed): the unit runs from ``/`` (system unit) or
    the account's home (user unit), where the relative name would find nothing and every run
    would refuse. A named file that does not exist, or is not a regular file, is refused here
    with ``PraktikaError`` (``env_file_path`` raises ``ConfigError`` first for most cases), so
    the install never reports success for a unit that cannot run.
    """
    env = os.environ if environ is None else environ
    path = env_file_path(environ=env).absolute()
    named = env.get(ENV_FILE_VAR)
    if named:
        if not path.is_file():
            raise PraktikaError(
                f"{ENV_FILE_VAR}={named} ({path}) is not a file; the retention timer would "
                f"refuse every run. Export {ENV_FILE_VAR} naming this deployment's env file "
                "and run this install again"
            )
        return path
    return path if path.is_file() else None


def systemd_user_unit_dir(environ: dict[str, str] | None = None) -> Path:
    """The systemd user unit directory: ``$XDG_CONFIG_HOME/systemd/user`` when that variable is
    an absolute path, else ``~/.config/systemd/user``."""
    env = os.environ if environ is None else environ
    xdg = env.get(XDG_CONFIG_HOME_VAR, "")
    if xdg and Path(xdg).expanduser().is_absolute():
        return Path(xdg).expanduser() / "systemd" / "user"
    return Path(SYSTEMD_USER_UNIT_DIR).expanduser()


def _write_0600(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(target, 0o600)


def _install_launchd(out: Path | None, data_dir: Path) -> None:
    executable = Path(shutil.which("praktika") or sys.argv[0]).resolve()
    target = out or Path("~/Library/LaunchAgents").expanduser() / f"{LAUNCHD_LABEL}.plist"
    _write_0600(target, launchd_plist(executable, data_dir / "retention.log"))
    log.info("retention.scheduler_installed", kind="launchd", path=str(target))
    ctx.console.print(f"Wrote {target}")
    ctx.console.print(f"Load it with: launchctl load -w {target}")


def _install_systemd(unit_dir: Path | None, data_dir: Path, run_as: str | None) -> None:
    """Write the unit pair; never one that runs as root (see ``retention_install``)."""
    user_dir = systemd_user_unit_dir()
    target_dir = (unit_dir or user_dir).expanduser()
    user_scope = target_dir.resolve() == user_dir.resolve()
    if user_scope:
        if run_as is not None:
            raise PraktikaError(
                "--run-as applies to a system unit directory (--unit-dir /etc/systemd/system); "
                "a user unit runs as the account that installs it"
            )
        if effective_uid() == 0:
            raise PraktikaError(
                "refusing to install a retention timer that runs as root: run the install as "
                "the service account, or pass --unit-dir /etc/systemd/system --run-as <account>"
            )
        account = None
    else:
        account = run_as or _owner_of(data_dir)
        if account is None:
            raise PraktikaError(
                f"cannot tell which account owns {data_dir}; pass --run-as <service account>"
            )
        if _uid_of(account) == 0:
            raise PraktikaError(
                f"refusing to install a retention timer that runs as root ({account}); set "
                "PRAKTIKA_DATA_DIR to the service account's data directory or pass --run-as"
            )
    env_file = env_file_to_pin()
    units = systemd_units(Path(sys.executable), env_file, data_dir, user=account)
    for name, text in units.items():
        _write_0600(target_dir / name, text)
        ctx.console.print(f"Wrote {target_dir / name}")
    if env_file is not None:
        ctx.console.print(f"Pinned {ENV_FILE_VAR}={env_file} in the service unit.")
    else:
        ctx.console.print(
            f"No env file pinned: {ENV_FILE_VAR} is not set and {env_file_path()} does not "
            f"exist, so the timer runs on the defaults with PRAKTIKA_DATA_DIR={data_dir}. To "
            f"pin one, export {ENV_FILE_VAR} and run this install again."
        )
    log.info(
        "retention.scheduler_installed", kind="systemd", path=str(target_dir), user=account or ""
    )
    scope = " --user" if user_scope else ""
    ctx.console.print(
        f"Enable it with: systemctl{scope} daemon-reload && "
        f"systemctl{scope} enable --now {SYSTEMD_UNIT}.timer"
    )
    if user_scope:
        ctx.console.print(
            "A user timer runs only while the account has a session; on a server keep it "
            "running with: loginctl enable-linger $(id -un)"
        )


#: True where the systemd units are written (Linux); the scheduler-agent option
#: of the other hosts is then hidden from ``--help``, and the systemd options are hidden there.
SYSTEMD_HOST = host_platform() != "darwin"

RETENTION_INSTALL_HELP = """\
Install the hourly scheduler for the retention timers (C-05); files are mode 0600.

On a systemd host (a Linux server) this writes praktika-retention.service and
praktika-retention.timer into --unit-dir. Hosts without systemd get a per-user scheduler agent
file (--out) instead. Options that do not apply to this host are refused rather than ignored.

The job never runs as root. The default is a user unit in the installing account's systemd
user directory (refused when that account is root). A unit written anywhere else is treated as
a system unit and gets User=: --run-as or, by default, the owner of the data directory; an
account with uid 0 is refused.

PRAKTIKA_DATA_DIR is pinned in the unit, and PRAKTIKA_ENV_FILE too when it is set (or the
default env file exists), so export PRAKTIKA_ENV_FILE before installing."""


@retention_app.command("install", help=RETENTION_INSTALL_HELP)
@ctx.guarded
def retention_install(
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Hosts without systemd only: the scheduler agent file "
            "(default: the account's per-user agent directory).",
            hidden=SYSTEMD_HOST,
        ),
    ] = None,
    unit_dir: Annotated[
        Path | None,
        typer.Option(
            "--unit-dir",
            help="Directory for the .service/.timer pair "
            "(default: the systemd user unit directory, ~/.config/systemd/user).",
            hidden=not SYSTEMD_HOST,
        ),
    ] = None,
    run_as: Annotated[
        str | None,
        typer.Option(
            "--run-as",
            help="System unit directory only: the account the service runs as "
            "(default: the owner of the data directory; root is refused).",
            hidden=not SYSTEMD_HOST,
        ),
    ] = None,
) -> None:
    """Install the hourly scheduler for the retention timers (C-05); files are mode 0600.

    Developer workstations (``darwin``) get a launchd agent (``--out``); Linux and every other
    platform get the systemd ``praktika-retention.service`` and ``.timer`` in ``--unit-dir``.
    The options that do not apply to this platform are refused rather than ignored, and hidden
    from ``--help``. The ``--help`` text is ``RETENTION_INSTALL_HELP``, which names no
    platform product, because it is read on Linux servers as well.

    The job never runs as root. The default is a user unit in the installing account's systemd
    user directory (refused when that account is root). A unit written anywhere else is treated
    as a system unit and gets ``User=``: ``--run-as`` or, by default, the owner of the data
    directory; an account with uid 0 is refused.
    """
    settings = ctx.load_settings()
    # Absolute, like the env file: the unit does not run from the installer's directory.
    data_dir = Path(settings.data_dir).absolute()
    if host_platform() == "darwin":
        if unit_dir is not None or run_as is not None:
            raise PraktikaError(
                "--unit-dir and --run-as apply to systemd hosts only; use --out on this host"
            )
        _install_launchd(out, data_dir)
        return
    if out is not None:
        raise PraktikaError("--out applies to hosts without systemd; use --unit-dir on this host")
    _install_systemd(unit_dir, data_dir, run_as)


def _hold(meeting_id: str, on: bool, reason: str) -> None:
    rt = ctx.open_runtime()
    meeting = rt.require_meeting(meeting_id)
    rt.store.set_hold(meeting_id, on, reason, rt.actor())
    rt.audit.append(
        "hold.set" if on else "hold.released",
        meeting_id,
        classification=meeting.classification.value,
        reason=reason,
    )
    ctx.console.print(f"Legal hold {'set on' if on else 'released for'} {meeting_id}.")


@hold_app.command("set")
@ctx.guarded
def hold_set(
    meeting_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason", help="Why the hold is placed.")],
) -> None:
    """Place a legal hold: no retention timer runs while it is active."""
    _hold(meeting_id, True, reason)


@hold_app.command("clear")
@ctx.guarded
def hold_clear(
    meeting_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason", help="Why the hold is released.")] = (
        "hold released"
    ),
) -> None:
    """Release a legal hold."""
    _hold(meeting_id, False, reason)


ParticipantOpt = Annotated[str, typer.Option("--participant", help="Name, alias or UPN.")]


@dsar_app.command("find")
@ctx.guarded
def dsar_find(participant: ParticipantOpt) -> None:
    """List meeting ids in which the participant appears (roster or transcript speakers)."""
    rt = ctx.open_runtime()
    ids = rt.store.dsar_find(participant)
    ctx.console.print("\n".join(ids) if ids else "No meetings found.")


@dsar_app.command("export")
@ctx.guarded
def dsar_export(
    participant: ParticipantOpt,
    out: Annotated[
        Path | None, typer.Option("--out", help="Bundle path (default: data_dir/exports/).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Allow a cloud-synced target.")] = False,
) -> None:
    """Write a JSON bundle (meeting, consent, redacted transcript, latest minutes) per meeting
    under ``data_dir/exports`` unless ``--out`` names a non-cloud-synced path."""
    rt = ctx.open_runtime()
    target = resolve_export_target(rt.settings, out, "dsar-export.json", force=force)
    ids = rt.store.dsar_find(participant)
    bundle: list[dict[str, Any]] = []
    for mid in ids:
        meeting = rt.store.get_meeting(mid)
        consent = rt.store.get_consent(mid)
        transcript = rt.store.get_transcript(mid)
        minutes = rt.store.latest_minutes(mid)
        bundle.append(
            {
                "meeting": meeting.model_dump(mode="json") if meeting else None,
                "consent": consent.model_dump(mode="json") if consent else None,
                "transcript": transcript.model_dump(mode="json") if transcript else None,
                "minutes": minutes.model_dump(mode="json") if minutes else None,
            }
        )
        rt.audit.append(
            "dsar.export",
            mid,
            classification=meeting.classification.value if meeting else None,
            participant_hash=_hash(participant),
        )
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2)
    os.chmod(target, 0o600)
    ctx.console.print(f"Exported {len(bundle)} meeting(s) to {target}")


@dsar_app.command("delete")
@ctx.guarded
def dsar_delete(
    participant: ParticipantOpt,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Erase every meeting the participant appears in.

    Legal holds are checked for every meeting before anything is purged: if any meeting is
    held the whole request is refused (exit 1) naming the held ids, so an erasure never stops
    half-way with some meetings gone and others untouched.
    """
    rt = ctx.open_runtime()
    ids = rt.store.dsar_find(participant)
    if not ids:
        ctx.console.print("No meetings found.")
        return
    meetings = {mid: rt.require_meeting(mid) for mid in ids}
    held = sorted(mid for mid, m in meetings.items() if m.legal_hold)
    if held:
        raise PraktikaError(
            f"{len(held)} of {len(ids)} meeting(s) under legal hold: {', '.join(held)}; "
            "deletion refused (nothing was erased)"
        )
    if not yes and not typer.confirm(f"Erase {len(ids)} meeting(s): {', '.join(ids)}?"):
        raise typer.Exit(ctx.EXIT_REFUSED)
    now = datetime.now(UTC)
    for mid in ids:
        meeting = meetings[mid]
        removed = rt.store.purge_meeting(mid, now)
        rt.audit.append(
            "dsar.delete",
            mid,
            classification=meeting.classification.value,
            files_removed=removed,
            participant_hash=_hash(participant),
        )
        ctx.console.print(f"Purged {mid} ({removed} audio file(s) removed).")


def _hash(value: str) -> str:
    """Short SHA-256 of a participant string so the audit log never carries the name itself."""
    import hashlib

    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()[:16]
